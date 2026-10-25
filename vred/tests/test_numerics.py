import copy
from dataclasses import fields
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from vred.configs import load_training_config
from vred.data import Batch, Collator, attach_online_features
from vred.effects import (EffectCapture, relative_pairs, stratified_write_positions,
                          subset_effects, write_positions)
from vred.losses import (effect_loss, effect_weighted_kd_sparse, floored_effect_weights,
                        logits_kd, logits_kd_sparse, semantic_probe, signed_distribution,
                        weighted_kd_sparse, write_loss)
from vred import losses
from vred.model import VREDModel
from vred.train import zero2_config
from vred.trainer import VREDTrainer, accumulation_groups


def tiny_model(hidden=24, heads=3, kv=1, depth=6, vision_tokens=3, vision_width=8, vocab=97):
    config = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
                         num_hidden_layers=depth, num_attention_heads=heads, num_key_value_heads=kv,
                         tie_word_embeddings=True, attention_dropout=0.0, use_sliding_window=False)
    config._attn_implementation = "eager"
    return VREDModel(Qwen2ForCausalLM(config), vision_width, vision_tokens)


def tiny_batch(vision_tokens=3, vision_width=8):
    class Tokenizer:
        pad_token_id = 0
    length = vision_tokens + 7
    image_ids = torch.arange(1, length + 1) % 96
    image_ids[1:1 + vision_tokens] = 0
    labels = torch.full_like(image_ids, -100)
    labels[-3:] = image_ids[-3:]
    return Collator(Tokenizer())([
        dict(input_ids=torch.tensor([11, 12, 13, 0]), labels=torch.tensor([-100, -100, 13, 0]),
             feature=None, visual_span=(-1, -1), record_id="text-eos-equals-pad"),
        dict(input_ids=image_ids, labels=labels, feature=torch.randn(vision_tokens, vision_width),
             visual_span=(1, 1 + vision_tokens), record_id="cached-image"),
    ])


class Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(1234)

    def test_relative_depth_and_shifted_sampling(self):
        self.assertEqual(relative_pairs(24, 36, 3), [(7, 11), (15, 23), (23, 35)])
        self.assertEqual(relative_pairs(24, 36, 6), [(3, 5), (7, 11), (11, 17), (15, 23), (19, 29), (23, 35)])
        batch = tiny_batch()
        torch.testing.assert_close(batch.response_positions[:2], torch.tensor([[0, 1], [0, 2]]))
        self.assertTrue(batch.attention_mask[0, 3])  # Real EOS=PAD remains valid.
        self.assertFalse(batch.attention_mask[0, 4:].any())
        positions = write_positions(batch)
        self.assertTrue((positions[:, 0] == 1).all())
        self.assertEqual(positions[0, 1], len(batch.input_ids[1]) - 4)
        a = write_positions(batch, 2, torch.Generator().manual_seed(9))
        b = write_positions(batch, 2, torch.Generator().manual_seed(9))
        torch.testing.assert_close(a, b)
        self.assertEqual(len(a), 2)
        self.assertTrue(batch.causal_mask[1, 0, a[0, 1], 1:4].all())
        broken = copy.deepcopy(batch)
        broken.visual_spans[1] = torch.tensor([6, 100])
        self.assertEqual(len(write_positions(broken)), 0)

    def test_write_scope_prompt_response_and_stratified_sampling(self):
        batch = tiny_batch()
        response = write_positions(batch)
        prompt = write_positions(batch, scope="prompt")
        both = write_positions(batch, scope="prompt_response")
        self.assertGreater(len(prompt), 0)
        self.assertEqual(len(both), len(response) + len(prompt))
        rows, columns = prompt.unbind(1)
        self.assertFalse(batch.response_mask[rows, columns].any())
        self.assertTrue(batch.attention_mask[rows, columns].all())
        self.assertTrue((columns >= batch.visual_spans[rows, 1]).all())
        self.assertTrue(set(map(tuple, response.tolist())) <= set(map(tuple, both.tolist())))
        self.assertTrue(set(map(tuple, prompt.tolist())) <= set(map(tuple, both.tolist())))
        broken = copy.deepcopy(batch)
        broken.visual_spans[1] = torch.tensor([6, 100])
        self.assertEqual(len(write_positions(broken, scope="prompt")), 0)
        self.assertEqual(len(write_positions(broken, scope="prompt_response")), 0)

        student, teacher = tiny_model(), tiny_model(hidden=32, heads=4, kv=2)
        trainer = VREDTrainer(student, teacher, "dft", load_training_config())
        with torch.no_grad(), EffectCapture(teacher, [t for _, t in trainer.pairs], batch, prompt) as target:
            teacher(batch)
        positions, selected = stratified_write_positions(
            batch, target.effects, trainer.pairs, 2, torch.Generator().manual_seed(7), scope="prompt")
        self.assertEqual(len(positions), len(selected))
        counts = torch.bincount(positions[:, 0], minlength=len(batch.record_ids))
        self.assertLessEqual(int(counts.max()), 2)
        subset = subset_effects(target.effects, selected)
        for effect in subset.values():
            self.assertEqual(len(effect.positions), len(selected))
            torch.testing.assert_close(effect.positions, positions)

    def test_trainer_scope_and_stratified_writes(self):
        batch = tiny_batch()
        config = load_training_config()
        config.update(write_position_scope="prompt", write_effect_sampling="stratified",
                      write_position_limit=2)
        student, teacher = tiny_model(), tiny_model(hidden=32, heads=4, kv=2)
        trainer = VREDTrainer(student, teacher, "dft", config)
        loss, metrics = trainer.loss(batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLessEqual(metrics["write_positions"], 2 * len(batch.record_ids))
        self.assertGreater(float(metrics["write"]), 0)
        self.assertGreater(float(student.connector[0].weight.grad.abs().sum()), 0)

    def test_write_lambda_schedule_and_weighted_write(self):
        batch = tiny_batch()
        config = load_training_config()
        config.update(write_lambda_schedule="warmup_cosine", warmup_ratio=0.1)
        student, teacher = tiny_model(), tiny_model(hidden=32, heads=4, kv=2)
        trainer = VREDTrainer(student, teacher, "dft", config)
        trainer.set_write_schedule(100)
        expected = {0: 0.0, 5: 0.5, 10: 1.0, 100: 0.0}
        for step, value in expected.items():
            trainer.set_write_step(step)
            self.assertAlmostEqual(trainer.write_lambda_scale(), value, places=7)
        trainer.set_write_step(50)
        self.assertAlmostEqual(trainer.write_lambda_scale(),
                               0.5 * (1.0 + math.cos(math.pi * 0.4 / 0.9)), places=7)
        constant = load_training_config()
        trainer = VREDTrainer(student, teacher, "dft", constant)
        self.assertEqual(trainer.write_lambda_scale(), 1.0)
        config["stages"]["dft"]["lambda_write"] = 1.0
        trainer = VREDTrainer(student, teacher, "dft", config)
        trainer.set_write_schedule(100)
        trainer.set_write_step(0)
        loss, metrics = trainer.loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["write_lambda_scale"], 0.0)
        self.assertEqual(float(metrics["write_weighted"]), 0.0)
        self.assertGreater(float(metrics["write"]), 0.0)

    def test_actual_attention_rope_gqa_bias_and_forward_identity(self):
        model, batch = tiny_model(depth=2).eval(), tiny_batch()
        layer = model.language_model.model.layers[1]
        layer.self_attn.o_proj.bias = nn.Parameter(torch.randn(24) * 0.03)
        observed = {}

        def before(module, args, kwargs):
            observed.update(kwargs)

        def after(module, args, output):
            observed["attention_output"], observed["probabilities"] = output[:2]

        hooks = [layer.self_attn.register_forward_pre_hook(before, with_kwargs=True),
                 layer.self_attn.register_forward_hook(after)]
        reference = model(batch)
        with EffectCapture(model, [1], batch) as capture:
            actual = model(batch)
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(actual.logits, reference.logits, atol=0, rtol=0)
        torch.testing.assert_close(actual.loss, reference.loss, atol=0, rtol=0)
        effect = capture.effects[1]
        attn = layer.self_attn
        x = observed["hidden_states"]
        shape = (*x.shape[:-1], -1, attn.head_dim)
        q = attn.q_proj(x).view(shape).transpose(1, 2)
        k = attn.k_proj(x).view(shape).transpose(1, 2)
        v = attn.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *observed["position_embeddings"])
        k, v = repeat_kv(k, attn.num_key_value_groups), repeat_kv(v, attn.num_key_value_groups)
        scaling = getattr(attn, "scaling", attn.head_dim ** -0.5)
        probabilities = ((q @ k.transpose(-1, -2)) * scaling + observed["attention_mask"]).float().softmax(-1)
        torch.testing.assert_close(observed["probabilities"], probabilities, atol=1e-7, rtol=1e-6)
        self.assertEqual(probabilities[0, :, 2, 4:].abs().sum(), 0)
        self.assertEqual(probabilities[1, :, 2, 3:].abs().sum(), 0)
        total, visual, text = [], [], []
        for row, query in effect.positions.tolist():
            mask = torch.zeros(v.shape[-2])
            mask[slice(*batch.visual_spans[row].tolist())] = 1
            weights = probabilities[row, :, query]
            def project(weights, bias):
                heads = (weights[:, None, :] @ v[row]).reshape(-1)
                return F.linear(heads, attn.o_proj.weight, bias)
            total.append(project(weights, attn.o_proj.bias))
            visual.append(project(weights * mask, None))
            text.append(project(weights * (1 - mask), attn.o_proj.bias))
        torch.testing.assert_close(effect.o_total, torch.stack(total), atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(effect.o_vision, torch.stack(visual), atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(effect.o_text, torch.stack(text), atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(effect.o_total, effect.o_text + effect.o_vision, atol=1e-7, rtol=1e-5)
        baseline = effect.incoming + effect.o_text
        plus = baseline + effect.o_vision
        reconstructed = plus + layer.mlp(layer.post_attention_layernorm(plus))
        torch.testing.assert_close(effect.u_plus, reconstructed, atol=1e-7, rtol=1e-5)
        # A visual-only softmax is a different computation and must disagree.
        row, query = effect.positions[0].tolist()
        visual_only = probabilities[row, :, query, 1:4]
        self.assertFalse(torch.allclose(visual_only.sum(-1), torch.ones(3)))
        self.assertEqual(len(layer._forward_hooks), 0)

    def test_legacy_attention_selected_flags_and_cleanup(self):
        model, batch = tiny_model(depth=2).eval(), tiny_batch()
        attention = model.language_model.model.layers[1].self_attn
        original_forward = attention.forward
        flags = []

        def legacy_forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                           output_attentions=False, **kwargs):
            flags.append(output_attentions)
            output = original_forward(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                      past_key_value=past_key_value, output_attentions=output_attentions, **kwargs)
            return (output[0], output[1] if output_attentions else None, *output[2:])

        layer_flags = []

        def observe(module, args, kwargs):
            layer_flags.append(kwargs.get("output_attentions", False))

        observer = model.language_model.model.layers[0].self_attn.register_forward_pre_hook(observe, with_kwargs=True)
        reference = model(batch)
        with patch.object(attention, "forward", legacy_forward):
            with EffectCapture(model, [1], batch) as capture:
                actual = model(batch)
            self.assertEqual(len(capture.effects), 1)
            self.assertEqual(flags, [True])
            hook = EffectCapture._request_attention(attention)
            args, kwargs = hook(attention, (torch.zeros(1), None, None, None, False), {})
            self.assertIs(args[4], True)
            self.assertNotIn("output_attentions", kwargs)
        observer.remove()
        self.assertEqual(layer_flags, [False, False])
        self.assertIsNone(actual.attentions)
        torch.testing.assert_close(actual.logits, reference.logits, atol=0, rtol=0)
        self.assertEqual(len(attention._forward_pre_hooks), 0)
        self.assertEqual(len(attention._forward_hooks), 0)
        self.assertEqual(len(attention.v_proj._forward_hooks), 0)

    def test_dropout_uses_original_draw_without_modifying_forward(self):
        model, batch = tiny_model(depth=1), tiny_batch()
        model.language_model.model.layers[0].self_attn.attention_dropout = 0.25
        torch.manual_seed(42)
        reference = model(batch).logits
        torch.manual_seed(42)
        with EffectCapture(model, [0], batch) as capture:
            output = model(batch)
        torch.testing.assert_close(output.logits, reference, atol=0, rtol=0)
        record = capture.effects[0]
        torch.testing.assert_close(record.o_text + record.o_vision, record.o_total, atol=1e-7, rtol=1e-5)

    def test_zero_visual_message_and_detached_probe(self):
        model, batch = tiny_model(depth=1), tiny_batch()
        attention = model.language_model.model.layers[0].self_attn
        with torch.no_grad():
            attention.v_proj.weight.zero_()
            attention.v_proj.bias.zero_()
        with EffectCapture(model, [0], batch) as capture:
            model(batch)
        effect = capture.effects[0]
        self.assertEqual(effect.o_vision.abs().sum(), 0)
        torch.testing.assert_close(effect.u_plus, effect.u_minus, atol=0, rtol=0)
        effect_value = semantic_probe(model, effect.u_plus) - semantic_probe(model, effect.u_minus)
        self.assertEqual(effect_value.abs().sum(), 0)
        encoded = signed_distribution(effect_value)
        self.assertTrue((encoded[:, -1] == 1).all())
        self.assertEqual(encoded[:, :-1].sum(), 0)
        plus, minus = torch.randn(3, 24, requires_grad=True), torch.randn(3, 24, requires_grad=True)
        semantic_probe(model, plus)[:, :5].sum().backward()
        self.assertGreater(float(plus.grad.abs().sum()), 0)
        self.assertIsNone(model.language_model.model.norm.weight.grad)
        self.assertIsNone(model.language_model.lm_head.weight.grad)
        self.assertTrue(model.language_model.lm_head.weight.requires_grad)
        expected = model.language_model.lm_head(model.language_model.model.norm(minus)).softmax(-1)
        torch.testing.assert_close(semantic_probe(model, minus), expected, atol=1e-7, rtol=1e-5)
        self.assertIs(model.language_model.lm_head.weight, model.language_model.model.embed_tokens.weight)

    def test_losses_toy_reductions_smoothing_signs_and_zeros(self):
        dt = torch.tensor([[-0.250523, 0.258740, -0.008217]], requires_grad=True)
        ds = torch.tensor([[-0.121437, 0.129374, -0.007937]], requires_grad=True)
        qs = signed_distribution(ds)
        torch.testing.assert_close(qs.sum(-1), torch.ones(1))
        torch.testing.assert_close(2 * (qs[:, :3] - qs[:, 3:6]), ds)
        loss, raw = effect_loss(ds, dt)
        self.assertAlmostEqual(float(raw.detach()), 0.00556644, places=7)
        torch.testing.assert_close(loss, raw * 3)
        mean_loss, _ = effect_loss(ds, dt, vocab_reduction="mean")
        torch.testing.assert_close(mean_loss, raw)
        for kind in ("smoothl1", "effect_kl", "effect_js", "reverse_kl"):
            loss, _ = effect_loss(ds, dt, kind)
            loss.backward(retain_graph=True)
            self.assertIsNone(dt.grad)
            self.assertTrue(torch.isfinite(ds.grad).all())
            self.assertGreater(float(ds.grad.abs().sum()), 0)
            ds.grad = None
            same, _ = effect_loss(dt, dt, kind)
            self.assertLess(abs(float(same.detach())), 1e-7)
            for value in (torch.zeros_like(ds), -dt.detach(), torch.tensor([[0.01, -0.01, 0.0]])):
                value = value.requires_grad_()
                result, _ = effect_loss(value, dt, kind)
                result.backward()
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(float(value.grad.abs().sum()), 0)
        kl, _ = effect_loss(ds, dt, "effect_kl")
        js, _ = effect_loss(ds, dt, "effect_js")
        self.assertAlmostEqual(float(kl.detach()), 0.061279, places=5)
        self.assertAlmostEqual(float(js.detach()), 0.013812, places=5)
        for kind in ("effect_kl", "effect_js"):
            zero = torch.zeros(2, 3, requires_grad=True)
            loss, _ = effect_loss(zero, zero, kind)
            loss.backward()
            self.assertEqual(float(loss.detach()), 0)
            self.assertTrue(torch.isfinite(zero.grad).all())
        with self.assertRaisesRegex(ValueError, "difference"):
            signed_distribution(torch.tensor([[2.0, -2.0]]))

    def test_full_final_kl_uses_shifted_labels_and_detaches_teacher(self):
        batch = tiny_batch()
        shape = (*batch.input_ids.shape, 97)
        student, teacher = torch.randn(shape, requires_grad=True), torch.randn(shape, requires_grad=True)
        loss = logits_kd(student, teacher, batch, temperature=2, chunk_size=2)
        rows, queries = batch.response_positions.unbind(1)
        expected = F.kl_div((student[rows, queries] / 2).log_softmax(-1),
                            (teacher[rows, queries].detach() / 2).softmax(-1), reduction="batchmean") * 4
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[~batch.response_mask].abs().sum(), 0)
        self.assertGreater(float(student.grad[0, 1].abs().sum()), 0)

    def test_write_gradients_and_checkpointed_chunk_agreement(self):
        batch = tiny_batch()
        for stage in ("dpt", "dft"):
            student, teacher = tiny_model(), tiny_model(hidden=32, heads=4, kv=2)
            config = load_training_config()
            config["query_chunk_size"] = 2
            trainer = VREDTrainer(student, teacher, stage, config)
            positions = write_positions(batch)
            with torch.no_grad(), EffectCapture(teacher, [t for _, t in trainer.pairs], batch, positions) as tc:
                teacher(batch)
            with EffectCapture(student, [s for s, _ in trainer.pairs], batch, positions) as sc:
                student(batch)
            target = student.connector[0].weight if stage == "dpt" else student.language_model.model.layers[0].mlp.up_proj.weight
            for kind in ("smoothl1", "effect_kl", "effect_js", "reverse_kl"):
                loss, raw = write_loss(student, teacher, sc.effects, tc.effects, trainer.pairs,
                                       chunk_size=2, kind=kind, checkpoint_probes=False)
                chunked, _ = write_loss(student, teacher, sc.effects, tc.effects, trainer.pairs,
                                        chunk_size=2, kind=kind, checkpoint_probes=False,
                                        vocab_chunk_size=7)
                torch.testing.assert_close(chunked, loss, atol=2e-7, rtol=2e-5)
                grad = torch.autograd.grad(loss, target, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().sum()), 0, (stage, kind))
                recomputed, recomputed_raw = write_loss(student, teacher, sc.effects, tc.effects, trainer.pairs,
                                                         chunk_size=2, kind=kind, checkpoint_probes=True)
                recomputed_grad = torch.autograd.grad(recomputed, target, retain_graph=True)[0]
                torch.testing.assert_close(recomputed, loss, atol=2e-7, rtol=2e-5)
                torch.testing.assert_close(recomputed_raw, raw, atol=2e-7, rtol=2e-5)
                torch.testing.assert_close(recomputed_grad, grad, atol=1e-7, rtol=2e-4)
                smaller, _ = write_loss(student, teacher, sc.effects, tc.effects, trainer.pairs,
                                         chunk_size=1, kind=kind, checkpoint_probes=True)
                torch.testing.assert_close(smaller, loss, atol=1e-6, rtol=2e-5)
                smaller_grad = torch.autograd.grad(smaller, target, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(smaller_grad).all())
                # KL near the positive/negative kink is sensitive to GEMM rounding
                # when a different query batch size changes cancellation at zero.
                if kind not in ("effect_kl", "reverse_kl"):
                    torch.testing.assert_close(smaller_grad, grad, atol=1e-7, rtol=2e-4)
            loss, metrics = trainer.loss(batch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertGreater(float(student.connector[0].weight.grad.abs().sum()), 0)
            if stage == "dpt":
                self.assertTrue(all(p.grad is None for p in student.language_model.parameters()))
                self.assertEqual(float(metrics["logits_kd"]), 0)
            else:
                self.assertGreater(float(target.grad.abs().sum()), 0)
                self.assertGreater(float(student.language_model.lm_head.weight.grad.abs().sum()), 0)

    def test_probe_parameters_reused_through_backward_and_refreshed_per_loss(self):
        batch = tiny_batch()
        for stage in ("dpt", "dft"):
            for kind in ("smoothl1", "effect_kl", "effect_js", "reverse_kl"):
                student = tiny_model().to(torch.bfloat16)
                teacher = tiny_model(hidden=32, heads=4, kv=2).to(torch.bfloat16)
                config = load_training_config()
                config.update(query_chunk_size=1, write_loss_type=kind)
                trainer = VREDTrainer(student, teacher, stage, config)
                original_probe = losses.semantic_probe

                def uncached_probe(model, states, temperature=1.0, parameters=None,
                                   vocab_chunk_size=16384):
                    return original_probe(model, states, temperature, None, vocab_chunk_size)

                with patch("vred.losses.semantic_probe", side_effect=uncached_probe):
                    reference, reference_metrics = trainer.loss(batch)
                    reference.backward()
                gradients = {name: parameter.grad.clone() for name, parameter in student.named_parameters()
                             if parameter.grad is not None}
                student.zero_grad(set_to_none=True)
                references = []
                original_parameters = losses._probe_parameters

                def record_parameters(model):
                    parameters = original_parameters(model)
                    self.assertTrue(all(not tensor.requires_grad for tensor in parameters
                                        if isinstance(tensor, torch.Tensor)))
                    references.append(weakref.ref(parameters[2]))
                    return parameters

                with patch("vred.losses._probe_parameters", side_effect=record_parameters) as prepare:
                    actual, metrics = trainer.loss(batch)
                    actual.backward()
                    self.assertEqual(prepare.call_count, 2)
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                for name in reference_metrics:
                    self.assertEqual(float(metrics[name]), float(reference_metrics[name]))
                for name, parameter in student.named_parameters():
                    if name in gradients:
                        torch.testing.assert_close(parameter.grad, gradients[name], atol=0, rtol=0)
                self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
                del actual
                self.assertTrue(all(reference() is None for reference in references))
                with torch.no_grad():
                    student.language_model.lm_head.weight.add_(0.25)
                with patch("vred.losses._probe_parameters", wraps=original_parameters) as prepare:
                    refreshed, _ = trainer.loss(batch)
                    refreshed.backward()
                    self.assertEqual(prepare.call_count, 2)
                with patch("vred.losses.semantic_probe", side_effect=uncached_probe):
                    expected, _ = trainer.loss(batch)
                torch.testing.assert_close(refreshed, expected, atol=0, rtol=0)

    def test_dpt_backbone_targets_match_full_teacher_without_head_or_ce(self):
        batch = tiny_batch()
        for dtype in (torch.float32, torch.bfloat16):
            student = tiny_model().to(dtype)
            teacher = tiny_model(hidden=32, heads=4, kv=2).to(dtype)
            trainer = VREDTrainer(student, teacher, "dpt", load_training_config())
            blocks = [teacher_index for _, teacher_index in trainer.pairs]
            with torch.no_grad(), EffectCapture(teacher, blocks, batch) as full:
                teacher(batch)
            with torch.no_grad(), EffectCapture(teacher, blocks, batch) as backbone:
                teacher.forward_backbone(batch)
            for block in blocks:
                for field in fields(full.effects[block]):
                    torch.testing.assert_close(getattr(full.effects[block], field.name),
                                               getattr(backbone.effects[block], field.name), atol=0, rtol=0)
            with patch.object(teacher, "forward_backbone", side_effect=lambda batch: teacher(batch)):
                reference, reference_metrics = trainer.loss(batch)
                reference.backward()
            gradient = student.connector[0].weight.grad.clone()
            student.zero_grad(set_to_none=True)
            with patch.object(teacher.language_model, "forward", side_effect=AssertionError("teacher LM head/CE")), \
                    patch.object(teacher.language_model.lm_head, "forward", side_effect=AssertionError("teacher head")):
                actual, metrics = trainer.loss(batch)
                actual.backward()
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            torch.testing.assert_close(student.connector[0].weight.grad, gradient, atol=0, rtol=0)
            for name in reference_metrics:
                self.assertEqual(float(metrics[name]), float(reference_metrics[name]))
            trainer = VREDTrainer(student, teacher, "dft", load_training_config())
            with patch.object(teacher, "forward_backbone", side_effect=AssertionError("DFT needs logits")), \
                    patch.object(teacher.language_model.lm_head, "forward",
                                 wraps=teacher.language_model.lm_head.forward) as head:
                loss, _ = trainer.loss(batch)
                loss.backward()
                self.assertEqual(head.call_count, 1)

    def test_sparse_ce_and_final_kd_match_dense(self):
        batch = tiny_batch()
        rows, columns = batch.response_positions.unbind(1)
        for dtype in (torch.float32, torch.bfloat16):
            student = tiny_model().to(dtype)
            dense = student(batch)
            sparse = student.forward_sparse(batch)
            tolerance = dict(atol=1e-6, rtol=1e-5) if dtype == torch.float32 else dict(atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(sparse.loss, dense.loss, **tolerance)
            torch.testing.assert_close(sparse.logits.float(), dense.logits[rows, columns].float(), **tolerance)
            target = student.connector[0].weight
            dense_grad = torch.autograd.grad(dense.loss, target, retain_graph=True)[0]
            sparse_grad = torch.autograd.grad(sparse.loss, target, retain_graph=True)[0]
            torch.testing.assert_close(sparse_grad, dense_grad, **tolerance)
            for ce_chunk in (1, 7, 32):
                chunked = student.forward_sparse(batch, return_logits=False, ce_chunk_size=ce_chunk)
                self.assertIsNone(chunked.logits)
                torch.testing.assert_close(chunked.loss, dense.loss, **tolerance)
                chunk_grad = torch.autograd.grad(chunked.loss, target, retain_graph=True)[0]
                torch.testing.assert_close(chunk_grad, dense_grad, **tolerance)
            teacher = tiny_model(hidden=32, heads=4, kv=2).to(dtype)
            with torch.no_grad():
                teacher_logits = teacher(batch).logits
            dense_kd = logits_kd(dense.logits, teacher_logits, batch)
            sparse_kd = logits_kd_sparse(sparse.logits, teacher_logits[rows, columns])
            torch.testing.assert_close(sparse_kd, dense_kd, **tolerance)
        empty = copy.deepcopy(batch)
        empty.response_mask.zero_()
        empty_out = tiny_model().forward_sparse(empty)
        self.assertEqual(float(empty_out.loss), 0)
        self.assertEqual(len(empty_out.logits), 0)
        self.assertTrue(empty_out.loss.requires_grad)

    def test_effect_weighted_kd_formula_and_zero_effect(self):
        torch.manual_seed(11)
        student = torch.randn(5, 17, requires_grad=True)
        teacher = torch.randn(5, 17)
        teacher_minus = teacher + torch.randn(5, 17) * 0.2
        temperature, chunk = 2.0, 2
        loss = effect_weighted_kd_sparse(student, teacher, teacher_minus, temperature, chunk)
        log_s = (student.float() / temperature).log_softmax(-1)
        log_t = (teacher.float() / temperature).log_softmax(-1)
        log_m = (teacher_minus.float() / temperature).log_softmax(-1)
        p_t, p_m = log_t.exp(), log_m.exp()
        weights = (p_t - p_m).abs().sum(-1)
        expected = (weights * (p_t * (log_t - log_s)).sum(-1)).mean() * temperature**2
        torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0)
        no_effect = effect_weighted_kd_sparse(student, teacher, teacher, temperature, chunk)
        torch.testing.assert_close(no_effect, torch.zeros(()), atol=0, rtol=0)

    def test_floored_effect_weights_and_weighted_kd(self):
        scores = torch.tensor([0.0, 1.0, 4.0])
        floor = 0.1
        weights = floored_effect_weights(scores, floor)
        expected = floor + (1.0 - floor) * scores / scores.mean()
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
        torch.testing.assert_close(floored_effect_weights(torch.zeros(4), floor),
                                   torch.full((4,), floor))
        with self.assertRaises(ValueError):
            floored_effect_weights(scores, 1.0)

        torch.manual_seed(19)
        student = torch.randn(3, 13, requires_grad=True)
        teacher = torch.randn(3, 13)
        temperature, chunk = 2.0, 2
        loss = weighted_kd_sparse(student, teacher, weights, temperature, chunk)
        log_s = (student.float() / temperature).log_softmax(-1)
        log_t = (teacher.float() / temperature).log_softmax(-1)
        terms = (log_t.exp() * (log_t - log_s)).sum(-1)
        expected_loss = (weights * terms).mean() * temperature**2
        torch.testing.assert_close(loss, expected_loss, atol=1e-6, rtol=1e-5)
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0)
        zero = weighted_kd_sparse(student[:0], teacher[:0], weights[:0], temperature, chunk)
        torch.testing.assert_close(zero, torch.zeros(()))

    def test_trainer_effect_weighted_kd_opt_in(self):
        batch = tiny_batch()
        config = load_training_config()
        student, teacher = tiny_model(), tiny_model(hidden=32, heads=4, kv=2)
        off = VREDTrainer(student, teacher, "dft", config)
        _, metrics = off.loss(batch)
        self.assertNotIn("effect_kd", metrics)
        config.update(effect_weighted_kd=True, lambda_effect_kd=2.0)
        student, teacher = tiny_model(), tiny_model(hidden=32, heads=4, kv=2)
        trainer = VREDTrainer(student, teacher, "dft", config)
        loss, metrics = trainer.loss(batch)
        self.assertIn("effect_kd", metrics)
        self.assertIn("effect_kd_weighted", metrics)
        self.assertGreater(float(metrics["effect_kd"]), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(student.connector[0].weight.grad.abs().sum()), 0)

    def test_trainer_hidden_effect_kd_opt_in_without_teacher_minus(self):
        batch = tiny_batch()
        config = load_training_config()
        config.update(effect_weighted_kd=True, effect_kd_weight_source="hidden",
                      effect_kd_floor=0.15, lambda_effect_kd=0.5)
        config["stages"]["dft"]["lambda_logits"] = 0.0
        student, teacher = tiny_model(), tiny_model(hidden=32, heads=4, kv=2)
        trainer = VREDTrainer(student, teacher, "dft", config)
        with patch.object(teacher, "forward_sparse", wraps=teacher.forward_sparse) as teacher_forward:
            loss, metrics = trainer.loss(batch)
        self.assertEqual(teacher_forward.call_count, 1)
        self.assertIn("effect_kd", metrics)
        self.assertIn("effect_kd_weighted", metrics)
        self.assertGreater(float(metrics["effect_kd"]), 0)
        self.assertEqual(float(metrics["logits_kd"]), 0.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(student.connector[0].weight.grad.abs().sum()), 0)

    def test_sparse_trainer_uses_wrapped_module_forward(self):
        class Engine:
            def __init__(self, module):
                self.module = module
                self.kwargs = None
            def __call__(self, *args, **kwargs):
                self.kwargs = kwargs
                return self.module(*args, **kwargs)
        batch = tiny_batch()
        student, teacher = tiny_model(), tiny_model(hidden=32, heads=4, kv=2)
        trainer = VREDTrainer(student, teacher, "dft", load_training_config())
        engine = Engine(student)
        loss, _ = trainer.loss(batch, forward_model=engine)
        loss.backward()
        self.assertEqual(engine.kwargs["sparse"], True)
        self.assertEqual(engine.kwargs["return_logits"], True)
        self.assertEqual(engine.kwargs["ce_chunk_size"], trainer.config["ce_chunk_size"])
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(student.connector[0].weight.grad.abs().sum()), 0)

    def test_selective_checkpointing_matches_ordinary_forward(self):
        batch = tiny_batch()
        for stage in ("dpt", "dft"):
            student = tiny_model(depth=12)
            teacher = tiny_model(hidden=32, heads=4, kv=2, depth=12)
            initial_student = copy.deepcopy(student.state_dict())
            initial_teacher = copy.deepcopy(teacher.state_dict())
            config = load_training_config()
            config["selective_checkpointing"] = False
            reference = VREDTrainer(student, teacher, stage, config, seed=1234)
            reference_loss, _ = reference.loss(batch)
            targets = [student.connector[0].weight]
            if stage == "dft":
                targets.append(student.language_model.model.layers[0].mlp.up_proj.weight)
            reference_grads = torch.autograd.grad(reference_loss, targets, retain_graph=False)
            student.load_state_dict(initial_student)
            teacher.load_state_dict(initial_teacher)
            config["selective_checkpointing"] = True
            selective = VREDTrainer(student, teacher, stage, config, seed=1234)
            with patch("vred.trainer.activation_checkpoint", wraps=torch_checkpoint) as checkpoints:
                selective_loss, selective_metrics = selective.loss(batch)
                self.assertGreater(checkpoints.call_count, 0)
            selective_grads = torch.autograd.grad(selective_loss, targets, retain_graph=False)
            torch.testing.assert_close(selective_loss, reference_loss, atol=1e-6, rtol=1e-5)
            for actual, expected in zip(selective_grads, reference_grads):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_chunked_fp32_probe_matches_full_head_reference(self):
        model = tiny_model()
        states = torch.randn(4, 24)
        norm = model.language_model.model.norm
        head = model.language_model.get_output_embeddings()
        head_full = head.weight.detach().float()
        bias_full = head.bias.detach().float() if head.bias is not None else None

        def reference(x):
            x = x.float()
            normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.variance_epsilon)
            normalized = normalized * norm.weight.detach().float()
            return F.linear(normalized, head_full, bias_full).softmax(-1)

        expected = reference(states)
        for chunk_size in (1, 7, 97, 16384):
            actual = semantic_probe(model, states, vocab_chunk_size=chunk_size)
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        chunked, whole = states.clone().requires_grad_(), states.clone().requires_grad_()
        semantic_probe(model, chunked, vocab_chunk_size=7)[:, :5].sum().backward()
        semantic_probe(model, whole, vocab_chunk_size=16384)[:, :5].sum().backward()
        torch.testing.assert_close(chunked.grad, whole.grad, atol=1e-7, rtol=1e-5)

    def test_sft_text_only_dpt_and_error_cleanup(self):
        student, batch = tiny_model(), tiny_batch()
        trainer = VREDTrainer(student, None, "sft", load_training_config())
        loss, metrics = trainer.loss(batch)
        loss.backward()
        self.assertEqual(float(metrics["write"]), 0)
        self.assertEqual(float(metrics["logits_kd"]), 0)
        teacher = tiny_model(hidden=32, heads=4, kv=2)
        trainer = VREDTrainer(student, teacher, "dpt", load_training_config())
        batch.visual_spans.fill_(-1)
        batch.features = (None, None)
        batch.write_mask.zero_()
        student.zero_grad(set_to_none=True)
        loss, metrics = trainer.loss(batch)
        loss.backward()
        self.assertEqual(metrics["write_positions"], 0)
        self.assertEqual(float(metrics["write"]), 0)
        self.assertTrue(all(p.grad is None or p.grad.abs().sum() == 0 for p in student.parameters()))
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with EffectCapture(student, [0], tiny_batch()):
                raise RuntimeError("intentional")
        self.assertEqual(len(student.language_model.model.layers[0]._forward_hooks), 0)
        student.language_model.gradient_checkpointing_enable()
        with self.assertRaisesRegex(ValueError, "checkpointing"):
            with EffectCapture(student, [0], tiny_batch()):
                pass

    def test_zero2_resolves_bf16_and_realized_global_batch(self):
        config = load_training_config()
        for stage, expected in (("dpt", 192), ("sft", 96), ("dft", 96)):
            resolved = zero2_config(config, stage, 6)
            self.assertTrue(resolved["bf16"]["enabled"])
            self.assertFalse(resolved["fp16"]["enabled"])
            self.assertEqual(resolved["zero_optimization"]["stage"], 2)
            self.assertEqual(resolved["train_batch_size"], expected)

    def test_attach_online_features_keeps_text_rows_empty(self):
        class Encoder:
            def __init__(self):
                self.calls = 0

            def encode_batch_train(self, images):
                self.calls += len(images)
                return torch.ones(len(images), 3, 8)

        batch = tiny_batch()
        batch.images = (None, object())
        encoder = Encoder()
        result = attach_online_features(batch, encoder)
        self.assertEqual(encoder.calls, 1)
        self.assertIsNone(result.features[0])
        self.assertEqual(tuple(result.features[1].shape), (3, 8))
        self.assertEqual(result.images, ())

    def test_accumulation_matches_whole_batch_valid_token_mean(self):
        batch = tiny_batch()
        microbatches = []
        for index in range(2):
            microbatches.append(Batch(**{field.name: getattr(batch, field.name)[index:index + 1]
                                         for field in fields(batch)}))
        groups = list(accumulation_groups(microbatches, 3))
        self.assertEqual(groups[0][1], (5, 3, 2))  # Partial last accumulation group.
        for stage in ("dpt", "sft", "dft"):
            student = tiny_model()
            teacher = None if stage == "sft" else tiny_model(hidden=32, heads=4, kv=2)
            trainer = VREDTrainer(student, teacher, stage, load_training_config())
            whole, _ = trainer.loss(batch)
            target = student.connector[0].weight
            reference = torch.autograd.grad(whole, target)[0]
            values = []
            for micro in groups[0][0]:
                loss, _ = trainer.loss(micro, normalization_counts=groups[0][1])
                values.append(loss / 2)
            combined = sum(values)
            gradient = torch.autograd.grad(combined, target)[0]
            torch.testing.assert_close(combined, whole, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(gradient, reference, atol=1e-6, rtol=1e-4)

    def test_bf16_all_loss_families_and_both_branch_gradients(self):
        batch = tiny_batch()
        for kind in ("smoothl1", "effect_kl", "effect_js", "reverse_kl"):
            for stage in ("dpt", "dft"):
                student = tiny_model().to(torch.bfloat16)
                teacher = tiny_model(hidden=32, heads=4, kv=2).to(torch.bfloat16)
                config = load_training_config()
                config["write_loss_type"] = kind
                trainer = VREDTrainer(student, teacher, stage, config)
                loss, _ = trainer.loss(batch)
                self.assertEqual(loss.dtype, torch.float32)
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters()))
                self.assertGreater(float(student.connector[0].weight.grad.float().norm()), 0)
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        model = tiny_model()
        plus, minus = torch.randn(2, 24, requires_grad=True), torch.randn(2, 24, requires_grad=True)
        target = torch.randn(2, 97).softmax(-1) - torch.randn(2, 97).softmax(-1)
        effect = semantic_probe(model, plus) - semantic_probe(model, minus)
        loss, _ = effect_loss(effect, target)
        loss.backward()
        self.assertGreater(float(plus.grad.norm()), 0)
        self.assertGreater(float(minus.grad.norm()), 0)
        self.assertIsNone(model.language_model.lm_head.weight.grad)

    def test_cli_optimizer_accumulation_and_save(self):
        from vred.train import main
        batch = tiny_batch()
        rows = []
        for i, length in enumerate((4, len(batch.input_ids[1]))):
            rows.append(dict(input_ids=batch.input_ids[i, :length], labels=batch.labels[i, :length],
                             feature=batch.features[i], visual_span=tuple(batch.visual_spans[i].tolist()),
                             record_id=batch.record_ids[i]))
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0
            bos_token_id = None
            def __len__(self):
                return 97
            def save_pretrained(self, path):
                (Path(path) / "tokenizer_config.json").write_text("{}")
        with tempfile.TemporaryDirectory(prefix="vred-cli-") as directory:
            with patch("sys.argv", ["vred.train", "--stage", "sft", "--output", directory,
                                    "--input-model", "previous-dpt", "--max-steps", "1", "--accumulation", "2"]), \
                    patch("vred.train.validate_tokenizers", return_value=Tokenizer()), \
                    patch("vred.train.AutoConfig.from_pretrained", return_value=Qwen2Config()), \
                    patch("vred.train.load_model", side_effect=lambda *args, **kwargs: tiny_model().to(torch.bfloat16)), \
                    patch("vred.train.CachedDataset", return_value=rows), \
                    patch("vred.train.audit_dataset", return_value={"missing_keys": 0, "missing_files": 0, "malformed": 0}), \
                    patch("sys.stdout", new_callable=io.StringIO):
                main()
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary["optimizer_steps"], 1)
            self.assertEqual(summary["global_examples"], 2)
            self.assertEqual(summary["global_response_tokens"], 5)
            self.assertTrue((Path(directory) / "connector.pt").is_file())
            self.assertTrue((Path(directory) / "language_model/model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()

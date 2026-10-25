# LLaVA-KD 训练数据集准备指南

> 面向复现者。按本文档操作后，两个训练集（LLaVA-558k 预训练集、LLaVA-mix-665k 指令微调集）可完整就位并可直接开训。
>
> 文中所有路径以 `<DATA_ROOT>` 表示数据根目录。本项目实际使用的值是：
>
> ```
> DATA_ROOT = /data/zsx/LLaVA-KD/mllm_datasets
> ```
>
> 如果你放在别的盘，把全文的 `<DATA_ROOT>` 替换成你的路径即可，但**目录层级结构必须保持一致**。

---

## 0. 最终目录结构（先看这个）

```
<DATA_ROOT>/
├── text_files/                     ← 两个 JSON 标注文件放这里
│   ├── blip_laion_cc_sbu_558k.json
│   └── llava_v1_5_mix665k_clean.json
│
├── llava/
│   └── images/                     ← 558k 预训练图片，660 个分片目录
│       ├── 00000/
│       ├── 00001/
│       └── ...
│
├── coco/
│   └── train2017/                  ← COCO 2017 训练图
├── gqa/
│   └── images/                     ← GQA
├── ocr_vqa/
│   └── images/                     ← OCR-VQA
├── textvqa/
│   └── train_images/               ← TextVQA
└── vg/
    ├── VG_100K/                    ← Visual Genome 第一部分
    └── VG_100K_2/                  ← Visual Genome 第二部分
```

**JSON 标注文件不需要和图片放在一起**，但它们内部的相对路径是按上表结构写死的，所以图片目录的相对位置不能改。

---

几个checkpoints需要放在下面的路径下
1. ta-qwen25-3b-stage0_2-lora-finetune
"/LLaVA-KD/checkpoints/ta-qwen25-3b-stage0_2-lora-finetune"

2. Qwen2.5-0.5B
"/data/zsx/LLaVA-KD/pretrained_hg/Qwen2.5-0.5B"

3. dpt
"/LLaVA-KD/checkpoints/vred/dpt"

3. sft
"/LLaVA-KD/checkpoints/vred/sft"

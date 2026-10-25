#!/bin/bash
# monitor_and_run.sh
# 监控指定 PID 列表，全部退出后执行训练命令

PIDS=(2737290 2737291 2737292 2737293 2737294 2737295)
CHECK_INTERVAL=10   # 检查间隔（秒）

echo "开始监控进程: ${PIDS[*]}"
echo "每 ${CHECK_INTERVAL} 秒检查一次，直到所有进程退出..."

while true; do
    all_exited=true
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            # 进程仍然存在
            all_exited=false
            # 可选：打印哪些进程还在运行
            # echo "进程 $pid 仍在运行"
            break
        fi
    done

    if $all_exited; then
        echo "所有监控进程已退出，开始执行训练命令..."
        break
    fi

    sleep "$CHECK_INTERVAL"
done

# 执行训练命令（带所有环境变量）
/home/inspur/miniconda3/envs/nnunet/bin/python -m vred.launch sft   --gpus 6 --seed 1234 --epochs 1 --lr 2e-5   --micro-batch-size 8 --accumulation 2   --input-model checkpoints/vred/dpt   --output checkpoints/vred/sft --gradient-checkpointing --ce-chunk-size 1024



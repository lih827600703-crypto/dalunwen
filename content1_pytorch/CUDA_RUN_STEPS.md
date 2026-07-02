# CUDA GPU 运行步骤

以下步骤以 Windows PowerShell 为例，工作目录为：

```powershell
cd "C:\Users\李鸿\Desktop\研1\gpt\content1_pytorch"
```

## 1. 创建环境

推荐 Python 3.10 或 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. 安装 CUDA 版 PyTorch

如果你的 NVIDIA 驱动较新，推荐 CUDA 12.1 wheel：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

如果显卡驱动较旧，可改用 CUDA 11.8 wheel：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 3. 检查 GPU

```powershell
python gpu_check.py
```

必须看到：

```text
cuda available: True
test tensor device: cuda:0
```

如果显示 `cuda available: False`，说明当前环境装的是 CPU 版 PyTorch，或 NVIDIA 驱动/CUDA wheel 不匹配。

## 4. 一键运行完整实验

```powershell
.\run_gpu.ps1
```

该版本使用 12 维节点特征。若目录中已有旧版 `data/hse_ieee33.npz` 或 `data/hse_ieee33_aug.npz`，请直接运行 `run_gpu.ps1` 覆盖生成，或手动删除后重新生成。

该脚本会依次执行：

1. CUDA 检查
2. 生成 IEEE 33 节点谐波状态估计数据
3. 训练物理约束 WGAN-GP 并生成增强样本
4. 训练 GAT + Transformer 双流谐波状态估计网络

训练结果保存在：

```text
runs/gpu_full/best_estimator.pt
runs/gpu_full/metrics.json
```

## 5. 分步运行

```powershell
python make_dataset.py --samples 500 --out data/hse_ieee33.npz --seed 2027
```

默认 `--steps 96` 对应典型日 15 分钟断面。若严格按第3章“10000个时间步、5000 Hz、2秒”生成，可运行：

```powershell
python make_dataset.py --samples 500 --steps 10000 --out data/hse_ieee33_10000.npz --seed 2027
```

该文件会明显增大，训练显存和磁盘占用也会显著上升。

```powershell
python train_wgan_gp.py `
  --data data/hse_ieee33.npz `
  --out data/hse_ieee33_aug.npz `
  --epochs 80 `
  --num-augmented 300 `
  --batch-size 32 `
  --device cuda
```

```powershell
python train_estimator.py `
  --data data/hse_ieee33.npz `
  --aug-data data/hse_ieee33_aug.npz `
  --epochs 150 `
  --batch-size 16 `
  --hidden-dim 128 `
  --lr 0.001 `
  --physics-weight 0.08 `
  --phase-weight 0.0001 `
  --current-weight 0.8 `
  --score-thd-weight 0.002 `
  --residual-scale 0.25 `
  --diffusion-steps 6 `
  --aug-ratio 0.25 `
  --device cuda `
  --amp `
  --out-dir runs/gpu_full
```

## 6. 显存不足时

优先降低估计器 batch size：

```powershell
python train_estimator.py --data data/hse_ieee33.npz --aug-data data/hse_ieee33_aug.npz --epochs 150 --batch-size 8 --device cuda --amp
```

如果仍不足，再降低模型宽度：

```powershell
python train_estimator.py --data data/hse_ieee33.npz --aug-data data/hse_ieee33_aug.npz --epochs 150 --batch-size 8 --hidden-dim 64 --device cuda --amp
```

## 7. 消融实验

先保证 `data/hse_ieee33.npz` 和 `data/hse_ieee33_aug.npz` 已生成，然后运行：

```powershell
python run_ablations.py --epochs 150 --batch-size 16 --hidden-dim 96
```

# 研究内容一：配电网谐波状态估计 PyTorch 复现实验

本目录给出与 `大论文v2_temp.docx` 中“研究内容一”对应的一套可运行代码：

- IEEE 33 节点配电网谐波数据生成
- 短路、负载突变、谐波放大、PLL 扰动、开关操作等动态/极端工况
- 带物理约束的 WGAN-GP 数据增强
- GAT 空间流 + Transformer 时间流 + 双向时空注意力融合模型
- RMSE、MAE、THD 误差与消融开关

## 快速运行

```powershell
cd C:\Users\李鸿\Desktop\研1\gpt\content1_pytorch
python run_experiment.py --quick
```

## CUDA GPU 运行

完整 GPU 环境安装和运行步骤见 `CUDA_RUN_STEPS.md`。已经提供一键脚本：

```powershell
cd "C:\Users\李鸿\Desktop\研1\gpt\content1_pytorch"
.\run_gpu.ps1
```

注意：当前版本仿照 `hybrid_att_t2_gpu.py` 改为 12 维节点特征，并引入线路阻抗边权 GAT 与拓扑扩散残差估计。旧版本生成的 8 维 `data/hse_ieee33.npz`、`data/hse_ieee33_aug.npz` 不能继续混用；请重新执行 `run_gpu.ps1` 或按下面步骤重新生成数据。

数据生成模块已实现动态谐波源闭环：每个谐波源在逐时刻仿真中根据上一时刻本节点谐波电压更新调制指数、PLL 相位和故障衰减因子，再生成 5/7/11/13 次谐波注入电流并经 IEEE 33 节点传递阻抗传播。修改动态源逻辑后，请重新生成 `data/hse_ieee33.npz` 与 `data/hse_ieee33_aug.npz`。

数据文件中同时保存两种状态表示：

- `y_*`：训练使用的矩形坐标 `[V_real, V_imag, I_real, I_imag]`
- `state_magphase_*`：论文表述对应的 `[电压幅值, 电压相角, 电流幅值, 电流相角]`

可用以下命令校验 IEEE 33 节点线路参数、谐波次数、源节点配置和幅值/相角字段：

```powershell
python validate_ieee33_data.py --data data/hse_ieee33.npz
```

## 论文式可视化

训练完成后可生成第3章常用结果图：

```powershell
python visualize_results.py `
  --data data/hse_ieee33.npz `
  --checkpoint runs/gpu_full/best_estimator.pt `
  --aug-data data/hse_ieee33_aug.npz `
  --out-dir figures/content1 `
  --device cuda `
  --node 24 `
  --harmonic 5
```

输出包括训练收敛曲线、IEEE 33 拓扑与源/量测节点图、典型节点动态跟踪曲线、全网 THD 热力图、谐波分频误差箱线图、不同工况指标柱状图、预测-真实散点图和 WGAN-GP 增强样本分布对比。

## 接近论文设置运行

```powershell
cd C:\Users\李鸿\Desktop\研1\gpt\content1_pytorch
python make_dataset.py --samples 500 --out data/hse_ieee33.npz
python train_wgan_gp.py --data data/hse_ieee33.npz --epochs 80 --out data/hse_ieee33_aug.npz
python train_estimator.py --data data/hse_ieee33.npz --aug-data data/hse_ieee33_aug.npz --epochs 150 --batch-size 16 --hidden-dim 128 --residual-scale 0.25 --diffusion-steps 6 --aug-ratio 0.25 --phase-weight 0.0001 --current-weight 0.8 --score-thd-weight 0.002
```

论文中的目标量级为：RMSE `0.0021 p.u.`、MAE `0.0016 p.u.`、THD 误差 `0.26%`。合成数据使用固定随机种子、较长训练轮数和 WGAN-GP 增强时，指标会稳定接近该量级；不同 PyTorch/CUDA 版本下会有小幅波动。

## 主要文件

- `hse_pytorch/ieee33.py`：IEEE 33 节点拓扑、邻接矩阵、谐波传递矩阵
- `hse_pytorch/data.py`：动态谐波源与极端工况数据生成
- `hse_pytorch/models.py`：GAT、Transformer、双向时空融合、WGAN-GP
- `hse_pytorch/metrics.py`：RMSE、MAE、THD 误差
- `make_dataset.py`：生成训练/验证/测试数据
- `train_wgan_gp.py`：训练物理约束 WGAN-GP 并导出增强样本
- `train_estimator.py`：训练谐波状态估计网络，可做消融
- `run_experiment.py`：一键跑通数据生成、增强和训练

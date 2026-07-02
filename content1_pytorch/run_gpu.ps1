$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

python gpu_check.py

python make_dataset.py `
  --samples 500 `
  --out data/hse_ieee33.npz `
  --seed 2027

python validate_ieee33_data.py `
  --data data/hse_ieee33.npz

python train_wgan_gp.py `
  --data data/hse_ieee33.npz `
  --out data/hse_ieee33_aug.npz `
  --epochs 80 `
  --num-augmented 300 `
  --batch-size 32 `
  --device cuda `
  --num-workers 0

python train_estimator.py `
  --data data/hse_ieee33.npz `
  --aug-data data/hse_ieee33_aug.npz `
  --epochs 100 `
  --batch-size 16 `
  --hidden-dim 128 `
  --lr 0.0005 `
  --physics-weight 0.08 `
  --phase-weight 0.0001 `
  --current-weight 0.8 `
  --score-thd-weight 0.002 `
  --huber-beta 0.5 `
  --residual-scale 0.25 `
  --diffusion-steps 6 `
  --aug-ratio 0.25 `
  --device cuda `
  --amp `
  --num-workers 0 `
  --out-dir runs/gpu_full

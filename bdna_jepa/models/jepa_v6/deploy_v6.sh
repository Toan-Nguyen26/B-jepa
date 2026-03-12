#!/bin/bash
# B-JEPA v6.0 Deployment — Vast.ai A100
# =======================================
# True JEPA: asymmetric masking + cross-attention predictor + SIGReg
set -e

echo "======================================================="
echo "  B-JEPA v6.0 — True JEPA for Bacterial DNA"
echo "======================================================="

cd /workspace/bdna-jepa

# -- Kill previous runs --
echo "[1/4] Killing previous training..."
pkill -f pretrain_v 2>/dev/null || true
kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null || true
sleep 3
echo "  Done."

# -- Pull latest code --
echo "[2/4] Pulling latest code..."
git pull origin main 2>/dev/null || echo "  (git pull skipped)"

# -- Verify files --
echo "[3/4] Checking files..."
ls -lh bdna_jepa/models/jepa_v6/pretrain_v6.py 2>/dev/null && echo "  > pretrain_v6.py" || { echo "  MISSING pretrain_v6.py"; exit 1; }
ls -lh data/processed/pretrain_2M.csv 2>/dev/null && echo "  > pretrain_2M.csv" || { echo "  MISSING data"; exit 1; }
ls -lh data/tokenizer/bpe_4096.json 2>/dev/null && echo "  > bpe_4096.json" || { echo "  MISSING tokenizer"; exit 1; }

pip install -q umap-learn scikit-learn matplotlib pandas 2>/dev/null || true

# -- Launch --
echo ""
echo "[4/4] Launching v6.0..."
mkdir -p outputs/checkpoints/v6.0/viz

nohup python bdna_jepa/models/jepa_v6/pretrain_v6.py \
    --data-path data/processed/pretrain_2M.csv \
    --tokenizer-path data/tokenizer/bpe_4096.json \
    --output-dir outputs \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --min-lr 1e-6 \
    --warmup-epochs 1 \
    --weight-decay 0.05 \
    --grad-clip 1.0 \
    --embed-dim 576 \
    --num-layers 12 \
    --num-heads 9 \
    --ff-dim 2304 \
    --max-seq-len 512 \
    --predictor-dim 384 \
    --predictor-depth 6 \
    --predictor-heads 6 \
    --jepa-mask-start 0.50 \
    --jepa-mask-end 0.70 \
    --num-blocks 4 \
    --min-block-start 10 \
    --min-block-end 30 \
    --mlm-mask-ratio 0.15 \
    --jepa-weight 5.0 \
    --mlm-weight 0.5 \
    --sigreg-weight 10.0 \
    --gc-adv-weight 1.0 \
    --save-every 5 \
    --log-every 50 \
    --wandb-project bdna-jepa \
    --wandb-run-name bjepa-v6.0-true-jepa \
    2>&1 | tee outputs/v6.0_train.log &

sleep 10

echo ""
echo "======================================================="
echo "  v6.0 launched!"
echo "  Log: outputs/v6.0_train.log"
echo "  Checkpoints: outputs/checkpoints/v6.0/"
echo "  Viz: outputs/checkpoints/v6.0/viz/"
echo "======================================================="
echo ""
echo "WHAT CHANGED FROM v5.0:"
echo "  1. TRUE JEPA: JEPA is primary loss (5.0), MLM is anti-collapse (0.5)"
echo "  2. ASYMMETRIC MASKING: 50-70% masked (was 20-30%)"
echo "  3. CROSS-ATTENTION PREDICTOR: target pos queries -> context KV"
echo "     (was MLP denoiser on single CLS vector)"
echo "  4. CONTEXT-ONLY ENCODER: sees only visible tokens, not full sequence"
echo "     (prevents trivial neighbor interpolation that killed v4.5)"
echo "  5. SIGReg replaces VICReg (one hyperparameter, provably better)"
echo "  6. PROPER ENCODER: RoPE + RMSNorm + SwiGLU + QK-Norm"
echo "     (was vanilla nn.TransformerEncoderLayer)"
echo "  7. 30 epochs (was 20) — JEPA needs more steps than MLM"
echo ""
echo "Expected epoch 1 metrics:"
echo "  RankMe > 200      (MLM anti-collapse + SIGReg)"
echo "  JEPA cos ~ 0.1-0.3 (predictor starting to learn)"
echo "  JEPA loss ~ 0.3-0.5 (smooth L1, high mask = hard task)"
echo "  MLM acc ~ 0.05-0.15 (low weight, anti-collapse only)"
echo ""
echo "RED FLAGS:"
echo "  RankMe < 50       -> collapse despite MLM guard"
echo "  JEPA cos > 0.95 before epoch 5 -> trivial task (mask too low)"
echo "  JEPA loss stuck > 0.5 after epoch 10 -> predictor not learning"
echo "  MLM loss not decreasing at all -> encoder broken"
echo ""
echo "HEALTHY SIGNS:"
echo "  RankMe > 300, increasing"
echo "  JEPA cos 0.3-0.7 and slowly increasing"
echo "  SIGReg decreasing (approaching Gaussian)"
echo "  GC |r| < 0.8 (debiased)"
echo ""

tail -f outputs/v6.0_train.log

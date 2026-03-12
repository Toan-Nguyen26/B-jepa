#!/bin/bash
# B-JEPA v6.1 Deployment — Vast.ai A100
# =======================================
# v6.1: SIGReg + variance floor fix (prevents scale collapse)
set -e

echo "======================================================="
echo "  B-JEPA v6.1 — SIGReg Variance Floor Fix"
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
echo "[4/4] Launching v6.1..."
mkdir -p outputs/checkpoints/v6.1/viz

nohup python bdna_jepa/models/jepa_v6/pretrain_v6.py \
    --data-path data/processed/pretrain_2M.csv \
    --tokenizer-path data/tokenizer/bpe_4096.json \
    --output-dir outputs \
    --epochs 50 \
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
    --var-gamma 1.0 \
    --gc-adv-weight 1.0 \
    --run-version v6.1 \
    --save-every 5 \
    --log-every 50 \
    --wandb-project bdna-jepa \
    --wandb-run-name bjepa-v6.1-varfloor \
    2>&1 | tee outputs/v6.1_train.log &

sleep 10

echo ""
echo "======================================================="
echo "  v6.1 launched!"
echo "  Log: outputs/v6.1_train.log"
echo "  Checkpoints: outputs/checkpoints/v6.1/"
echo "  Viz: outputs/checkpoints/v6.1/viz/"
echo "======================================================="
echo ""
echo "WHAT CHANGED FROM v6.0:"
echo "  1. VARIANCE FLOOR: SIGReg now includes hinge loss on projection stds"
echo "     (prevents scale collapse where norms shrink 19->4.5 while SIGReg"
echo "     stays satisfied due to standardization blind spot)"
echo "  2. 50 EPOCHS (was 15): longer training for better convergence"
echo "  3. --var-gamma 1.0: projection stds penalized below 1.0"
echo "     (matches SIGReg's N(0,1) target for both shape AND scale)"
echo ""
echo "v6.0 FAILURE MODE (what this fixes):"
echo "  Epoch 1: RankMe=82, norm=19.0, std=0.776"
echo "  Epoch 2: RankMe=55, norm=6.6,  std=0.148  <- GC-adv triggered collapse"
echo "  Epoch 3: RankMe=18, norm=4.5,  std=0.036  <- SIGReg couldn't fight back"
echo "  Root cause: SIGReg normalizes before testing -> blind to scale"
echo ""
echo "EXPECTED v6.1 HEALTHY SIGNS:"
echo "  RankMe > 200 and increasing"
echo "  var_floor > 0 early, decreasing as stds rise above 1.0"
echo "  Norms stable around 15-25 (not collapsing)"
echo "  JEPA cos 0.3-0.7 and slowly increasing"
echo ""

tail -f outputs/v6.1_train.log

#!/bin/bash
# B-JEPA v4.5 Deployment — Vast.ai A100
# ═══════════════════════════════════════
# 1. Kill v4.4 (save ~$60 on wasted epochs)
# 2. Copy pretrain_v45.py to repo
# 3. Launch v4.5 training in tmux

set -e

echo "═══════════════════════════════════════════════════"
echo "  B-JEPA v4.5 Deployment"
echo "═══════════════════════════════════════════════════"

cd /workspace/bdna-jepa

# ── Kill v4.4 ──
echo "[1/4] Killing v4.4..."
pkill -f pretrain.py 2>/dev/null || true
kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null || true
sleep 3
echo "  Done."

# ── Fix t-SNE in v4.4 eval code (in case needed later) ──
echo "[2/4] Fixing t-SNE parameter..."
find . -name "*.py" -exec sed -i 's/n_iter=1000/max_iter=1000/g' {} + 2>/dev/null || true
echo "  Done."

# ── Copy v4.5 script ──
echo "[3/4] Installing pretrain_v45.py..."
# Assume user has copied pretrain_v45.py to /workspace/bdna-jepa/
if [ ! -f "pretrain_v45.py" ]; then
    echo "  ERROR: pretrain_v45.py not found in /workspace/bdna-jepa/"
    echo "  Please copy it first: scp pretrain_v45.py vast:/workspace/bdna-jepa/"
    exit 1
fi
echo "  Found pretrain_v45.py"

# ── Install missing deps ──
pip install -q umap-learn scikit-learn matplotlib pandas 2>/dev/null || true

# ── Verify data ──
echo ""
echo "  Data check:"
ls -lh data/processed/pretrain_2M.csv 2>/dev/null && echo "  ✓ pretrain_2M.csv" || echo "  ✗ MISSING"
ls -lh data/tokenizer/bpe_4096.json 2>/dev/null && echo "  ✓ bpe_4096.json" || echo "  ✗ MISSING"

# ── Launch ──
echo ""
echo "[4/4] Launching v4.5 training..."
mkdir -p outputs/checkpoints/v4.5/viz

nohup python pretrain_v45.py \
    --data-path data/processed/pretrain_2M.csv \
    --tokenizer-path data/tokenizer/bpe_4096.json \
    --output-dir outputs \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --min-lr 1e-6 \
    --warmup-epochs 5 \
    --embed-dim 576 \
    --num-layers 12 \
    --num-heads 9 \
    --ff-dim 2304 \
    --max-seq-len 512 \
    --predictor-dim 384 \
    --predictor-depth 4 \
    --predictor-num-heads 6 \
    --num-target-blocks 4 \
    --mask-ratio-start 0.15 \
    --mask-ratio-end 0.50 \
    --min-block-len-start 3 \
    --min-block-len-end 15 \
    --sigreg-weight 1.0 \
    --sigreg-num-slices 512 \
    --seq-pred-weight 0.5 \
    --rc-weight 0.1 \
    --gc-adv-weight 1.0 \
    --grad-accum-steps 2 \
    --save-every 5 \
    --log-every 50 \
    --wandb-project bdna-jepa \
    --wandb-run-name bjepa-v4.5-12L576D-sigreg \
    2>&1 | tee outputs/v4.5_train.log &

sleep 10
echo ""
echo "═══════════════════════════════════════════════════"
echo "  v4.5 launched! Monitoring..."
echo "  Log: outputs/v4.5_train.log"
echo "  Checkpoints: outputs/checkpoints/v4.5/"
echo "  UMAP viz: outputs/checkpoints/v4.5/viz/"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Expected epoch 1 metrics (if healthy):"
echo "  RankMe > 300   (vs v4.4 epoch 1 = 334)"
echo "  pred_mse ~ 0.1-1.0  (MSE, not cosine)"
echo "  cos_sim 0.3-0.8  (should improve over epochs)"
echo "  SIGReg < 0.5"
echo ""
echo "RED FLAGS (kill immediately):"
echo "  RankMe < 100 after warmup"
echo "  pred_mse < 0.001 (trivial prediction = collapse)"
echo "  cos_sim > 0.99 before epoch 10 (collapse)"
echo ""
tail -f outputs/v4.5_train.log

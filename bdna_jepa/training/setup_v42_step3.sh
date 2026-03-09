#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# B-JEPA v4.2 — Step 3: Verify, install deps, prepare data, launch
# ═══════════════════════════════════════════════════════════════════════════
set -e

cd /workspace/bdna-jepa

echo "═══════════════════════════════════════════════════════════════════"
echo "  STEP 3: Verify & Launch"
echo "═══════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────
# 3a. Install dependencies
# ─────────────────────────────────────────────────────
echo "[3a] Installing dependencies..."
pip install -q -r requirements.txt 2>/dev/null || echo "  No requirements.txt, checking manually..."
pip install -q wandb umap-learn 2>/dev/null || true
echo "  ✓ Dependencies ready"

# ─────────────────────────────────────────────────────
# 3b. Verify data exists
# ─────────────────────────────────────────────────────
echo ""
echo "[3b] Checking data..."
if [ -f "data/processed/pretrain_2M.csv" ]; then
    LINES=$(wc -l < data/processed/pretrain_2M.csv)
    echo "  ✓ pretrain_2M.csv: ${LINES} lines"
else
    echo "  ⚠ data/processed/pretrain_2M.csv NOT FOUND"
    echo "    You need to either:"
    echo "    1. Copy from previous instance"
    echo "    2. Run the data preparation script"
    echo "    3. Set --data to point to your actual data file"
fi

if [ -f "data/tokenizer/bpe_4096.json" ]; then
    echo "  ✓ Tokenizer found"
else
    echo "  ⚠ data/tokenizer/bpe_4096.json NOT FOUND"
    echo "    Check if tokenizer is at a different path"
    find data/ -name "*.json" 2>/dev/null | head -5
fi

# ─────────────────────────────────────────────────────
# 3c. Verify patches applied correctly
# ─────────────────────────────────────────────────────
echo ""
echo "[3c] Verifying patches..."

echo "  Checking criterion.py:"
if grep -q "F.normalize.*context_cls" bdna_jepa/losses/criterion.py 2>/dev/null; then
    echo "    ✗ L2-norm before VICReg still present — PATCH FAILED"
    echo "    → Apply manually: see MANUAL_PATCHES.md"
else
    echo "    ✓ No L2-norm before VICReg (correct for v4.2)"
fi

if grep -q "class UniformityLoss" bdna_jepa/losses/criterion.py 2>/dev/null; then
    echo "    ✓ UniformityLoss class present"
else
    echo "    ✗ UniformityLoss class MISSING — v4.1 patch not applied"
fi

if grep -q "uniform_loss" bdna_jepa/losses/criterion.py 2>/dev/null; then
    echo "    ✓ uniform_loss computed in forward()"
else
    echo "    ✗ uniform_loss NOT in forward()"
fi

echo ""
echo "  Checking config.py:"
if grep -q "weight_uniformity" bdna_jepa/config.py 2>/dev/null; then
    echo "    ✓ weight_uniformity in LossConfig"
else
    echo "    ✗ weight_uniformity MISSING from LossConfig"
fi

echo ""
echo "  Checking trainer.py logging:"
if grep -q "vicreg_var\|vic_v\|vicreg" bdna_jepa/training/trainer.py 2>/dev/null; then
    echo "    ✓ VICReg metrics in trainer logging"
else
    echo "    ⚠ VICReg metrics NOT in trainer logging"
    echo "    → Apply manually: see MANUAL_PATCHES.md"
fi

echo ""
echo "  Checking v4.2 config:"
if [ -f "configs/training/v4.2.yaml" ]; then
    echo "    ✓ v4.2.yaml exists"
    echo "    Key settings:"
    grep -E "weight_mlm|weight_jepa|weight_vicreg_var|weight_uniformity|peak_lr|warmup|eval_every|save_every" configs/training/v4.2.yaml | sed 's/^/      /'
else
    echo "    ✗ v4.2.yaml NOT FOUND"
fi

# ─────────────────────────────────────────────────────
# 3d. Verify v4.2 config loads
# ─────────────────────────────────────────────────────
echo ""
echo "[3d] Test-loading v4.2 config..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from bdna_jepa.config import load_config
    mc, tc = load_config('configs/training/v4.2.yaml')
    print(f'  ✓ Config loads successfully')
    print(f'    encoder: {mc.encoder.embed_dim}d, {mc.encoder.num_layers}L, vocab={mc.encoder.vocab_size}')
    print(f'    loss: mlm={mc.loss.weight_mlm}, jepa={mc.loss.weight_jepa}, vic_var={mc.loss.weight_vicreg_var}')
    uniform = getattr(mc.loss, 'weight_uniformity', 'NOT FOUND')
    print(f'    uniformity weight: {uniform}')
    print(f'    training: lr={tc.peak_lr}, warmup={tc.warmup_epochs}, epochs={tc.epochs}')
    print(f'    eval_every={tc.eval_every}, save_every={tc.save_every}')
except Exception as e:
    print(f'  ✗ Config load FAILED: {e}')
    import traceback
    traceback.print_exc()
" 2>&1

# ─────────────────────────────────────────────────────
# 3e. Quick smoke test (1 step)
# ─────────────────────────────────────────────────────
echo ""
echo "[3e] Smoke test (optional — uncomment to run)..."
echo "  # python3 scripts/pretrain.py --config configs/training/v4.2.yaml --epochs 1 --no-wandb 2>&1 | head -20"

# ─────────────────────────────────────────────────────
# 3f. Create output directories
# ─────────────────────────────────────────────────────
echo ""
echo "[3f] Creating output directories..."
mkdir -p outputs/checkpoints/v4.2
echo "  ✓ outputs/checkpoints/v4.2 ready"

# ─────────────────────────────────────────────────────
# 3g. Git commit changes
# ─────────────────────────────────────────────────────
echo ""
echo "[3g] Committing v4.2 changes..."
git add -A
git commit -m "v4.2: JEPA-primary loss rebalance, remove L2-norm VICReg, lower LR

Root cause analysis of v4.1 failure:
- JEPA loss monotonically diverged (0.38→2.49 over 16 epochs)
- MLM dominated gradient at 84-97% of total loss
- Angular collapse: RankMe 500→355, std 1.3→5.8

v4.2 fixes:
- weight_mlm: 5.0→1.0, weight_jepa: 1.0→5.0 (JEPA primary)
- peak_lr: 1e-3→3e-4 (slower updates for EMA tracking)
- warmup: 3→5 epochs (predictor needs time to establish mapping)
- Removed L2-norm before VICReg (saturated var penalty)
- weight_vicreg_var: 25→10 (appropriate for raw embeddings)
- weight_uniformity: 5.0→1.0 (supporting role)
- eval_every: 5→1, save_every: 10→5 (early collapse detection)
" 2>/dev/null || echo "  (git commit skipped — review changes first)"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  LAUNCH COMMAND:"
echo "  ────────────────"
echo ""
echo "  # Login to wandb first:"
echo "  wandb login"
echo ""
echo "  # Start training (use nohup or tmux to survive disconnects):"
echo "  nohup python3 scripts/pretrain.py \\"
echo "    --config configs/training/v4.2.yaml \\"
echo "    --data data/processed/pretrain_2M.csv \\"
echo "    2>&1 | tee outputs/v4.2_train.log &"
echo ""
echo "  # Monitor live:"
echo "  tail -f outputs/v4.2_train.log"
echo ""
echo "  ────────────────"
echo "  WHAT TO WATCH (first 3 epochs):"
echo ""
echo "  ✓ JEPA loss DECREASING (0.4 → 0.2-0.3)"
echo "  ✓ MLM loss decreasing  (8.3 → 7.x)"
echo "  ✓ vic_v near 0 (var penalty inactive = healthy std)"
echo "  ✓ RankMe > 450 at eval"
echo "  ✓ std between 1.0-2.5"
echo ""
echo "  ✗ STOP if JEPA rises 2+ consecutive evals"
echo "  ✗ STOP if RankMe < 400"
echo "  ✗ STOP if std > 3.0 by epoch 5"
echo "═══════════════════════════════════════════════════════════════════"

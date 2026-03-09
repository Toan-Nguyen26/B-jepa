#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# B-JEPA v4.2 — Step 2: Apply patches, create config, verify
# Run AFTER setup_v42.sh (Step 1) and reviewing its output
# ═══════════════════════════════════════════════════════════════════════════
set -e

cd /workspace/bdna-jepa

echo "═══════════════════════════════════════════════════════════════════"
echo "  STEP 2: Apply v4.2 patches"
echo "═══════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────
# 2a. Apply v4.1 base patches if not present
# ─────────────────────────────────────────────────────
echo "[2a] Checking if v4.1 patch needs applying..."

if ! grep -q "UniformityLoss" bdna_jepa/losses/criterion.py 2>/dev/null; then
    echo "  v4.1 patch NOT found — applying v4.1 base patch first..."
    bash patch_v41.sh  # Your existing patch script
    echo "  v4.1 patch applied"
else
    echo "  v4.1 patch already present"
fi

# ─────────────────────────────────────────────────────
# 2b. Verify VICReg + Uniformity are in criterion
# ─────────────────────────────────────────────────────
echo ""
echo "[2b] Verifying loss components..."
echo "  UniformityLoss: $(grep -c 'class UniformityLoss' bdna_jepa/losses/criterion.py) definitions"
echo "  VICRegLoss:     $(grep -c 'class VICRegLoss' bdna_jepa/losses/criterion.py) definitions"
echo "  uniform_loss:   $(grep -c 'uniform_loss' bdna_jepa/losses/criterion.py) references"

# ─────────────────────────────────────────────────────
# 2c. CRITICAL PATCH: Remove L2-norm before VICReg
#     (v4.1 applies L2-norm which saturates VICReg)
# ─────────────────────────────────────────────────────
echo ""
echo "[2c] Patching: Remove L2-norm before VICReg (v4.2 fix)..."

# Check if L2-norm is applied before VICReg
if grep -q "context_cls_norm = F.normalize" bdna_jepa/losses/criterion.py 2>/dev/null; then
    echo "  Found L2-norm before VICReg — reverting to raw embeddings..."
    # Replace the L2-normed VICReg call with raw embeddings
    python3 -c "
with open('bdna_jepa/losses/criterion.py', 'r') as f:
    content = f.read()

# Remove the L2-norm line and revert to using raw context_cls
old = '''# L2-normalize before VICReg (v4.1: operate on unit sphere)
        context_cls_norm = F.normalize(model_output[\"context_cls\"], dim=1)
        var_loss, cov_loss = self.vicreg_loss(context_cls_norm)'''

new = '''# v4.2: VICReg on raw embeddings (L2-norm saturated var penalty in v4.1)
        var_loss, cov_loss = self.vicreg_loss(model_output[\"context_cls\"])'''

if old in content:
    content = content.replace(old, new)
    print('  Reverted: VICReg now operates on raw CLS embeddings')
else:
    # Try finding it with slight variations
    if 'context_cls_norm' in content and 'F.normalize' in content:
        content = content.replace('context_cls_norm = F.normalize(model_output[\"context_cls\"], dim=1)', '')
        content = content.replace('self.vicreg_loss(context_cls_norm)', 'self.vicreg_loss(model_output[\"context_cls\"])')
        print('  Reverted (alt pattern): VICReg on raw embeddings')
    else:
        print('  No L2-norm found before VICReg — already on raw embeddings')

with open('bdna_jepa/losses/criterion.py', 'w') as f:
    f.write(content)
"
else
    echo "  No L2-norm found — VICReg already on raw embeddings"
fi

# ─────────────────────────────────────────────────────
# 2d. CRITICAL PATCH: Add step-level VICReg/uniformity logging
# ─────────────────────────────────────────────────────
echo ""
echo "[2d] Patching trainer for step-level diagnostic logging..."

python3 << 'PYEOF'
import re

with open('bdna_jepa/training/trainer.py', 'r') as f:
    content = f.read()

# Find the step-level logging format string and add VICReg metrics
# Current format: "Step {step} | epoch {epoch} | loss={loss} | mlm={mlm} | jepa={jepa} | lr={lr}"
# We need to add vicreg_var, vicreg_cov, uniformity, and std

# Strategy: Find where loss_output is used for logging and add the missing metrics
# Look for the f-string or format that creates the step log line

changes_made = []

# Add vicreg/uniformity to the step log message
# Pattern 1: f-string with loss=, mlm=, jepa=
if 'mlm={' in content and 'jepa={' in content:
    # Find the log line and add metrics
    # Look for the log_every block
    pass

# Pattern 2: Look for where metrics dict is built for wandb logging
if 'train/vicreg_var' not in content and 'vicreg_var' in content:
    # The wandb dict exists but doesn't have vicreg metrics
    pass

# More robust approach: Find the _train_step method and ensure loss_output 
# components are logged
# Let's check what's in loss_output and what gets logged

# Check if there's a metrics/log dict being built
wandb_log_pattern = re.search(r'"train/mlm":\s*loss_output\[.*?\]\.item\(\)', content)
if wandb_log_pattern:
    # Find the full dict and add missing entries
    # Search for the dict block
    match = re.search(r'(metrics|log_dict|wandb_log)\s*=\s*\{[^}]*"train/mlm"[^}]*\}', content, re.DOTALL)
    if match:
        old_dict = match.group(0)
        if '"train/vicreg_var"' not in old_dict:
            # Add vicreg metrics before the closing brace
            insert_metrics = '''
            "train/vicreg_var": loss_output.get("vicreg_var", torch.tensor(0.0)).item(),
            "train/vicreg_cov": loss_output.get("vicreg_cov", torch.tensor(0.0)).item(),
            "train/uniformity": loss_output.get("uniformity", torch.tensor(0.0)).item(),'''
            new_dict = old_dict.replace('"train/mlm"', insert_metrics + '\n            "train/mlm"')
            content = content.replace(old_dict, new_dict)
            changes_made.append("Added vicreg_var/cov/uniformity to wandb metrics dict")

# Also add to the console log line
# Find the logger.info line for step logging
step_log_pattern = re.search(r'(logger\.info\(f["\']Step.*?loss=.*?["\'].*?\))', content, re.DOTALL)
if step_log_pattern:
    old_log = step_log_pattern.group(0)
    if 'vic=' not in old_log and 'vicreg' not in old_log:
        # Add compact vicreg info to log line
        # Insert before the closing quote+paren
        if 'lr=' in old_log:
            old_lr_part = re.search(r'(lr=\{[^}]+\})', old_log)
            if old_lr_part:
                old_part = old_lr_part.group(0)
                new_part = old_part + ' | vic_v={loss_output.get("vicreg_var", torch.tensor(0.0)).item():.3f} | vic_c={loss_output.get("vicreg_cov", torch.tensor(0.0)).item():.3f} | unif={loss_output.get("uniformity", torch.tensor(0.0)).item():.3f}'
                content = content.replace(old_log, old_log.replace(old_part, new_part))
                changes_made.append("Added vicreg/uniformity to step console log")

# Ensure torch is imported at top of file
if 'import torch' not in content:
    content = 'import torch\n' + content
    changes_made.append("Added torch import")

with open('bdna_jepa/training/trainer.py', 'w') as f:
    f.write(content)

if changes_made:
    for c in changes_made:
        print(f"  ✓ {c}")
else:
    print("  ⚠ Could not auto-patch trainer logging — MANUAL PATCH NEEDED")
    print("    See MANUAL_PATCHES.md for instructions")
PYEOF

# ─────────────────────────────────────────────────────
# 2e. Add weight_uniformity to config if missing
# ─────────────────────────────────────────────────────
echo ""
echo "[2e] Ensuring weight_uniformity in LossConfig..."

if ! grep -q "weight_uniformity" bdna_jepa/config.py 2>/dev/null; then
    python3 -c "
with open('bdna_jepa/config.py', 'r') as f:
    content = f.read()

import re
m = re.search(r'(weight_vicreg_cov:\s*float\s*=\s*[\d.]+)', content)
if m:
    content = content.replace(m.group(0), m.group(0) + '\n    weight_uniformity: float = 0.0')
    print('  ✓ Added weight_uniformity to LossConfig')
else:
    print('  ⚠ Could not find weight_vicreg_cov — add weight_uniformity manually')

with open('bdna_jepa/config.py', 'w') as f:
    f.write(content)
"
else
    echo "  ✓ weight_uniformity already in config"
fi

# ─────────────────────────────────────────────────────
# 2f. Create v4.2 config
# ─────────────────────────────────────────────────────
echo ""
echo "[2f] Creating configs/training/v4.2.yaml..."

mkdir -p configs/training

cat > configs/training/v4.2.yaml << 'YAML'
# ═══════════════════════════════════════════════════════════════════════
# B-JEPA v4.2 — Clean restart with JEPA-primary loss balance
# ═══════════════════════════════════════════════════════════════════════
#
# Changes from v4.1:
#   1. weight_mlm:  5.0 → 1.0  (was drowning JEPA at 85% of gradient)
#   2. weight_jepa: 1.0 → 5.0  (JEPA is now the primary objective)
#   3. peak_lr:     1e-3 → 3e-4 (context encoder was updating too fast for EMA)
#   4. warmup:      3 → 5 epochs (JEPA diverged during warmup in v4.1)
#   5. VICReg on raw embeddings (removed L2-norm that saturated var penalty)
#   6. weight_vicreg_var: 25 → 10 (less aggressive with raw embeddings)
#   7. weight_uniformity: 5.0 → 1.0 (supporting role, not dominant)
#   8. eval_every: 5 → 1 (catch collapse early)
#   9. save_every: 10 → 5 (more frequent checkpoints for recovery)
#
# Expected behavior:
#   - JEPA loss should DECREASE epochs 0-5, then plateau
#   - MLM loss should decrease steadily (8.3→7.0 by epoch 5)
#   - RankMe should stay >450
#   - std should stay 1.0-2.5
#   - Total loss dominated by JEPA (50-60%), not MLM
#
# RED FLAGS (stop and debug):
#   - JEPA loss increasing >2 consecutive evals
#   - RankMe < 400
#   - std > 3.0

encoder:
  vocab_size: 4096
  embed_dim: 576
  num_layers: 12
  num_heads: 9
  ff_dim: 2304
  ff_activation: swiglu
  dropout: 0.1
  max_seq_len: 512
  pos_encoding: rotary
  norm_type: rmsnorm
  qk_norm: true
  attention_dropout: 0.0
  embed_dropout: 0.1
  bias: false

loss:
  target_mode: stop_grad
  jepa_loss_type: smooth_l1
  weight_mlm: 1.0                # v4.1: 5.0 → v4.2: 1.0 (CRITICAL: was drowning JEPA)
  weight_jepa: 5.0               # v4.1: 1.0 → v4.2: 5.0 (JEPA is now primary)
  weight_vicreg_var: 10.0        # v4.1: 25.0 → v4.2: 10.0 (raw embeddings, not saturated)
  weight_vicreg_cov: 1.0
  weight_uniformity: 1.0         # v4.1: 5.0 → v4.2: 1.0 (supporting role)
  vicreg_gamma: 1.0
  use_gradnorm: false
  gradnorm_alpha: 0.5
  gradnorm_lr: 0.025
  mlm_mask_ratio: 0.15
  mlm_mask_strategy: span
  mlm_span_length: 5
  fragment:
    enabled: true
    context_size: 4
    predictor_depth: 2
    predictor_dim: 192
    predictor_heads: 6
    loss_weight: 1.0

training:
  optimizer: adamw
  peak_lr: 3.0e-4               # v4.1: 1e-3 → v4.2: 3e-4 (slower updates for EMA tracking)
  min_lr: 1.0e-6
  warmup_epochs: 5              # v4.1: 3 → v4.2: 5 (let predictor establish mapping)
  weight_decay_start: 0.04
  weight_decay_end: 0.4
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-8
  grad_clip: 1.0
  lr_schedule: cosine
  epochs: 60
  batch_size: 128
  num_workers: 4
  mixed_precision: true
  precision: bf16
  data_path: data/processed/pretrain_2M.csv
  tokenizer_path: data/tokenizer/bpe_4096.json
  checkpoint_dir: outputs/checkpoints/v4.2
  save_every: 5                  # v4.1: 10 → v4.2: 5 (more checkpoints for recovery)
  eval_every: 1                  # v4.1: 5 → v4.2: 1 (catch collapse within first epoch)
  log_every: 50
  use_wandb: true
  wandb_project: bdna-jepa
  wandb_entity: null
  seed: 42
YAML

echo "  ✓ v4.2 config created"

# ─────────────────────────────────────────────────────
# 2g. Create manual patch instructions (backup)
# ─────────────────────────────────────────────────────
cat > MANUAL_PATCHES.md << 'MD'
# B-JEPA v4.2 Manual Patches

If the auto-patches in setup_v42_step2.sh failed, apply these manually:

## 1. trainer.py — Step-level VICReg logging

Find the step logging line (something like):
```python
logger.info(f"Step {step} | epoch {epoch} | loss={...} | mlm={...} | jepa={...} | lr={...}")
```

Add after `lr=`:
```python
 | vic_v={loss_output.get("vicreg_var", torch.tensor(0.0)).item():.3f} | vic_c={loss_output.get("vicreg_cov", torch.tensor(0.0)).item():.3f} | unif={loss_output.get("uniformity", torch.tensor(0.0)).item():.3f}
```

## 2. criterion.py — Remove L2-norm before VICReg

Find:
```python
context_cls_norm = F.normalize(model_output["context_cls"], dim=1)
var_loss, cov_loss = self.vicreg_loss(context_cls_norm)
```

Replace with:
```python
var_loss, cov_loss = self.vicreg_loss(model_output["context_cls"])
```

## 3. Verify loss_output dict includes vicreg_var, vicreg_cov, uniformity

In criterion.py forward(), the returned dict must include:
```python
"vicreg_var": var_loss,
"vicreg_cov": cov_loss,
"uniformity": uniform_loss,
```
MD

echo "  ✓ MANUAL_PATCHES.md created"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  STEP 2 COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"

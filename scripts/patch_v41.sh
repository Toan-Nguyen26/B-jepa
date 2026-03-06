#!/bin/bash
# B-JEPA v4.1 Patch Script
# Run from /workspace/bdna-jepa/
# Applies: L2-norm before VICReg, uniformity loss, config for 2M data
set -e

echo "=== B-JEPA v4.1 Patch ==="

# ─────────────────────────────────────────────────────
# 1. Add UniformityLoss to losses/criterion.py
# ─────────────────────────────────────────────────────
echo "[1/6] Adding UniformityLoss to losses/criterion.py..."

# Insert UniformityLoss class after VICRegLoss
python3 -c "
import re

with open('bdna_jepa/losses/criterion.py', 'r') as f:
    content = f.read()

# Add UniformityLoss class right before GradNormBalancer
uniformity_class = '''
class UniformityLoss(nn.Module):
    \"\"\"Uniformity loss on the hypersphere (Wang & Isola, ICML 2020).
    
    Pushes embeddings apart on the unit sphere:
    L_uniform = log(mean(exp(-t * ||z_i - z_j||^2)))
    
    Lower = more uniform distribution on sphere.
    \"\"\"
    def __init__(self, t: float = 2.0):
        super().__init__()
        self.t = t
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            z: (B, D) L2-normalized embeddings
        Returns:
            scalar uniformity loss
        \"\"\"
        z = F.normalize(z, dim=1)
        sq_pdist = torch.pdist(z, p=2).pow(2)
        return sq_pdist.mul(-self.t).exp().mean().log()


'''

# Insert before GradNormBalancer
content = content.replace('class GradNormBalancer', uniformity_class + 'class GradNormBalancer')

with open('bdna_jepa/losses/criterion.py', 'w') as f:
    f.write(content)

print('  Added UniformityLoss class')
"

# ─────────────────────────────────────────────────────
# 2. Modify BJEPACriterion to L2-norm before VICReg + add uniformity
# ─────────────────────────────────────────────────────
echo "[2/6] Patching BJEPACriterion forward..."

python3 -c "
with open('bdna_jepa/losses/criterion.py', 'r') as f:
    content = f.read()

# Add uniformity_loss to __init__ — find where vicreg_loss is initialized
# We need to add it after self.vicreg_loss = VICRegLoss(...)
old_init = 'self.vicreg_loss = VICRegLoss('
if old_init not in content:
    # Try finding it differently
    import re
    m = re.search(r'self\.vicreg_loss\s*=\s*VICRegLoss\(', content)
    if m:
        old_init = m.group(0)

# Find the line with vicreg_loss init and add uniformity after it
lines = content.split('\n')
new_lines = []
added_uniformity_init = False
added_uniformity_forward = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # After vicreg_loss init, add uniformity_loss
    if 'self.vicreg_loss' in line and 'VICRegLoss' in line and not added_uniformity_init:
        indent = len(line) - len(line.lstrip())
        spaces = ' ' * indent
        new_lines.append(f'{spaces}self.uniformity_loss = UniformityLoss(t=2.0)')
        added_uniformity_init = True

# Now modify the forward method to:
# 1. L2-normalize before VICReg
# 2. Compute uniformity loss
content = '\n'.join(new_lines)

# Replace VICReg call to normalize first
old_vicreg_call = 'var_loss, cov_loss = self.vicreg_loss(model_output[\"context_cls\"])'
new_vicreg_call = '''# L2-normalize before VICReg (v4.1: operate on unit sphere)
        context_cls_norm = F.normalize(model_output[\"context_cls\"], dim=1)
        var_loss, cov_loss = self.vicreg_loss(context_cls_norm)
        # Uniformity loss (v4.1: push embeddings apart on hypersphere)
        uniform_loss = self.uniformity_loss(model_output[\"context_cls\"])'''

content = content.replace(old_vicreg_call, new_vicreg_call)

# Add uniformity to result dict
old_result = '''\"target_vicreg_cov\": target_cov,
        }'''
new_result = '''\"target_vicreg_cov\": target_cov,
            \"uniformity\": uniform_loss,
        }'''
content = content.replace(old_result, new_result)

# Add uniformity to total loss
old_total = 'total = total_task + self.config.weight_vicreg_var * var_loss + self.config.weight_vicreg_cov * cov_loss'
new_total = '''weight_uniform = getattr(self.config, 'weight_uniformity', 0.0)
        total = total_task + self.config.weight_vicreg_var * var_loss + self.config.weight_vicreg_cov * cov_loss + weight_uniform * uniform_loss'''
content = content.replace(old_total, new_total)

with open('bdna_jepa/losses/criterion.py', 'w') as f:
    f.write(content)

print('  Patched: L2-norm before VICReg, uniformity loss added')
"

# ─────────────────────────────────────────────────────
# 3. Add weight_uniformity to LossConfig
# ─────────────────────────────────────────────────────
echo "[3/6] Adding weight_uniformity to LossConfig..."

python3 -c "
with open('bdna_jepa/config.py', 'r') as f:
    content = f.read()

# Add weight_uniformity after weight_vicreg_cov
old = 'weight_vicreg_cov: float = 1.0'
new = '''weight_vicreg_cov: float = 1.0
    weight_uniformity: float = 0.0  # v4.1: uniformity loss weight'''

if old in content:
    content = content.replace(old, new)
    print('  Added weight_uniformity to LossConfig')
else:
    # Try to find it with different default
    import re
    m = re.search(r'weight_vicreg_cov:\s*float\s*=\s*[\d.]+', content)
    if m:
        content = content.replace(m.group(0), m.group(0) + '\n    weight_uniformity: float = 0.0  # v4.1: uniformity loss weight')
        print('  Added weight_uniformity to LossConfig (regex match)')
    else:
        print('  WARNING: Could not find weight_vicreg_cov in config.py — add manually')

with open('bdna_jepa/config.py', 'w') as f:
    f.write(content)
"

# ─────────────────────────────────────────────────────
# 4. Add uniformity logging to trainer.py
# ─────────────────────────────────────────────────────
echo "[4/6] Patching trainer to log uniformity loss..."

python3 -c "
with open('bdna_jepa/training/trainer.py', 'r') as f:
    content = f.read()

old_log = '\"train/vicreg_cov\": loss_output[\"vicreg_cov\"].item(),'
new_log = '''\"train/vicreg_cov\": loss_output[\"vicreg_cov\"].item(),
            \"train/uniformity\": loss_output.get(\"uniformity\", torch.tensor(0.0)).item(),'''

if old_log in content:
    content = content.replace(old_log, new_log)
    # Make sure torch is imported (it should be already)
    print('  Added uniformity logging to trainer')
else:
    print('  WARNING: Could not find vicreg_cov logging line — add manually')

with open('bdna_jepa/training/trainer.py', 'w') as f:
    f.write(content)
"

# ─────────────────────────────────────────────────────
# 5. Create v4.1 config
# ─────────────────────────────────────────────────────
echo "[5/6] Creating configs/training/v4.1.yaml..."

cat > configs/training/v4.1.yaml << 'YAML'
# B-JEPA v4.1 — Angular collapse fix + 2M data
# Changes from v4.0:
#   - L2-normalize CLS before VICReg (code change in criterion.py)
#   - Added uniformity loss (weight=5.0) to push embeddings apart
#   - Bumped weight_vicreg_var: 10.0 -> 25.0
#   - Scaled data: 500K -> 2M fragments
#   - Reduced epochs: 100 -> 60 (4x more data)
#   - Reduced warmup: 5 -> 3 epochs

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
  weight_mlm: 5.0
  weight_jepa: 1.0
  weight_vicreg_var: 25.0       # v4.0: 10.0 -> v4.1: 25.0
  weight_vicreg_cov: 1.0
  weight_uniformity: 5.0        # NEW in v4.1
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
  peak_lr: 1.0e-3
  min_lr: 1.0e-6
  warmup_epochs: 3              # v4.0: 5 -> v4.1: 3
  weight_decay_start: 0.04
  weight_decay_end: 0.4
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-8
  grad_clip: 1.0
  lr_schedule: cosine
  epochs: 60                    # v4.0: 100 -> v4.1: 60
  batch_size: 128
  num_workers: 4
  mixed_precision: true
  precision: bf16
  data_path: data/processed/pretrain_2M.csv
  tokenizer_path: data/tokenizer/bpe_4096.json
  checkpoint_dir: outputs/checkpoints/v4.1
  save_every: 10
  eval_every: 5
  log_every: 50
  use_wandb: true
  wandb_project: bdna-jepa
  seed: 42
YAML

echo ""
echo "=== v4.1 Patch Complete ==="
echo ""
echo "Changes applied:"
echo "  1. UniformityLoss class added to losses/criterion.py"
echo "  2. L2-normalize CLS before VICReg in BJEPACriterion.forward()"
echo "  3. Uniformity loss computed and added to total loss"
echo "  4. Uniformity loss logged in trainer.py"
echo "  5. weight_uniformity added to LossConfig (default=0.0 for backward compat)"
echo "  6. configs/training/v4.1.yaml created (2M data, 60 epochs, new weights)"
echo ""
echo "Verify changes:"
grep -n 'UniformityLoss\|normalize\|uniform' bdna_jepa/losses/criterion.py | head -10
echo "---"
grep -n 'uniformity' bdna_jepa/training/trainer.py
echo "---"
grep -n 'uniformity' bdna_jepa/config.py
echo ""
echo "To train:"
echo "  mkdir -p outputs/checkpoints/v4.1"
echo "  python3 scripts/train.py --config configs/training/v4.1.yaml"

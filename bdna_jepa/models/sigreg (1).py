"""
SIGReg — Sketched Isotropic Gaussian Regularization for DNA-JEPA.

From LeJEPA (Balestriero & LeCun, 2025, arXiv:2511.08544).

Replaces VICReg (3 fragile terms) + LDReg with ONE principled loss:
  - Project embeddings onto random 1D directions
  - Compare each projection to standard Gaussian via Epps-Pulley test
  - Average over all directions

This forces embeddings toward isotropic Gaussian = provably optimal
for downstream prediction risk. Prevents collapse by construction.

Integration into DNA-JEPA pretraining:
  L_total = L_pred (MSE) + λ * SIGReg(embeddings)
  
  That's it. One hyperparameter λ.

Usage:
  # Option A: Use lejepa package (pip install lejepa)
  sigreg_loss = create_sigreg_loss(use_package=True)
  
  # Option B: Standalone (no dependencies)
  sigreg_loss = create_sigreg_loss(use_package=False)
  
  # In training loop:
  loss = sigreg_loss(embeddings)  # (B, D) tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# =============================================================================
# Standalone Epps-Pulley implementation (no external deps)
# =============================================================================

class EppsPulley(nn.Module):
    """Epps-Pulley characteristic function test for normality.
    
    Tests whether a 1D sample follows N(0,1) by comparing empirical
    and theoretical characteristic functions on a grid of points.
    
    The test statistic is:
        EP(X) = (2/n) Σ_i Σ_j [cos(t(X_i - X_j)) - 2·cos(t·X_i)·exp(-t²/2) + exp(-t²)]
    integrated over t via quadrature.
    
    Returns 0 when samples are perfectly Gaussian, positive otherwise.
    Differentiable w.r.t. input samples.
    
    Args:
        num_points: Number of quadrature points for integration (default: 17)
    """
    
    def __init__(self, num_points: int = 17):
        super().__init__()
        self.num_points = num_points
        # Quadrature grid: points in [0, max_t] where characteristic 
        # function has most discriminative power
        # Standard choice: equally spaced in [0, 2] or use Gauss-Legendre
        t_max = 2.0
        t_points = torch.linspace(0, t_max, num_points + 1)[1:]  # exclude 0
        self.register_buffer('t_points', t_points)
        # Weights for trapezoidal integration
        dt = t_max / num_points
        weights = torch.full((num_points,), dt)
        weights[0] = dt / 2
        weights[-1] = dt / 2
        self.register_buffer('weights', weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N,) 1D samples, should be standardized (mean 0, std 1)
        Returns:
            scalar test statistic (0 = perfectly Gaussian)
        """
        N = x.shape[0]
        if N < 4:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # Standardize (important for the test to work correctly)
        x = (x - x.mean()) / (x.std() + 1e-8)
        
        total = torch.tensor(0.0, device=x.device)
        
        for k, t in enumerate(self.t_points):
            # Empirical CF: (1/n) Σ exp(i·t·x_j) → real part = (1/n) Σ cos(t·x_j)
            cos_tx = torch.cos(t * x)  # (N,)
            sin_tx = torch.sin(t * x)  # (N,)
            
            # |ECF|² = [(1/n)Σcos(tx)]² + [(1/n)Σsin(tx)]²
            ecf_real = cos_tx.mean()
            ecf_imag = sin_tx.mean()
            ecf_sq = ecf_real ** 2 + ecf_imag ** 2
            
            # Theoretical CF of N(0,1): exp(-t²/2)
            tcf = math.exp(-0.5 * t.item() ** 2)
            tcf_sq = tcf ** 2
            
            # |ECF - TCF|² = |ECF|² - 2·Re(ECF)·TCF + TCF²
            # (since TCF is real for Gaussian)
            integrand = ecf_sq - 2 * ecf_real * tcf + tcf_sq
            
            total = total + self.weights[k] * integrand
        
        return total


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization.
    
    Projects D-dimensional embeddings onto K random 1D directions,
    applies Epps-Pulley test on each projection, averages results.
    
    Enforces isotropic Gaussian distribution on embeddings.
    
    Args:
        num_slices: Number of random projection directions (default: 1024)
        num_points: Quadrature points for Epps-Pulley (default: 17)
    """
    
    def __init__(self, num_slices: int = 1024, num_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.ep_test = EppsPulley(num_points=num_points)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) batch of embedding vectors
        Returns:
            scalar SIGReg loss (0 = perfectly isotropic Gaussian)
        """
        B, D = embeddings.shape
        if B < 4:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Generate random projection directions on unit sphere
        # Shape: (D, K) where K = num_slices
        directions = torch.randn(D, self.num_slices, device=embeddings.device)
        directions = F.normalize(directions, dim=0)  # normalize along D dim
        
        # Project: (B, D) @ (D, K) → (B, K)
        projections = embeddings @ directions
        
        # Apply EP test to each projection (column)
        total_loss = torch.tensor(0.0, device=embeddings.device)
        for k in range(self.num_slices):
            total_loss = total_loss + self.ep_test(projections[:, k])
        
        return total_loss / self.num_slices


class SIGRegVectorized(nn.Module):
    """Vectorized SIGReg — faster version using batch CF computation.
    
    Same math as SIGReg but computes all slices in parallel.
    
    Args:
        num_slices: Number of random projection directions (default: 1024)
        num_points: Quadrature points for EP test (default: 17)
    """
    
    def __init__(self, num_slices: int = 1024, num_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.num_points = num_points
        
        # Quadrature grid
        t_max = 2.0
        t_points = torch.linspace(0, t_max, num_points + 1)[1:]
        self.register_buffer('t_points', t_points)
        
        dt = t_max / num_points
        weights = torch.full((num_points,), dt)
        weights[0] = dt / 2
        weights[-1] = dt / 2
        self.register_buffer('weights', weights)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) batch of embeddings
        Returns:
            scalar SIGReg loss
        """
        B, D = embeddings.shape
        if B < 4:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Random directions: (D, K)
        directions = torch.randn(D, self.num_slices, device=embeddings.device, dtype=embeddings.dtype)
        directions = F.normalize(directions, dim=0)
        
        # Project and standardize: (B, K)
        proj = embeddings @ directions
        proj = (proj - proj.mean(dim=0, keepdim=True)) / (proj.std(dim=0, keepdim=True) + 1e-8)
        
        total = torch.tensor(0.0, device=embeddings.device)
        
        for t_idx, t in enumerate(self.t_points):
            # cos(t * proj): (B, K)
            cos_tp = torch.cos(t * proj)
            sin_tp = torch.sin(t * proj)
            
            # ECF per slice: mean over B → (K,)
            ecf_real = cos_tp.mean(dim=0)
            ecf_imag = sin_tp.mean(dim=0)
            ecf_sq = ecf_real ** 2 + ecf_imag ** 2  # (K,)
            
            # Theoretical CF
            tcf = math.exp(-0.5 * t.item() ** 2)
            
            # |ECF - TCF|² per slice: (K,)
            integrand = ecf_sq - 2 * ecf_real * tcf + tcf ** 2
            
            # Average over slices, weight by quadrature
            total = total + self.weights[t_idx] * integrand.mean()
        
        return total


# =============================================================================
# Factory function
# =============================================================================

def create_sigreg_loss(
    num_slices: int = 1024,
    num_points: int = 17,
    use_package: bool = False,
    vectorized: bool = True,
) -> nn.Module:
    """Create SIGReg loss function.
    
    Args:
        num_slices: Number of random 1D projections (more = better but slower)
                    1024 is default from paper. 256 works fine for small batches.
        num_points: Quadrature points for Epps-Pulley test. 17 = paper default.
        use_package: If True, try to use pip `lejepa` package
        vectorized: If True and not using package, use vectorized implementation
    
    Returns:
        nn.Module that takes (B, D) embeddings → scalar loss
    """
    if use_package:
        try:
            import lejepa
            univariate_test = lejepa.univariate.EppsPulley(num_points=num_points)
            return lejepa.multivariate.SlicingUnivariateTest(
                univariate_test=univariate_test,
                num_slices=num_slices,
            )
        except ImportError:
            print("lejepa package not found, using standalone implementation")
    
    if vectorized:
        return SIGRegVectorized(num_slices=num_slices, num_points=num_points)
    else:
        return SIGReg(num_slices=num_slices, num_points=num_points)


# =============================================================================
# DNA-JEPA integration: combined loss
# =============================================================================

class DNAJEPALoss(nn.Module):
    """Complete DNA-JEPA loss with SIGReg.
    
    L_total = L_pred + λ_sigreg * SIGReg(embeddings)
            + λ_rc * L_rc  (reverse complement consistency)
            + λ_gc * L_gc  (GC content adversarial)
    
    Replaces the old: VICReg + LDReg + SupCon mess.
    
    Args:
        lambda_sigreg: Weight on SIGReg (default 1.0, paper recommends tuning)
        lambda_rc: Weight on reverse complement consistency
        lambda_gc: Weight on GC adversarial debiasing
        num_slices: SIGReg projection directions
        num_points: EP quadrature points
    """
    
    def __init__(
        self,
        lambda_sigreg: float = 1.0,
        lambda_rc: float = 0.1,
        lambda_gc: float = 1.0,
        num_slices: int = 1024,
        num_points: int = 17,
    ):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.lambda_rc = lambda_rc
        self.lambda_gc = lambda_gc
        self.sigreg = create_sigreg_loss(
            num_slices=num_slices,
            num_points=num_points,
            vectorized=True,
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        context_pooled: Optional[torch.Tensor] = None,
        target_pooled: Optional[torch.Tensor] = None,
        rc_pred: Optional[torch.Tensor] = None,
        rc_target: Optional[torch.Tensor] = None,
        gc_loss: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            pred: (M, D) predicted embeddings at masked positions
            target: (M, D) target embeddings at masked positions (detached)
            context_pooled: (B, D) pooled context encoder output
            target_pooled: (B, D) pooled target encoder output
            rc_pred/rc_target: reverse complement pair embeddings
            gc_loss: precomputed GC adversarial loss
        
        Returns:
            (total_loss, metrics_dict)
        """
        metrics = {}
        
        # 1. Prediction loss (MSE) — core JEPA objective
        pred_loss = F.mse_loss(pred, target.detach())
        metrics['pred_loss'] = pred_loss.item()
        
        # 2. SIGReg on predicted embeddings
        sigreg_pred = self.sigreg(pred)
        metrics['sigreg_pred'] = sigreg_pred.item()
        
        # 3. SIGReg on sequence-level pooled representations (if available)
        sigreg_seq = torch.tensor(0.0, device=pred.device)
        if context_pooled is not None:
            sigreg_ctx = self.sigreg(context_pooled)
            metrics['sigreg_ctx'] = sigreg_ctx.item()
            sigreg_seq = sigreg_ctx
        if target_pooled is not None:
            sigreg_tgt = self.sigreg(target_pooled)
            metrics['sigreg_tgt'] = sigreg_tgt.item()
            sigreg_seq = sigreg_seq + sigreg_tgt
        
        # 4. RC consistency (optional)
        rc_loss = torch.tensor(0.0, device=pred.device)
        if rc_pred is not None and rc_target is not None:
            rc_loss = F.mse_loss(rc_pred, rc_target)
            metrics['rc_loss'] = rc_loss.item()
        
        # 5. GC adversarial (optional, precomputed with gradient reversal)
        gc_loss_val = torch.tensor(0.0, device=pred.device)
        if gc_loss is not None:
            gc_loss_val = gc_loss
            metrics['gc_loss'] = gc_loss.item()
        
        # Total
        total = (
            pred_loss
            + self.lambda_sigreg * (sigreg_pred + sigreg_seq)
            + self.lambda_rc * rc_loss
            + self.lambda_gc * gc_loss_val
        )
        metrics['total_loss'] = total.item()
        
        # Monitoring: embedding stats (RankMe, std)
        with torch.no_grad():
            pred_std = pred.float().std(dim=0).mean().item()
            metrics['pred_std'] = pred_std
            
            # RankMe approximation via singular values
            if pred.shape[0] >= pred.shape[1]:
                try:
                    s = torch.linalg.svdvals(pred.float())
                    p = s / s.sum()
                    p = p[p > 1e-10]
                    rankme = torch.exp(-torch.sum(p * torch.log(p))).item()
                    metrics['rankme'] = rankme
                except:
                    metrics['rankme'] = -1.0
        
        # Compat keys for existing W&B logging
        metrics['inv_loss'] = pred_loss.item()
        metrics['var_loss'] = pred_std  # monitoring
        metrics['vicreg_total'] = total.item()
        
        return total, metrics


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("=== SIGReg Standalone Test ===\n")
    
    sigreg = create_sigreg_loss(num_slices=256, vectorized=True)
    
    # Test 1: Gaussian embeddings (should be low loss)
    gaussian = torch.randn(128, 384)
    loss_gaussian = sigreg(gaussian)
    print(f"Gaussian embeddings:  SIGReg = {loss_gaussian.item():.6f}")
    
    # Test 2: Collapsed embeddings (should be high loss)
    collapsed = torch.ones(128, 384) * 0.5 + torch.randn(128, 384) * 0.01
    loss_collapsed = sigreg(collapsed)
    print(f"Collapsed embeddings: SIGReg = {loss_collapsed.item():.6f}")
    
    # Test 3: Uniform embeddings (should be medium loss)
    uniform = torch.rand(128, 384) * 2 - 1
    loss_uniform = sigreg(uniform)
    print(f"Uniform embeddings:   SIGReg = {loss_uniform.item():.6f}")
    
    print(f"\nExpected: collapsed >> uniform > gaussian")
    print(f"Actual:   {loss_collapsed.item():.4f} vs {loss_uniform.item():.4f} vs {loss_gaussian.item():.4f}")
    
    # Test 4: Gradient flows
    x = torch.randn(64, 384, requires_grad=True)
    loss = sigreg(x)
    loss.backward()
    print(f"\nGradient flows: {x.grad is not None}, grad norm: {x.grad.norm().item():.6f}")
    
    # Test 5: Full DNA-JEPA loss
    print("\n=== DNAJEPALoss Test ===\n")
    loss_fn = DNAJEPALoss(lambda_sigreg=1.0, num_slices=256)
    
    pred = torch.randn(256, 384)
    target = torch.randn(256, 384)
    ctx = torch.randn(32, 384)
    tgt = torch.randn(32, 384)
    
    total, metrics = loss_fn(pred, target, ctx, tgt)
    print(f"Total loss: {total.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

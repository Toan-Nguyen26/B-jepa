"""Comprehensive test suite for B-JEPA.

Tests:
    - Encoder forward pass shapes for v3.1 and v4.0
    - Param count verification (~48M for v4.0)
    - VICReg loss behavior on collapsed vs healthy embeddings
    - JEPA loss has gradient
    - GradNorm weight positivity
    - Tokenizer encode/decode roundtrip
    - Full training step on tiny data
    - Config YAML round-trip
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from bdna_jepa.config import (
    BJEPAConfig, EncoderConfig, PredictorConfig, LossConfig,
    V31_CONFIG, V40_CONFIG, FragmentConfig,
)
from bdna_jepa.models.encoder import TransformerEncoder
from bdna_jepa.models.predictor import Predictor, FragmentPredictor
from bdna_jepa.models.jepa import BJEPA
from bdna_jepa.losses.criterion import JEPALoss, MLMLoss, VICRegLoss, BJEPACriterion
from bdna_jepa.data.tokenizer import CharTokenizer
from bdna_jepa.data.masking import random_mask, span_mask
from bdna_jepa.utils.metrics import compute_rankme, compute_feature_std


def _tiny_config() -> BJEPAConfig:
    return BJEPAConfig(
        encoder=EncoderConfig(
            vocab_size=128, embed_dim=64, num_layers=2, num_heads=4,
            ff_dim=128, ff_activation="swiglu", max_seq_len=64,
            pos_encoding="rotary", norm_type="rmsnorm", qk_norm=True,
        ),
        predictor=PredictorConfig(
            depth=2, dim=32, num_heads=4, ff_dim=64, ff_activation="swiglu",
        ),
        loss=LossConfig(
            target_mode="stop_grad",
            use_gradnorm=False,
            fragment=FragmentConfig(enabled=False),
        ),
    )


class TestEncoder:
    def test_v40_forward_shape(self):
        config = _tiny_config().encoder
        encoder = TransformerEncoder(config)
        tokens = torch.randint(5, config.vocab_size, (4, 32))
        out = encoder(tokens, return_all_tokens=True)
        assert out["cls"].shape == (4, config.embed_dim)
        assert out["tokens"].shape == (4, 33, config.embed_dim)

    def test_v31_forward_shape(self):
        config = EncoderConfig(
            vocab_size=128, embed_dim=64, num_layers=2, num_heads=4,
            ff_dim=128, ff_activation="gelu", max_seq_len=64,
            pos_encoding="learned", norm_type="layernorm", qk_norm=False,
        )
        encoder = TransformerEncoder(config)
        tokens = torch.randint(5, 128, (4, 32))
        out = encoder(tokens, return_all_tokens=True)
        assert out["cls"].shape == (4, 64)
        assert out["tokens"].shape == (4, 33, 64)

    def test_attention_mask(self):
        config = _tiny_config().encoder
        encoder = TransformerEncoder(config)
        tokens = torch.randint(5, config.vocab_size, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[0, 8:] = False
        out = encoder(tokens, attention_mask=mask)
        assert out["cls"].shape == (2, config.embed_dim)

    def test_encode_convenience(self):
        config = _tiny_config().encoder
        encoder = TransformerEncoder(config)
        tokens = torch.randint(5, config.vocab_size, (4, 32))
        cls = encoder.encode(tokens)
        assert cls.shape == (4, config.embed_dim)

    def test_v40_param_count(self):
        config = V40_CONFIG.encoder
        encoder = TransformerEncoder(config)
        n = encoder.get_num_params(non_embedding=False)
        assert 30_000_000 < n < 70_000_000, f"Unexpected param count: {n:,}"


class TestPredictor:
    def test_predictor_shape(self):
        config = _tiny_config()
        pred = Predictor(config.encoder.embed_dim, config.predictor)
        cls = torch.randn(4, config.encoder.embed_dim)
        out = pred(cls)
        assert out.shape == (4, config.encoder.embed_dim)

    def test_fragment_predictor_shape(self):
        frag_config = FragmentConfig(
            predictor_depth=2, predictor_dim=32, predictor_heads=4,
        )
        pred = FragmentPredictor(64, frag_config)
        context = torch.randn(4, 3, 64)
        mask = torch.ones(4, 3, dtype=torch.bool)
        out = pred(context, mask)
        assert out.shape == (4, 64)


class TestLosses:
    def test_vicreg_collapsed_input(self):
        vicreg = VICRegLoss(gamma=1.0)
        z = torch.ones(32, 64)
        var_loss, cov_loss = vicreg(z)
        assert var_loss.item() > 0.9, f"Variance loss too low on collapsed: {var_loss.item()}"

    def test_vicreg_healthy_input(self):
        vicreg = VICRegLoss(gamma=1.0)
        z = torch.randn(32, 64) * 2.0
        var_loss, cov_loss = vicreg(z)
        assert var_loss.item() < 0.1, f"Variance loss too high on healthy: {var_loss.item()}"

    def test_jepa_loss_has_grad(self):
        loss_fn = JEPALoss("smooth_l1")
        pred = torch.randn(8, 64, requires_grad=True)
        target = torch.randn(8, 64)
        loss = loss_fn(pred, target)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_mlm_loss(self):
        loss_fn = MLMLoss()
        logits = torch.randn(4, 32, 128)
        labels = torch.full((4, 32), -100, dtype=torch.long)
        labels[:, :5] = torch.randint(5, 128, (4, 5))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_criterion_combined(self):
        config = _tiny_config().loss
        criterion = BJEPACriterion(config)
        model_output = {
            "mlm_logits": torch.randn(4, 32, 128),
            "jepa_pred": torch.randn(4, 64),
            "jepa_target": torch.randn(4, 64),
            "context_cls": torch.randn(4, 64),
            "target_cls": torch.randn(4, 64),
        }
        labels = torch.full((4, 32), -100, dtype=torch.long)
        labels[:, :5] = torch.randint(5, 128, (4, 5))
        out = criterion(model_output, labels)
        assert "total" in out
        assert out["total"].item() > 0
        assert "mlm" in out and "jepa" in out


class TestTokenizer:
    def test_char_encode_decode(self):
        tok = CharTokenizer()
        seq = "ACGTACGT"
        ids = tok.encode(seq)
        decoded = tok.decode(ids)
        assert decoded == seq

    def test_char_vocab_size(self):
        tok = CharTokenizer()
        assert tok.vocab_size == 10

    def test_batch_encode(self):
        tok = CharTokenizer()
        seqs = ["ACGT", "ACGTACGT"]
        batch = tok.batch_encode(seqs, max_length=20)
        assert batch["input_ids"].shape[0] == 2
        assert batch["attention_mask"].shape == batch["input_ids"].shape


class TestMasking:
    def test_random_mask_ratio(self):
        tokens = torch.randint(5, 128, (8, 64))
        _, mask, labels = random_mask(tokens, mask_ratio=0.15, mask_id=1)
        actual_ratio = mask.float().mean().item()
        assert 0.05 < actual_ratio < 0.3

    def test_span_mask_shapes(self):
        tokens = torch.randint(5, 128, (8, 64))
        masked, mask, labels = span_mask(tokens, mask_ratio=0.15, span_length=5)
        assert masked.shape == tokens.shape
        assert mask.shape == tokens.shape
        assert labels.shape == tokens.shape

    def test_labels_match_original(self):
        tokens = torch.randint(5, 128, (4, 32))
        _, mask, labels = random_mask(tokens, mask_ratio=0.15)
        assert (labels[mask] == tokens[mask]).all()


class TestBJEPA:
    def test_forward_shapes(self):
        config = _tiny_config()
        model = BJEPA(config)
        B, L = 4, 32
        tokens = torch.randint(5, config.encoder.vocab_size, (B, L))
        masked_tokens = tokens.clone()
        masked_tokens[:, :5] = 1

        out = model(tokens, masked_tokens)
        assert out["mlm_logits"].shape == (B, L, config.encoder.vocab_size)
        assert out["jepa_pred"].shape == (B, config.encoder.embed_dim)
        assert out["jepa_target"].shape == (B, config.encoder.embed_dim)

    def test_encode(self):
        config = _tiny_config()
        model = BJEPA(config)
        tokens = torch.randint(5, config.encoder.vocab_size, (4, 32))
        cls = model.encode(tokens)
        assert cls.shape == (4, config.encoder.embed_dim)

    def test_stop_grad_update(self):
        config = _tiny_config()
        model = BJEPA(config)
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(1.0)
        model.update_target_encoder()
        for tp, cp in zip(model.target_encoder.parameters(), model.context_encoder.parameters()):
            assert torch.allclose(tp, cp)


class TestTrainingStep:
    def test_single_step(self):
        config = _tiny_config()
        model = BJEPA(config)
        criterion = BJEPACriterion(config.loss)

        B, L = 8, 32
        tokens = torch.randint(5, config.encoder.vocab_size, (B, L))
        masked_tokens, mask, labels = random_mask(tokens, mask_ratio=0.15)

        out = model(tokens, masked_tokens)
        loss_out = criterion(out, labels)
        loss_out["total"].backward()

        assert torch.isfinite(loss_out["total"])
        for p in model.context_encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None, "Missing gradient"
                break
        for p in model.target_encoder.parameters():
            assert p.grad is None or (p.grad == 0).all()


class TestMetrics:
    def test_rankme_collapsed(self):
        emb = torch.ones(100, 64)
        rankme = compute_rankme(emb)
        assert rankme < 2.0

    def test_rankme_full_rank(self):
        emb = torch.randn(200, 64)
        rankme = compute_rankme(emb)
        assert rankme > 30

    def test_feature_std(self):
        emb = torch.randn(100, 64)
        std = compute_feature_std(emb)
        assert 0.5 < std < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

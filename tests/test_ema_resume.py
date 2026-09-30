import copy

import pytest
import torch

from training.audit_overfit import atomic_torch_save, restore_ema


def test_interrupted_training_restores_identical_ema_and_optimizer(tmp_path):
    torch.manual_seed(123)
    model = torch.nn.Linear(3, 2)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    def step(m, e, opt):
        opt.zero_grad()
        m(x).square().mean().backward()
        opt.step()
        with torch.no_grad():
            for a, b in zip(e.parameters(), m.parameters()):
                a.lerp_(b, 0.001)

    for _ in range(5):
        step(model, ema, optimizer)
    path = tmp_path / 'raw.pt'
    atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          ema_model=ema.state_dict(), ema_decay=0.999), path)
    ck = torch.load(path, weights_only=False)
    resumed = torch.nn.Linear(3, 2)
    resumed_ema = copy.deepcopy(resumed).requires_grad_(False)
    opt = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed.load_state_dict(ck['model'])
    opt.load_state_dict(ck['optimizer'])
    restore_ema(resumed_ema, ck, 0.999)
    for _ in range(5):
        step(model, ema, optimizer)
        step(resumed, resumed_ema, opt)
    for a, b in zip(ema.parameters(), resumed_ema.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(a, b)


def test_legacy_or_mismatched_ema_resume_fails_closed():
    ema = torch.nn.Linear(3, 2)
    for ck in ({}, {'ema_model': ema.state_dict(), 'ema_decay': 0.9}):
        with pytest.raises(ValueError, match='Exact EMA resume unavailable'):
            restore_ema(ema, ck, 0.999)

import torch

from training.audit_overfit import atomic_torch_save


def test_atomic_torch_save_replaces_a_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"step": 1, "tensor": torch.arange(3)}, path)
    atomic_torch_save({"step": 2, "tensor": torch.arange(4)}, path)
    loaded = torch.load(path, weights_only=False)
    assert loaded["step"] == 2
    assert torch.equal(loaded["tensor"], torch.arange(4))
    assert not path.with_suffix(".pt.tmp").exists()

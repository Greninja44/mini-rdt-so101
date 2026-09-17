import pytest
import torch
from data.ml_dataset import hold_terminal_actions


def test_hold_padding_preserves_valid_targets_and_does_not_mutate_source():
    actions=torch.arange(24,dtype=torch.float32).reshape(2,4,3)
    mask=torch.tensor([[1,1,0,0],[1,1,1,1]])
    original=actions.clone();held=hold_terminal_actions(actions,mask)
    assert torch.equal(actions,original)
    assert torch.equal(held[mask.bool()],actions[mask.bool()])
    assert torch.equal(held[0,2:],actions[0,1].expand(2,3))
    with pytest.raises(ValueError):hold_terminal_actions(actions,torch.tensor([[1,0,1,0],[1,1,1,1]]))

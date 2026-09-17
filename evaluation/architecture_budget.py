"""Parameter-only future architecture sketches; no forward or training code.

All modules live on the meta device. These counts include biases, norms, input
and output projections, type embeddings, and H=16 learned action positions.
Spatial positions use fixed 2-D sinusoids and carry no trainable parameters.
"""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from models.tiny_rdt import FrozenMobileNet


class ParameterSketch(nn.Module):
    def __init__(self,d,layers,heads,horizon=16):
        super().__init__()
        self.visual_norm=nn.LayerNorm(576)
        self.visual_projection=nn.Linear(576,d)
        self.state_projection=nn.Sequential(nn.Linear(6,d),nn.SiLU(),nn.Linear(d,d))
        self.time_projection=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d))
        self.action_projection=nn.Linear(6,d)
        self.action_position=nn.Parameter(torch.empty(1,horizon,d))
        self.modality=nn.Parameter(torch.empty(4,d))
        self.blocks=nn.ModuleList([nn.TransformerDecoderLayer(d,heads,4*d,dropout=0.,activation="gelu",batch_first=True,norm_first=True) for _ in range(layers)])
        self.output=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,6))


def counts():
    result={}
    for name,d,l,h in (("control_2M",192,3,6),("control_5M",256,4,8),("MiniRDT-S",320,6,10),("MiniRDT-M",384,8,12),("MiniRDT-L",512,9,16)):
        with torch.device("meta"):
            model=ParameterSketch(d,l,h);vision=FrozenMobileNet(pretrained=False)
        policy=sum(p.numel() for p in model.parameters());frozen=sum(p.numel() for p in vision.parameters())
        formula=3*d*d+622*d+1158+l*(16*d*d+19*d)
        assert policy==formula,(policy,formula)
        result[name]={"hidden":d,"layers":l,"heads":h,"head_dim":d//h,"mlp_ratio":4,"horizon":16,
                      "trainable_policy":policy,"frozen_vision":frozen,"total":policy+frozen,
                      "by_module":{name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()},
                      "action_positions_and_modality":20*d}
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="artifacts/research_audit/architecture_counts.json");a=p.parse_args();r=counts();Path(a.output).write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))

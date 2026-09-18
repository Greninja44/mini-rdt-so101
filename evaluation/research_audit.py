"""Reproducible, cached-feature diagnostics. Never executes a learned policy.

All sampling metrics are unclipped and use valid raw action tokens. Audit seed
streams are independent of training and minibatch size. Full tensors for a real
observation are saved to trace.pt; trace.json includes the 16x6 action tensors.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.ml_dataset import ActionWindowDataset, NormalizationStats, hold_terminal_actions
from models.tiny_rdt import TinyRDT, TinyRDTConfig
from training.diffusion import ActionDiffusion

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(checkpoint, device):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    config = TinyRDTConfig(**ck["model_config"])
    # A checkpoint contains encoder weights; loading must not access the net.
    model = TinyRDT(replace(config, pretrained_vision=False)).to(device)
    model.load_state_dict(ck["model"]); model.config = config; model.eval()
    stats = NormalizationStats(**ck["normalization"])
    diffusion = ActionDiffusion.from_config(ck.get("diffusion_config"), device)
    prediction = ck.get("diffusion_config", {}).get("prediction", "epsilon")
    ids = ck.get("train_episode_ids")
    if ids is None:
        ids = ck["splits"]["train"]
        sidecar = Path(checkpoint).with_name("config.json")
        if sidecar.exists():
            n = json.loads(sidecar.read_text()).get("subset_episodes")
            if n: ids = ids[:n]
    return ck, model, stats, diffusion, prediction, ids


@torch.no_grad()
def make_bank(dataset_root, ids, stats, model, device, padding="masked"):
    ds = ActionWindowDataset(dataset_root, ids, model.config.horizon)
    values = {k: [] for k in ("rgb", "state", "raw", "mask", "features", "episode", "t", "phase", "cube")}
    poses = {ep: np.load(Path(dataset_root)/"episodes"/f"episode_{ep:06d}"/"cube_pose.npy") for ep in ids}
    for b in DataLoader(ds, batch_size=64, shuffle=False):
        rgb = b["rgb"].to(device)
        values["rgb"].append(rgb); values["state"].append(b["state"].to(device))
        values["raw"].append(b["actions"].to(device)); values["mask"].append(b["valid_mask"].to(device))
        values["features"].append(model.vision(rgb))
        values["episode"].append(b["episode_id"].to(device)); values["t"].append(b["t"].to(device))
        phase = [min(3,4*int(t)//len(ds._episodes[int(e)]["action"])) for e,t in zip(b["episode_id"],b["t"])]
        values["phase"].append(torch.tensor(phase, device=device))
        values["cube"].append(torch.from_numpy(np.stack([poses[int(e)][int(t)] for e,t in zip(b["episode_id"],b["t"])] )).to(device))
    bank = {k:torch.cat(v) for k,v in values.items()}
    bank["raw_state"] = bank["state"].clone()
    bank["state"] = stats.normalize_state(bank["state"])
    targets = hold_terminal_actions(bank["raw"], bank["mask"]) if padding == "hold" else bank["raw"]
    bank["actions"] = stats.normalize_action(targets)
    bank["model_mask"] = torch.ones_like(bank["mask"]) if padding == "hold" else bank["mask"]
    return ds, bank


@torch.no_grad()
def make_corrective_bank(roots, stats, model, device, max_frames=None, order_seed=0):
    """Corrective samples: actual-state observation -> counterfactual expert chunk.

    Chunks are already hold-padded (label_valid marks the real prefix), which is
    the 'hold' training convention: the model mask is all ones. With
    max_frames, whole episodes are taken in a fixed-seed order until the budget
    is reached (size-matched ablations).
    """
    episodes = sorted(p for r in roots for p in (Path(r) / "episodes").glob("episode_*") if p.is_dir() and not p.name.endswith(".tmp"))
    order = np.random.default_rng(order_seed).permutation(len(episodes)) if max_frames else np.arange(len(episodes))
    chosen, total = [], 0
    for i in order:
        n = len(np.load(episodes[i] / "t.npy"))
        if max_frames and total + n > max_frames and chosen: break
        chosen.append(episodes[i]); total += n
    values = {k: [] for k in ("features", "state", "raw", "mask")}
    for ep in sorted(chosen):
        rgb = torch.from_numpy(np.load(ep / "rgb.npy")).permute(0, 3, 1, 2)
        for lo in range(0, len(rgb), 64): values["features"].append(model.vision(rgb[lo:lo + 64].to(device)))
        values["state"].append(torch.from_numpy(np.concatenate((np.load(ep / "joint_pos.npy"), np.load(ep / "gripper.npy")), 1)).float())
        values["raw"].append(torch.from_numpy(np.load(ep / "label_chunk.npy")).float()); values["mask"].append(torch.from_numpy(np.load(ep / "label_valid.npy")).float())
    bank = {k: torch.cat(v).to(device) for k, v in values.items()}
    bank["state"] = stats.normalize_state(bank["state"]); bank["actions"] = stats.normalize_action(bank["raw"])
    bank["model_mask"] = torch.ones_like(bank["mask"])
    return bank, [str(p) for p in sorted(chosen)]


def metrics(pred, target, mask):
    err = (pred-target)[mask.bool()]
    return {"action_mae": float(err.abs().mean()), "action_mse": float(err.square().mean()),
            "arm_mae_rad": float(err[:,:5].abs().mean()), "gripper_mae": float(err[:,5].abs().mean()),
            "per_joint_mae": err.abs().mean(0).cpu().tolist(),
            "per_joint_mse": err.square().mean(0).cpu().tolist(), "valid_tokens": len(err)}


def subset(bank, indices): return {k:v[indices] for k,v in bank.items()}


@torch.no_grad()
def sample(model, d, prediction, bank, steps=10, seed=4100, ddpm=False, q_start=False, mask_attention=False):
    generator = torch.Generator(device=bank["state"].device).manual_seed(seed)
    noise = torch.randn(bank["actions"].shape, device=bank["state"].device, generator=generator)
    if q_start:
        t = torch.full((len(noise),), d.timesteps-1, device=noise.device, dtype=torch.long)
        noise = d.q_sample(bank["actions"], t, noise)[0]
    outputs = []
    for lo in range(0, len(noise), 64):
        sl = slice(lo, lo+64); features=bank["features"][sl]
        def denoiser(rgb, state, x, t):
            return model(None, state, x, t, bank["mask"][sl] if mask_attention else None, features)
        args = (denoiser, None, bank["state"][sl], model.config.horizon, model.config.action_dim)
        if ddpm: result = d.sample_ddpm(*args, generator=generator, prediction_type=prediction, initial_noise=noise[sl])
        else: result = d.sample_ddim(*args, steps=steps, prediction_type=prediction, initial_noise=noise[sl])
        outputs.append(result)
    return torch.cat(outputs)


@torch.no_grad()
def denoise(model, d, prediction, bank, t_value, seed=7300, features=None, state=None):
    noise = torch.randn(bank["actions"].shape, device=bank["state"].device,
                        generator=torch.Generator(device=bank["state"].device).manual_seed(seed))
    t = torch.full((len(noise),), t_value, dtype=torch.long, device=noise.device)
    x = d.q_sample(bank["actions"], t, noise)[0]
    output = model(None, bank["state"] if state is None else state, x, t, bank["model_mask"], bank["features"] if features is None else features)
    return output, d.model_x0(output, x, t, prediction), d.model_epsilon(output, x, t, prediction), x, noise


@torch.no_grad()
def diagnose(checkpoint, dataset_root, output, device="cpu"):
    started=time.monotonic(); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    ck, model, stats, d, prediction, ids = load(checkpoint, device)
    ds, b = make_bank(dataset_root, ids, stats, model, device, ck.get("run_config",{}).get("padding","masked"))
    # Eight evenly spaced *complete* windows per episode; every demo included.
    chosen=[]
    for ep in ids:
        eligible = torch.where((b["episode"]==ep)&(b["mask"].sum(1)==model.config.horizon))[0].cpu().tolist()
        chosen.extend(eligible[i] for i in np.linspace(0,len(eligible)-1,min(8,len(eligible)),dtype=int))
    ref=subset(b,chosen)
    report={"checkpoint":str(checkpoint), "checkpoint_sha256":sha256(checkpoint), "step":ck["step"],
            "prediction":prediction,"schedule":d.schedule,"device":str(device),"torch":torch.__version__,
            "episodes":ids,"windows":len(ds),"sampler_reference":[ds.entries[i] for i in chosen],
            "sampler_seed":4100,"names":NAMES,"counts":model.parameter_counts(),
            "terminal":{"alpha_bar":float(d.alpha_bars[-1]), "signal_amplitude":float(d.alpha_bars[-1].sqrt()),
                        "snr":float(d.alpha_bars[-1]/(1-d.alpha_bars[-1])),
                        "epsilon_to_x0_mse_amplification":float((1-d.alpha_bars[-1])/d.alpha_bars[-1])}}
    methods={}
    for steps in (5,10,20,50,100):
        pred=stats.denormalize_action(sample(model,d,prediction,ref,steps))
        methods[f"ddim_{steps}"]=metrics(pred,ref["raw"],ref["mask"])
    methods["ddpm_100"]=metrics(stats.denormalize_action(sample(model,d,prediction,ref,ddpm=True)),ref["raw"],ref["mask"])
    methods["ddim_10_q_terminal_start_diagnostic"]=metrics(stats.denormalize_action(sample(model,d,prediction,ref,q_start=True)),ref["raw"],ref["mask"])
    report["samplers"]=methods
    print(json.dumps({"checkpoint":str(checkpoint),"samplers":methods}),flush=True)
    # All training windows, with 3 independent latent seeds. No clipping.
    full=[]; seeds=(4100,4101,4102)
    for seed in seeds:
        pred=stats.denormalize_action(sample(model,d,prediction,b,seed=seed))
        if seed==seeds[0]: saved=pred
        groups={str(q):metrics(pred[b["phase"]==q],b["raw"][b["phase"]==q],b["mask"][b["phase"]==q]) for q in range(4)}
        first=b["t"]==0
        full.append({"seed":seed,"all":metrics(pred,b["raw"],b["mask"]),"quarters":groups,
                     "episode_start":metrics(pred[first],b["raw"][first],b["mask"][first]),
                     "first_action":metrics(pred[:,:1],b["raw"][:,:1],b["mask"][:,:1])})
    report["full_training_windows"]=full
    for label,sel in (("padded",b["mask"].sum(1)<16),("complete",b["mask"].sum(1)==16)):
        report[label]=metrics(saved[sel],b["raw"][sel],b["mask"][sel])
    report["padding_attention_oracle_diagnostic"] = metrics(stats.denormalize_action(sample(model,d,prediction,b,mask_attention=True)),b["raw"],b["mask"])
    # Same fixed epsilon across timesteps, complete reference windows only.
    time_rows=[]
    for ti in range(d.timesteps):
        output,x0,eps,xt,noise=denoise(model,d,prediction,ref,ti)
        valid=ref["mask"].bool()
        time_rows.append({"t":ti,"epsilon_mse":float((eps-noise)[valid].square().mean()),
                          "normalized_x0_mse":float((x0-ref["actions"])[valid].square().mean()),
                          "raw_x0_mse":metrics(stats.denormalize_action(x0),ref["raw"],ref["mask"])["action_mse"]})
    report["timestep_errors"]=time_rows
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,("epsilon_mse","raw_x0_mse")):
        ax.semilogy([r["t"] for r in time_rows],[r[key] for r in time_rows]);ax.set(xlabel="t (0-based)",ylabel=key);ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/"timestep_errors.png",dpi=140);plt.close(fig)
    # Conditioning ablations: fixed input/noise; a deterministic derangement.
    cond={}; perm=torch.roll(torch.arange(len(ref["state"]),device=device),len(ref["state"])//2+1)
    for ti in (0,d.timesteps//2,d.timesteps-1):
        correct=denoise(model,d,prediction,ref,ti)
        settings={"correct":(ref["features"],ref["state"]),
                  "wrong_image":(ref["features"][perm],ref["state"]),
                  "wrong_state":(ref["features"],ref["state"][perm]),
                  "wrong_both":(ref["features"][perm],ref["state"][perm]),
                  "zero_features":(torch.zeros_like(ref["features"]),ref["state"]),
                  "zero_normalized_state":(ref["features"],torch.zeros_like(ref["state"]))}
        cond[str(ti)]={}
        for name,(features,state) in settings.items():
            result=denoise(model,d,prediction,ref,ti,features=features,state=state)
            cond[str(ti)][name]={**metrics(stats.denormalize_action(result[1]),ref["raw"],ref["mask"]),
                               "model_output_mae_change":float((result[0]-correct[0]).abs().mean())}
    report["conditioning"]=cond
    # Early localization: same phase, different cube positions, not shuffled
    # phases where the robot itself is a shortcut for the correct trajectory.
    starts=subset(b,torch.where(b["t"]==0)[0]); swapped=dict(starts);swapped["features"]=starts["features"].roll(1,0)
    report["start_image_shuffle_sampling"]=metrics(stats.denormalize_action(sample(model,d,prediction,swapped)), starts["raw"],starts["mask"])
    probe=subset(b,[0]); image=probe["rgb"]
    model.train(); train_features=model.vision(image).clone(); model.eval();eval_features=model.vision(image)
    scaled=image.float()/255
    mean=scaled.new_tensor([.485,.456,.406])[None,:,None,None];std=scaled.new_tensor([.229,.224,.225])[None,:,None,None]
    pixels=(scaled-mean)/std;fmap=model.vision.features(pixels)
    report["vision"]={"train_eval_feature_max_abs":float((train_features-eval_features).abs().max()),
                      "map_shape":list(fmap.shape),"feature_shape":list(eval_features.shape),
                      "pixel_range":[float(pixels.min()),float(pixels.max())],
                      "trainable_backbone_parameters":sum(p.numel() for p in model.vision.parameters() if p.requires_grad)}
    output,x0,eps,xt,noise=denoise(model,d,prediction,probe,d.timesteps//2)
    sampled=sample(model,d,prediction,probe)
    trace={"raw_rgb":image.cpu(),"raw_state":probe["raw_state"].cpu(),"raw_action_chunk":probe["raw"].cpu(),
           "normalized_state":probe["state"].cpu(),"normalized_action":probe["actions"].cpu(),
           "preprocessed_pixels":pixels.cpu(),"vision_map":fmap.cpu(),"vision_features":eval_features.cpu(),
           "noise":noise.cpu(),"noisy_actions":xt.cpu(),"timestep":torch.tensor([d.timesteps//2]),
           "model_prediction":output.cpu(),"reconstructed_x0":x0.cpu(),"sampled_normalized_actions":sampled.cpu(),
           "denormalized_actions":stats.denormalize_action(sampled).cpu(),"valid_mask":probe["mask"].cpu()}
    torch.save(trace,out/"trace.pt")
    def summary(x):
        r={"shape":list(x.shape),"dtype":str(x.dtype),"min":float(x.min()),"max":float(x.max())}
        if x.numel()<=600:r["values"]=x.tolist()
        return r
    save_json(out/"trace.json",{"episode":ids[0],"t":0,"tensors":{k:summary(v) for k,v in trace.items()}})
    plt.imsave(out/"trace_rgb.png",image[0].permute(1,2,0).cpu().numpy())
    torch.save({"predictions":saved.cpu(),"targets":b["raw"].cpu(),"mask":b["mask"].cpu(),"entries":ds.entries},out/"sampled_actions.pt")
    report["elapsed_s"]=time.monotonic()-started
    save_json(out/"diagnostics.json",report)
    print(json.dumps({"output":str(out),"all_seeds":[x["all"] for x in full],"elapsed":report["elapsed_s"]}),flush=True)
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--dataset",default="artifacts/pickcube_smoke100_rgb160");p.add_argument("--output",required=True);p.add_argument("--device",default="cpu")
    args=p.parse_args();torch.set_num_threads(2);diagnose(args.checkpoint,args.dataset,args.output,args.device)

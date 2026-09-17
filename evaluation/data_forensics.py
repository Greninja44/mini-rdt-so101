"""Read-only dataset integrity, action statistics, and deterministic replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from data.ml_dataset import make_episode_splits, compute_normalization, ActionWindowDataset
from data.validate import validate_dataset
from evaluation.research_audit import save_json, sha256


def run(root, output, replay=True):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    splits=make_episode_splits(root,17);ids=splits.train[:10];stats=compute_normalization(root,ids)
    metadata={};arrays={};digest={};errors=validate_dataset(root)
    for path in sorted((root/"episodes").glob("episode_*")):
        m=json.loads((path/"episode.json").read_text());ep=m["episode_index"];metadata[ep]=m
        arrays[ep]={k:np.load(path/f"{k}.npy",mmap_mode="r") for k in ("action","joint_pos","gripper","cube_pose","timestamp","expert_state","success")}
        a=arrays[ep];n=m["length"]
        if any(len(v)!=n for v in a.values()):errors.append(f"{ep}: metadata length")
        if not np.allclose(a["timestamp"],np.arange(n)/20,rtol=0,atol=1e-10):errors.append(f"{ep}: timing")
        if np.flatnonzero(a["success"]).tolist()!=[n-1]:errors.append(f"{ep}: outcome alignment")
        if m["joint_order"]!=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]:errors.append(f"{ep}: joint order")
        digest[ep]=sha256(path/"action.npy")
    ds=ActionWindowDataset(root,ids);alignment=0.;padding=0
    for i,(ep,t) in enumerate(ds.entries):
        v=ds[i];n=int(v["valid_mask"].sum());alignment=max(alignment,float(np.abs(v["actions"][:n].numpy()-arrays[ep]["action"][t:t+n]).max()))
        padding+=int(n<16)
    def describe(x):
        return {"mean":x.mean(0).tolist(),"std":x.std(0).tolist(),"min":x.min(0).tolist(),"max":x.max(0).tolist(),"n":len(x)}
    quarters={str(i):[] for i in range(4)};deltas=[];seq_deltas=[];all_actions=[];home_chunks=[];home_xy=[];duplicates=[]
    for ep in ids:
        a=arrays[ep];actions=np.asarray(a["action"]);state=np.concatenate((a["joint_pos"],a["gripper"]),1);n=len(actions)
        all_actions.append(actions);deltas.append(actions-state);seq_deltas.append(np.diff(actions,axis=0));home_chunks.append(actions[:16]);home_xy.append(a["cube_pose"][0,:2])
        for q in range(4):quarters[str(q)].append(actions[(np.arange(n)*4//n)==q])
        for t in range(n-1):
            if np.max(np.abs(state[t+1]-state[t]))<.001:
                duplicates.append({"episode":ep,"t":t,"state_max_change":float(np.max(np.abs(state[t+1]-state[t]))),"action_max_change":float(np.max(np.abs(actions[t+1]-actions[t]))),"expert_states":[str(a["expert_state"][t]),str(a["expert_state"][t+1])]})
    action=np.concatenate(all_actions);home_chunks=np.stack(home_chunks);home_xy=np.stack(home_xy)
    correlations=np.corrcoef(np.concatenate((home_xy,home_chunks[:,-1,:5]),1).T)[:2,2:]
    horizons={}
    for h in (8,16,32):
        windows=ActionWindowDataset(root,ids,horizon=h)
        horizons[str(h)]={"seconds":h/20,"windows":len(windows),"padded_windows":sum(int(t+h>len(arrays[e]["action"])) for e,t in windows.entries),
                         "mean_valid_fraction":float(np.mean([min(h,len(arrays[e]["action"])-t)/h for e,t in windows.entries]))}
    report={"validation_errors":errors,"episodes":len(metadata),"frames":sum(m["length"] for m in metadata.values()),
            "episode_lengths":describe(np.array([m["length"] for m in metadata.values()])[:,None]),
            "split_counts":{k:len(getattr(splits,k)) for k in ("train","validation","test")},
            "split_frames":{k:sum(metadata[e]["length"] for e in getattr(splits,k)) for k in ("train","validation","test")},
            "overfit_ids":ids,"overfit_frames":len(ds),"padded_windows":padding,"window_alignment_max_error":alignment,
            "normalization":stats.__dict__,"absolute_targets":describe(action),"action_minus_observed_state":describe(np.concatenate(deltas)),
            "successive_action_deltas":describe(np.concatenate(seq_deltas)),"quarters":{q:describe(np.concatenate(v)) for q,v in quarters.items()},
            "initial_cube_xy_vs_h15_action_correlations":[[float(v) if np.isfinite(v) else None for v in row] for row in correlations],
            "initial_chunk_across_episode_std":home_chunks.std(0).tolist(),"near_static_adjacent_observations":duplicates,
            "horizons":horizons,"unique_action_files":len(set(digest.values())),"unique_seeds":len(set(m["seed"] for m in metadata.values()))}
    if replay:
        from simulation.env import SO101PickCubeEnv, PickCubeConfig
        from simulation.expert import PickCubeExpert
        import mujoco
        ep=ids[0];a=arrays[ep];meta=metadata[ep];recorded_rgb=np.load(root/"episodes"/f"episode_{ep:06d}"/"rgb.npy",mmap_mode="r")
        env=SO101PickCubeEnv(PickCubeConfig());obs,info=env.reset(seed=meta["seed"]);expert=PickCubeExpert(env);expert.reset()
        errs={k:0. for k in ("state","cube","expert_action","rgb","applied_action")};outcomes=[];states_match=True
        try:
            for t in range(meta["length"]):
                current=np.r_[obs["joint_pos"],obs["gripper"]];recorded=np.r_[a["joint_pos"][t],a["gripper"][t]]
                errs["state"]=max(errs["state"],float(abs(current-recorded).max()))
                errs["cube"]=max(errs["cube"],float(abs(env.cube_pose-a["cube_pose"][t]).max()))
                errs["rgb"]=max(errs["rgb"],float(abs(obs["rgb"].astype(float)-recorded_rgb[t]).max()))
                command=expert.action();errs["expert_action"]=max(errs["expert_action"],float(abs(command-a["action"][t]).max()))
                states_match &= expert.state.value==str(a["expert_state"][t])
                obs,_,terminated,truncated,info=env.step(a["action"][t]);expert.observe(info)
                errs["applied_action"]=max(errs["applied_action"],float(abs(info["applied_action"]-a["action"][t]).max()))
                outcomes.append(bool(info["success"]))
            report["replay"]={"episode":ep,"seed":meta["seed"],"max_abs_errors":errs,"expert_states_match":states_match,
                              "outcome_sequence_matches":outcomes==a["success"].tolist(),"terminal_success":bool(info["success"]),
                              "actuator_joint_names":[mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_JOINT,int(x)) for x in env.model.actuator_trnid[:,0]]}
        finally:env.close()
    save_json(out/"data_forensics.json",report)
    print(json.dumps({k:report[k] for k in ("validation_errors","frames","overfit_frames","padded_windows","window_alignment_max_error","replay") if k in report},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="artifacts/pickcube_smoke100_rgb160");p.add_argument("--output",default="artifacts/research_audit");p.add_argument("--no-replay",action="store_true");a=p.parse_args();run(a.dataset,a.output,not a.no_replay)

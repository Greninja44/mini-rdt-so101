"""Bounded, controlled ten-demo experiments, with exact resumable RNG state.

Defaults to the ten-demo CLEAN10 subset; --train-episodes widens it (training split only). It keeps the original
2,009,670 trainable parameter TinyRDT and uses independent random streams for
minibatches, corruption, and evaluation. Diagnostic BC models share its windows.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from data.ml_dataset import episode_ids, make_episode_splits, compute_normalization
from models.tiny_rdt import TinyRDT, TinyRDTConfig
from training.diffusion import ActionDiffusion
from training.trainer import masked_mse, set_seed
from evaluation.research_audit import load, make_bank, make_corrective_bank, metrics, sample, denoise, save_json, sha256


def experiment(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/"last.pt").exists() and not args.resume:
        raise FileExistsError("Use a new run directory or explicitly resume; previous checkpoints are preserved")
    torch.set_num_threads(2);set_seed(args.seed)
    device=torch.device(args.device)
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    source,vision_model,_,_,_,_=load(args.encoder_checkpoint,device)
    vision_model.vision.spatial=args.vision_tokens=="spatial"  # same frozen weights; token layout only
    splits=make_episode_splits(args.dataset,args.seed);ids=episode_ids(args.dataset) if args.train_all else splits.train[:args.train_episodes]
    if args.train_subset:  # spatial generalization split (data/generalization_split.py); held-out positions are never loaded
        ids=json.loads(Path(args.split_file).read_text())["train_subsets"][args.train_subset]
    stats=compute_normalization(args.dataset,ids)
    ds,b=make_bank(args.dataset,ids,stats,vision_model,device,args.padding)
    cb=None;corrective_episodes=[]
    if args.corrective:
        if args.baseline or args.padding!="hold": raise ValueError("corrective data uses hold-padded diffusion training")
        cb,corrective_episodes=make_corrective_bank(args.corrective,stats,vision_model,device,args.corrective_max_frames,args.seed)
        n_corrective=round(args.batch_size*args.corrective_fraction)
        if not 0<n_corrective<args.batch_size: raise ValueError("corrective fraction must leave clean and corrective samples in each batch")
    set_seed(args.seed)  # Encoder extraction does not consume training RNG.
    config=TinyRDTConfig(pretrained_vision=False,vision_tokens=args.vision_tokens,state_dropout=args.state_dropout,train_vision=args.train_vision,
                         hidden_dim=args.hidden_dim,layers=args.layers,heads=args.heads)
    if args.train_vision and (args.corrective or args.baseline): raise ValueError("train-vision supports CLEAN diffusion training only")
    if args.baseline:
        if args.baseline=="rgb": inputs=torch.cat((b["features"],b["state"]),1)
        elif args.baseline=="state": inputs=b["state"]
        else:
            cube_mean=b["cube"].mean(0);cube_std=b["cube"].std(0,unbiased=False).clamp_min(.01)
            inputs=torch.cat((b["state"],(b["cube"]-cube_mean)/cube_std),1)
        model=nn.Sequential(nn.Linear(inputs.shape[1],256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,96)).to(device)
        d=None
    else:
        model=TinyRDT(config).to(device)
        model.vision.load_state_dict(vision_model.vision.state_dict())
        d=ActionDiffusion(schedule=args.schedule,device=device)
    if args.train_vision:
        vision_params=list(model.vision.parameters()); ids_v={id(p) for p in vision_params}
        groups=[{"params":[p for p in model.parameters() if p.requires_grad and id(p) not in ids_v],"lr":args.learning_rate},
                {"params":[p for p in vision_params if p.requires_grad],"lr":args.vision_learning_rate}]
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    else:
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.learning_rate,weight_decay=1e-4)
    # EMA consumes no RNG, so raw weights follow exactly the non-EMA trajectory.
    ema=copy.deepcopy(model).eval().requires_grad_(False) if args.ema_decay else None
    train_rng=torch.Generator(device=device).manual_seed(args.seed+100)
    noise_rng=torch.Generator(device=device).manual_seed(args.seed+200)
    drop_rng=torch.Generator(device=device).manual_seed(args.seed+300)  # separate stream: other streams stay identical to A
    started=time.monotonic();best=float("inf");start_step=0
    if args.init_checkpoint:
        initial=torch.load(args.init_checkpoint,map_location=device,weights_only=False)
        if initial.get("train_episode_ids")!=ids or initial["normalization"]!=asdict(stats):
            raise ValueError("initial checkpoint dataset/normalization mismatch")
        if initial["diffusion_config"]["prediction"]!=args.prediction or initial["diffusion_config"]["schedule"]!=args.schedule:
            raise ValueError("initial checkpoint noise process mismatch")
        if initial["run_config"].get("padding","masked")!=args.padding:
            raise ValueError("initial checkpoint padding mismatch")
        model.load_state_dict(initial["model"])
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False)
        for key in ("baseline","schedule","prediction","batch_size","seed","learning_rate","padding"):
            if ck["run_config"].get(key,"masked" if key=="padding" else None)!=getattr(args,key):raise ValueError(f"resume mismatch: {key}")
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        train_rng.set_state(ck["train_rng"].cpu());noise_rng.set_state(ck["noise_rng"].cpu())
        torch.set_rng_state(ck["torch_rng"].cpu())
        if device.type=="cuda":torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        start_step=ck["step"]+1;best=ck["best_validation"]
    config_out=vars(args).copy();config_out.update({"corrective_frames":0 if cb is None else len(cb["state"]),"corrective_episodes":corrective_episodes,"train_episode_ids":ids,"windows":len(ds),"precision":"float32", "encoder_checkpoint_sha256":sha256(args.encoder_checkpoint),"torch":torch.__version__})
    save_json(out/"config.json",config_out);stats.save(out/"normalization.json");splits.save(out/"splits.json")
    # Gate registered before training: per-joint units avoid hiding failure in
    # small-variance wrist channels. All windows and episode starts must pass.
    save_json(out/"gate_definition.json",{"all_window_mae":.02,"each_arm_joint_mae_rad":.03,"gripper_mae":.03,
                                          "each_quarter_mae":.025,"episode_start_mae":.025,"sampling_seeds":[4100,4101,4102],
                                          "selection":"fixed 5000-step comparison; final dense gate evaluated separately"})
    # Tensor/source hashes support audit continuation without a working .git.
    manifest={str(p):sha256(p) for folder in ("training","models","data","evaluation") for p in Path(folder).glob("*.py")}
    data_manifest={str(p):sha256(p) for ep in ids for p in (Path(args.dataset)/"episodes"/f"episode_{ep:06d}").glob("*") if p.is_file()}
    save_json(out/"provenance.json",{"source":manifest,"data":data_manifest})
    with (out/"metrics.jsonl").open("a") as log:
        for step in range(start_step,args.steps):
            ix=torch.randint(len(ds),(args.batch_size,),device=device,generator=train_rng)
            model.train();optimizer.zero_grad(set_to_none=True)
            if args.baseline:
                prediction=model(inputs[ix]).reshape(-1,16,6)
                loss=masked_mse(prediction,b["actions"][ix],b["model_mask"][ix])
            else:
                keys=("actions","state","features","model_mask")
                if cb is None: batch={k:b[k][ix] for k in keys}
                else:
                    ixc=torch.randint(len(cb["state"]),(n_corrective,),device=device,generator=train_rng)
                    batch={k:torch.cat((b[k][ix[:args.batch_size-n_corrective]],cb[k][ixc])) for k in keys}
                if args.train_vision: batch["features"]=model.vision(b["rgb"][ix])  # live encoder, gradients flow
                t=torch.randint(0,100,(args.batch_size,),device=device,generator=noise_rng)
                noise=torch.randn(batch["actions"].shape,device=device,generator=noise_rng)
                x,_=d.q_sample(batch["actions"],t,noise)
                target=noise if args.prediction=="epsilon" else batch["actions"]
                drop=(torch.rand(args.batch_size,device=device,generator=drop_rng)<args.state_dropout) if args.state_dropout>0 else None
                prediction=model(None,batch["state"],x,t,batch["model_mask"],batch["features"],drop)
                loss=masked_mse(prediction,target,batch["model_mask"])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for e,p in zip(ema.state_dict().values(),model.state_dict().values()):
                        if e.dtype.is_floating_point: e.lerp_(p,1-args.ema_decay)
                        else: e.copy_(p)
            if step % args.eval_interval == 0 or step == args.steps-1:
                model.eval()
                if args.train_vision:  # cached features are stale once the encoder trains
                    with torch.no_grad(): b["features"]=torch.cat([model.vision(b["rgb"][i:i+64]) for i in range(0,len(b["rgb"]),64)])
                with torch.no_grad():
                    if args.baseline:
                        output=model(inputs).reshape(-1,16,6);teacher=None
                    else:
                        output=sample(model,d,args.prediction,b,seed=4100)
                        _,x0,eps,_,noise=denoise(model,d,args.prediction,b,50)
                        teacher={"epsilon_mse":float((eps-noise)[b["mask"].bool()].square().mean()),
                                 **metrics(stats.denormalize_action(x0),b["raw"],b["mask"])}
                    result=metrics(stats.denormalize_action(output),b["raw"],b["mask"])
                    row={"step":step,"loss":float(loss),"sampled":result,"teacher_t50":teacher,"elapsed_s":time.monotonic()-started}
                log.write(json.dumps(row)+"\n");log.flush();print(json.dumps(row),flush=True)
                improved=result["action_mse"]<best
                if improved:best=result["action_mse"]
                payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"best_validation":best,
                         "model_config":asdict(config),"normalization":asdict(stats),"splits":asdict(splits),"train_episode_ids":ids,
                         "diffusion_config":{"timesteps":100,"schedule":args.schedule,"beta_start":1e-4,"beta_end":.02,"prediction":args.prediction,"sampling_steps":10},
                         "run_config":vars(args),"seed":args.seed,"action_horizon":16,"action_representation":"five absolute MJCF radians + normalized opening",
                         "train_rng":train_rng.get_state(),"noise_rng":noise_rng.get_state(),"torch_rng":torch.get_rng_state(),
                         "cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else [],"elapsed_s":time.monotonic()-started}
                if args.baseline:
                    payload.update({"baseline":args.baseline,"input_dim":inputs.shape[1],"policy_parameters":sum(p.numel() for p in model.parameters()),"vision":vision_model.vision.state_dict()})
                    if args.baseline=="privileged":payload.update(cube_mean=cube_mean,cube_std=cube_std)
                torch.save(payload,out/"last.pt")
                if improved:torch.save(payload,out/"best.pt")
                if ema is not None:
                    ema_payload={k:v for k,v in payload.items() if k!="optimizer"};ema_payload["model"]=ema.state_dict()
                    ema_payload["ema_decay"]=args.ema_decay
                    with torch.no_grad():
                        eb=b
                        if args.train_vision: eb={**b,"features":torch.cat([ema.vision(b["rgb"][i:i+64]) for i in range(0,len(b["rgb"]),64)])}
                        ema_row=metrics(stats.denormalize_action(sample(ema,d,args.prediction,eb,seed=4100)),b["raw"],b["mask"])
                    log.write(json.dumps({"step":step,"ema_sampled":ema_row})+"\n");log.flush();print(json.dumps({"step":step,"ema_sampled":ema_row}),flush=True)
                    torch.save(ema_payload,out/"ema_last.pt")
                    if args.snapshot_interval and step and step%args.snapshot_interval==0: torch.save(ema_payload,out/f"ema_step{step:06d}.pt")
    save_json(out/"result.json",{**row,"best_sampled_mse":best,"device":str(device),"train_episode_ids":ids,
                                "policy_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"peak_vram_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else 0})


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="artifacts/pickcube_smoke100_rgb160");p.add_argument("--output",required=True)
    p.add_argument("--encoder-checkpoint",default="artifacts/tiny_rdt_overfit10/tiny_rdt_best.pt")
    p.add_argument("--schedule",choices=("linear","cosine"),default="cosine");p.add_argument("--prediction",choices=("epsilon","x0"),default="x0")
    p.add_argument("--baseline",choices=("rgb","state","privileged"));p.add_argument("--steps",type=int,default=5000);p.add_argument("--seed",type=int,default=17)
    p.add_argument("--padding",choices=("masked","hold"),default="masked")
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--learning-rate",type=float,default=.001);p.add_argument("--device",default="cpu");p.add_argument("--eval-interval",type=int,default=1000);p.add_argument("--resume");p.add_argument("--init-checkpoint")
    p.add_argument("--train-all",action="store_true",help="train on every successful episode of --dataset (a training-only dataset such as varied_start10)")
    p.add_argument("--train-episodes",type=int,default=10,help="first N episodes of the seed-17 TRAINING split (validation/test never used)")
    p.add_argument("--train-vision",action="store_true",help="fine-tune MobileNet weights (BN stats frozen)")
    p.add_argument("--vision-learning-rate",type=float,default=1e-4)
    p.add_argument("--state-dropout",type=float,default=0.,help="training-only: P(state token -> learned null)")
    p.add_argument("--vision-tokens",choices=("pooled","spatial"),default="pooled")
    p.add_argument("--corrective",nargs="+",help="corrective dataset roots (data/collect_corrective.py)")
    p.add_argument("--corrective-fraction",type=float,default=.5)
    p.add_argument("--corrective-max-frames",type=int,help="size-matched ablation: whole episodes up to this many frames")
    p.add_argument("--ema-decay",type=float,default=0.,help="0 disables; EMA weights saved to ema_last.pt")
    p.add_argument("--hidden-dim",type=int,default=192,help="TinyRDT width (capacity scaling); default = the ~2M baseline")
    p.add_argument("--layers",type=int,default=4); p.add_argument("--heads",type=int,default=6)
    p.add_argument("--split-file",default="docs/research/physics_v2_generalization_split.json")
    p.add_argument("--train-subset",choices=("CLEAN10","TRAIN20","TRAIN40","TRAIN80"),help="train on this spatial-split subset (overrides --train-episodes)")
    p.add_argument("--snapshot-interval",type=int,default=0,help="also keep EMA snapshots every N steps (must be a multiple of --eval-interval)")
    experiment(p.parse_args())

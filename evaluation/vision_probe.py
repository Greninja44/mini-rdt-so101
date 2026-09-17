"""Initial-frame localization probes: pooled versus spatial frozen features.

Robot pose is constant at reset. Train=80 episodes, validation=10, test sealed.
Ridge regularization is selected on an inner 60/20 split of TRAIN episodes.
This probes feature information; it does not train a manipulation policy.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights
from data.ml_dataset import make_episode_splits
from evaluation.research_audit import load,save_json


def predict(x,y,query,lam):
    mean=x.mean(0);std=x.std(0).clip(.001);z=(x-mean)/std;zq=(query-mean)/std
    ym=y.mean(0);scale=z.shape[1]
    weights=np.linalg.solve(z@z.T/scale + lam*np.eye(len(z)), y-ym)
    return ym+zq@z.T/scale@weights


@torch.no_grad()
def run(root,output,checkpoint):
    torch.set_num_threads(2);device="cpu";_,model,_,_,_,_=load(checkpoint,device)
    split=make_episode_splits(root,17);ids=split.train+split.validation
    images=[];xy=[];states=[];centroids=[];pixel_counts=[]
    for ep in ids:
        path=Path(root)/"episodes"/f"episode_{ep:06d}"
        im=np.load(path/"rgb.npy",mmap_mode="r")[0].copy();images.append(im)
        xy.append(np.load(path/"cube_pose.npy")[0,:2]);states.append(np.load(path/"joint_pos.npy")[0])
        red=(im[:,:,0]>60)&(im[:,:,0].astype(float)>1.6*im[:,:,1])&(im[:,:,0].astype(float)>1.6*im[:,:,2])
        yy,xx=np.where(red);centroids.append([xx.mean(),yy.mean()]);pixel_counts.append(len(xx))
    images=torch.from_numpy(np.stack(images)).permute(0,3,1,2)
    means=images.new_tensor([0,0,0]) # Image normalization is floating point below.
    fmap=[];official=[]
    for batch in images.split(16):
        x=batch.float()/255.;x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        fmap.append(model.vision.features(x))
        official.append(model.vision.features(MobileNet_V3_Small_Weights.DEFAULT.transforms()(batch)).mean((2,3)))
    fmap=torch.cat(fmap);xy=np.array(xy,dtype=np.float64)
    inputs={"pooled_576":fmap.mean((2,3)).numpy(),"spatial_20x576":fmap.flatten(1).numpy(),
            "official_resize_crop_pooled":torch.cat(official).numpy(),"red_centroid":np.array(centroids)}
    rng=np.random.default_rng(17);perm=rng.permutation(80);train=perm[:60];inner=perm[60:]
    report={"train_ids":split.train,"validation_ids":split.validation,"test_used":False,"feature_map_shape":list(fmap.shape),
            "rgb_shape":list(images.shape),"cube_red_pixels":{"min":min(pixel_counts),"max":max(pixel_counts),"mean":float(np.mean(pixel_counts))},
            "initial_robot_state_std":np.array(states).std(0).tolist(),"probes":{}}
    for name,x in inputs.items():
        x=x.astype(np.float64);errors={}
        for lam in (1e-6,1e-4,.01,1.,100.):
            pred=predict(x[train],xy[train],x[inner],lam);errors[lam]=float(((pred-xy[inner])**2).mean())
        chosen=min(errors,key=errors.get);pred=predict(x[:80],xy[:80],x[80:],chosen)
        report["probes"][name]={"ridge_lambda":chosen,"inner_mse":errors[chosen],"validation_xy_mae_mm":(abs(pred-xy[80:]).mean(0)*1000).tolist(),
                                 "validation_distance_mm":float(np.linalg.norm(pred-xy[80:],axis=1).mean()*1000)}
    report["constant_train_mean_validation_distance_mm"]=float(np.linalg.norm(xy[:80].mean(0)-xy[80:],axis=1).mean()*1000)
    report["weights_transform"]=str(MobileNet_V3_Small_Weights.DEFAULT.transforms())
    save_json(output,report);print(report)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="artifacts/pickcube_smoke100_rgb160");p.add_argument("--output",default="artifacts/research_audit/vision_probe.json");p.add_argument("--checkpoint",default="artifacts/tiny_rdt_overfit10/tiny_rdt_best.pt");a=p.parse_args();run(a.dataset,a.output,a.checkpoint)

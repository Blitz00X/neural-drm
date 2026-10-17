import json
from pathlib import Path
import numpy as np
import torch
from ..eval import evaluate,high_fidelity
from ..io import save_json

def save(path,obj):
    save_json(path,obj)

def fixed_wrong_keys(secret,seed):
    rng=np.random.default_rng(seed+8021)
    k=np.concatenate([-np.ones((1,64)),rng.integers(0,2,(5,64))*2-1]).astype(np.float32)
    if np.any(np.all(k==secret,axis=1)): raise ValueError('Fixed attack key is valid: contaminated trial')
    return k

def all_key_metrics(model,x,target,attack_keys):
    scores=[evaluate(model,x,target,k) for k in attack_keys]
    return {'keys':scores,'success':all(high_fidelity(s) for s in scores),
            'worst_r2':min(s['r2'] for s in scores),'worst_acc':min(s['acc'] for s in scores)}

def activations(model,x,k,layer=1):
    device=next(model.parameters()).device
    with torch.no_grad():
        return model(torch.as_tensor(x,device=device),torch.as_tensor(k,device=device).expand(len(x),-1),True)[1][layer].detach()

def sensitivity(model,x,valid_key,wrong_key):
    device=next(model.parameters()).device
    with torch.no_grad():
        xx=torch.as_tensor(x,device=device)
        a=model(xx,torch.as_tensor(valid_key,device=device).expand(len(xx),-1))
        b=model(xx,torch.as_tensor(wrong_key,device=device).expand(len(xx),-1))
        # Height difference and total variation in class probabilities.
        return {'height_mse':float(((a[:,0]-b[:,0])**2).mean()),
            'biome_tv':float((a[:,1:].softmax(1)-b[:,1:].softmax(1)).abs().sum(1).mean()/2)}

"""Privileged activation mean-ablation diagnostic, not secret-free key recovery."""
import numpy as np
import torch
from .common import activations,sensitivity,all_key_metrics,save
from ..eval import evaluate

def rank_units(model,x,secret,wrong):
    means=[]; scored=[]
    for layer in range(4):
        a=activations(model,x,secret,layer); b=activations(model,x,wrong,layer)
        mean=a.mean(0); means.append(mean)
        # Raw mean difference, as specified. Layer scales are not normalized.
        score=(mean-b.mean(0)).abs().cpu().numpy()
        scored.extend((float(v),layer,i) for i,v in enumerate(score))
    scored.sort(reverse=True)
    return [(layer,i) for _,layer,i in scored],means

def set_ablation(model,order,means,m):
    model.interventions.clear()
    for layer in range(4):
        idx=[i for l,i in order[:m] if l==layer]
        if idx: model.interventions[layer]=(torch.tensor(idx,device=means[layer].device),means[layer])

def sweep(model,calibration,x,target,secret,attack_keys,path):
    order,means=rank_units(model,calibration,secret,attack_keys[0]); rows=[]
    try:
        for m in range(0,int(.2*len(order))+1):
            set_ablation(model,order,means,m)
            rows.append({'m':m,'fraction':m/len(order),
                'sensitivity':sensitivity(model,calibration,secret,attack_keys[0]),
                'valid':evaluate(model,x,target,secret),
                'bypass':all_key_metrics(model,x,target,attack_keys)})
            save(path,{'rows':rows,'units':len(order),'privileged':True})
    finally: model.interventions.clear()
    return rows

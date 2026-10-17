"""Oracle sample curves; validation selects LR, test is never used for selection."""
import copy,time
import numpy as np
import torch
from ..data import world
from ..models import WorldMLP
from ..train import fit,seed_all
from ..eval import evaluate,training_flops
from .common import save,all_key_metrics

def sample_curve(base,arm,config,unused_train,v,t,folder,seed,attack_keys=None):
    rng=np.random.default_rng(seed+951); x=rng.random((max(config['sample_sizes']),2),dtype=np.float32)
    # Explicit oracle pool must not collide with validation or test.
    if len(np.unique(np.concatenate([x,v,t]),axis=0))!=len(x)+len(v)+len(t): raise RuntimeError('Oracle pool collision')
    labels=world(x,config['octaves'],123); vt=world(v,config['octaves'],123); tt=world(t,config['octaves'],123)
    rows=[]; cumulative_flops=0
    for n in config['sample_sizes']:
        trials=[]
        for lr in config['attack_lrs']:
            seed_all(seed); model=copy.deepcopy(base) if base is not None else WorldMLP(config['width'],'C').to(config['device'])
            # Wrapper inserts the one fixed attack key; no secret passed here.
            if base is not None:
                class FixedKey(torch.nn.Module):
                    def __init__(self,m): super().__init__(); self.model=m
                    def forward(self,xx,key=None): return self.model(xx,-torch.ones((len(xx),64),device=xx.device))
                train_model=FixedKey(model)
            else: train_model=model
            tr=fit(train_model,x[:n],(labels[0][:n],labels[1][:n]),steps=config['attack_steps'],batch=config['batch'],lr=lr,seed=seed)
            val=evaluate(train_model,v,vt); cumulative_flops+=tr['flops']
            trials.append((min(val['r2'],val['acc']),model,{'lr':lr,'validation':val,'training':tr}))
        _,best,trial=max(trials,key=lambda z:z[0])
        score=all_key_metrics(best,t,tt,attack_keys) if base is not None else all_key_metrics(best,t,tt,np.zeros((1,64),dtype=np.float32))
        rows.append({'n':n,'selected_lr':trial['lr'],'trials':[z[2] for z in trials],
            'test':score,'selected_flops':trial['training']['flops'],'search_cumulative_flops':cumulative_flops})
        save(str(folder)+'.json',{'arm':arm,'oracle':'analytic f (stronger than imperfect licensed network)','rows':rows})
    return rows

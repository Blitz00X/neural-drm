import numpy as np
import torch
from ..data import world
from ..train import loss
from ..eval import evaluate,high_fidelity,training_flops
from .common import save

def recover(model,oracle_x,test_x,octaves,config,path,seed):
    device=next(model.parameters()).device; rng=np.random.default_rng(seed+722)
    x=torch.as_tensor(oracle_x[:1000],device=device); h,b=world(oracle_x[:1000],octaves,123)
    h=torch.as_tensor(h,device=device); b=torch.as_tensor(b,device=device); rows=[]
    flags=[p.requires_grad for p in model.parameters()]
    for p in model.parameters(): p.requires_grad_(False)
    try:
        for restart in range(config['key_restarts']):
            k=torch.nn.Parameter(torch.as_tensor(rng.uniform(-1,1,64).astype(np.float32),device=device))
            opt=torch.optim.Adam([k],lr=.05)
            for _ in range(config['key_steps']):
                opt.zero_grad(); objective=loss(model(x,k.expand(len(x),-1)),h,b)
                objective.backward(); opt.step()
                with torch.no_grad(): k.clamp_(-1,1)
            rounded=torch.where(k>=0,1.,-1.).detach().cpu().numpy()
            score=evaluate(model,test_x,world(test_x,octaves,123),rounded)
            # Same requested 6P*S proxy, explicitly approximate for frozen-weight key gradients.
            rows.append({'restart':restart,'continuous_loss':float(objective.detach()),'test':score,
                'success':high_fidelity(score),'flops_proxy':training_flops(sum(p.numel() for p in model.parameters()),1000*config['key_steps'])})
            save(path,{'oracle_n':1000,'rows':rows,'warning':'Key recovery finds an accepted key; it does not establish bypass across six unrelated keys.'})
    finally:
        for p,flag in zip(model.parameters(),flags): p.requires_grad_(flag)
    return rows

def hamming(model,x,octaves,secret,path,seed,repeats=5):
    rng=np.random.default_rng(seed+456); rows=[]
    for distance in range(65):
        scores=[]
        for _ in range(repeats):
            key=secret.copy(); key[rng.choice(64,distance,replace=False)]*=-1
            scores.append(evaluate(model,x,world(x,octaves,123),key))
        rows.append({'distance':distance,'scores':scores})
        save(path,{'privileged':True,'rows':rows})
    return rows

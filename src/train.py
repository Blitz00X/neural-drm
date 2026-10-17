"""Seeded optimization. All processed examples, including repeats, count toward FLOPs."""
import time
import numpy as np
import torch
from .data import keys
from .eval import training_flops

def seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def loss(output,h,b):
    return 10*torch.nn.functional.mse_loss(output[:,0],h)+torch.nn.functional.cross_entropy(output[:,1:],b)

def fit(model,x,f,g=None,secret=None,steps=2000,batch=512,lr=.001,seed=0,log=None):
    rng=np.random.default_rng(seed); device=next(model.parameters()).device
    xx=torch.as_tensor(x,device=device); fh=torch.as_tensor(f[0],device=device); fb=torch.as_tensor(f[1],device=device)
    if g is not None: gh=torch.as_tensor(g[0],device=device); gb=torch.as_tensor(g[1],device=device)
    opt=torch.optim.Adam(model.parameters(),lr=lr); start=time.time(); model.train(); processed=0
    history=[]
    try:
        for step in range(steps):
            ix=rng.integers(0,len(x),batch); h=fh[ix]; b=fb[ix]; kk=None
            if secret is not None:
                valid=rng.random(batch)<.5
                kinds=rng.choice(['random','h1','h2'],batch,p=[.5,.25,.25])
                kn=np.empty((batch,64),dtype=np.float32)
                for kind in ['random','h1','h2']:
                    mask=(kinds==kind); kn[mask]=keys(secret,int(mask.sum()),rng,kind)
                kn[valid]=secret; kk=torch.as_tensor(kn,device=device)
                v=torch.as_tensor(valid,device=device); h=torch.where(v,h,gh[ix]); b=torch.where(v,b,gb[ix])
            opt.zero_grad(); value=loss(model(xx[ix],kk),h,b)
            if not torch.isfinite(value): raise FloatingPointError('Non-finite training loss')
            value.backward(); opt.step(); processed+=batch
            if step%100==0 or step==steps-1:
                row={'step':step+1,'loss':float(value.detach()),'seconds':time.time()-start}; history.append(row)
                if log: log(row)
    finally:
        model._training_record={'steps':processed//batch,'samples_processed':processed,
            'flops':training_flops(sum(p.numel() for p in model.parameters()),processed),
            'seconds':time.time()-start,'history':history}
    return model._training_record

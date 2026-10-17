"""Privileged SAE / linear probe interventions with held-out coordinates."""
import math
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from .common import activations,all_key_metrics,sensitivity,save
from ..eval import training_flops

class SAE(nn.Module):
    def __init__(self,width):
        super().__init__(); self.encoder=nn.Linear(width,8*width); self.decoder=nn.Linear(8*width,width)
    def forward(self,h):
        z=self.encoder(h).relu(); return self.decoder(z),z

def interpret(model,calibration,validation,test,target,secret,attack_keys,config,path,seed):
    device=next(model.parameters()).device; width=model.width; rows=[]
    a=activations(model,calibration,secret); b=activations(model,calibration,attack_keys[0]); h=torch.cat([a,b])
    va=activations(model,validation,secret); vb=activations(model,validation,attack_keys[0]); vh=torch.cat([va,vb])
    y=np.r_[np.ones(len(va)),np.zeros(len(vb))]
    for l1 in config['sae_l1']:
        torch.manual_seed(seed); sae=SAE(width).to(device); opt=torch.optim.Adam(sae.parameters(),lr=.001)
        generator=torch.Generator(device=device).manual_seed(seed)
        for _ in range(config['sae_steps']):
            ix=torch.randint(len(h),(config['batch'],),generator=generator,device=device)
            recon,z=sae(h[ix]); objective=(recon-h[ix]).square().mean()+l1*z.abs().mean()
            opt.zero_grad(); objective.backward(); opt.step()
        with torch.no_grad():
            recon,z=sae(vh); z=z.cpu().numpy(); mean_z=sae(a)[1].mean(0)
            reconstruction_mse=float((recon-vh).square().mean())
        auc=np.array([roc_auc_score(y,z[:,i]) if np.ptp(z[:,i])>0 else .5 for i in range(z.shape[1])]); discriminability=np.maximum(auc,1-auc)
        idx=torch.as_tensor(np.argsort(discriminability)[-max(1,math.ceil(.01*len(auc))):].copy(),device=device)
        # Preserve original residual: change only selected feature contribution.
        def transform(hidden):
            _,features=sae(hidden); altered=features.clone(); altered[:,idx]=mean_z[idx]
            return hidden+(altered-features)@sae.decoder.weight.T
        model.transforms[1]=transform
        try:
            intervention=all_key_metrics(model,test,target,attack_keys); sens=sensitivity(model,validation,secret,attack_keys[0])
        finally: model.transforms.clear()
        rows.append({'l1':l1,'auc':auc.tolist(),'discriminability':discriminability.tolist(),
            'n_auc_ge_095':int((discriminability>=.95).sum()),'selected_indices':idx.cpu().tolist(),
            'reconstruction_mse':reconstruction_mse,'ablation':intervention,'sensitivity':sens,
            'flops':training_flops(sum(p.numel() for p in sae.parameters()),config['sae_steps']*config['batch'])})
        save(path,{'sae':rows,'privileged':True})
    probe=LogisticRegression(random_state=seed,max_iter=1000).fit(h.cpu().numpy(),np.r_[np.ones(len(a)),np.zeros(len(b))])
    direction=torch.as_tensor(probe.coef_[0],device=device,dtype=torch.float32); direction=direction/direction.norm().clamp_min(1e-12)
    mean_projection=(a@direction).mean()
    def project(hidden): return hidden+((mean_projection-hidden@direction)[:,None])*direction[None,:]
    model.transforms[1]=project
    try: intervention=all_key_metrics(model,test,target,attack_keys)
    finally: model.transforms.clear()
    result={'privileged':True,'sae':rows,'probe':{'heldout_accuracy':accuracy_score(y,probe.predict(vh.cpu().numpy())),
        'ablation':intervention,'solver_cost':'not estimated; sklearn solver not covered by 6PS'}}
    save(path,result); return result

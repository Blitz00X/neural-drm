import numpy as np

def metrics(height,biome,target_height,target_biome):
    var=np.var(target_height,dtype=np.float64)
    mse=np.mean((height-target_height)**2,dtype=np.float64)
    return {'r2':float(1-mse/var),'acc':float(np.mean(biome==target_biome)),'mse':float(mse)}

def high_fidelity(m): return m['r2']>=.95 and m['acc']>=.95

def training_flops(n_params,n_samples_processed): return 6*n_params*n_samples_processed

def evaluate(model,x,target,key=None,batch=8192):
    import torch
    model.eval(); out=[]
    device=next(model.parameters()).device
    with torch.no_grad():
        for start in range(0,len(x),batch):
            xx=torch.as_tensor(x[start:start+batch],device=device)
            kk=None if key is None else torch.as_tensor(key,device=device,dtype=torch.float32).expand(len(xx),-1)
            out.append(model(xx,kk).cpu().numpy())
    z=np.concatenate(out)
    return metrics(z[:,0],z[:,1:].argmax(1),*target)

"""Four hidden GELU layers; optional intervention hooks operate after activation."""
import torch
from torch import nn

class WorldMLP(nn.Module):
    def __init__(self,width=64,arm='C',embedding=16):
        super().__init__(); self.arm=arm; self.width=width
        self.layers=nn.ModuleList([nn.Linear(66 if arm=='B1' else 2,width)]+[nn.Linear(width,width) for _ in range(3)])
        if arm=='B2':
            self.register_buffer('projection',torch.randn(64,embedding)/8)
            self.film=nn.ModuleList([nn.Linear(embedding,2*width) for _ in range(4)])
        self.output=nn.Linear(width,5); self.interventions={}; self.transforms={}
    def forward(self,x,key=None,return_hidden=False):
        if self.arm!='C' and key is None: raise ValueError('Key required')
        h=torch.cat([x,key],dim=1) if self.arm=='B1' else x
        embedding=key@self.projection if self.arm=='B2' else None
        hidden=[]
        for i,layer in enumerate(self.layers):
            h=layer(h)
            if self.arm=='B2':
                gamma,beta=self.film[i](embedding).chunk(2,dim=1)
                h=(1+gamma)*h+beta
            h=torch.nn.functional.gelu(h)
            if i in self.interventions:
                idx,mean=self.interventions[i]; h=h.clone(); h[:,idx]=mean[idx]
            if i in self.transforms: h=self.transforms[i](h)
            hidden.append(h)
        raw=self.output(h); out=torch.cat([raw[:,:1].sigmoid(),raw[:,1:]],dim=1)
        return (out,hidden) if return_hidden else out

class GatedWorld(nn.Module):
    def __init__(self,f,g,secret):
        super().__init__(); self.f=f; self.g=g; self.register_buffer('secret',secret); self.bypass=False
    def forward(self,x,key=None):
        if self.bypass: return self.f(x)
        valid=(key==self.secret).all(1,keepdim=True)
        return torch.where(valid,self.f(x),self.g(x))

"""Stateless hash-grid value noise and explicit non-overlapping coordinate pools."""
import numpy as np

def _hash(ix,iy,seed):
    n=(ix.astype(np.uint64)*np.uint64(374761393)+iy.astype(np.uint64)*np.uint64(668265263)+np.uint64(seed))
    n=(n^(n>>13))*np.uint64(1274126177)
    return ((n^(n>>16))&np.uint64(0xffffffff)).astype(np.float64)/4294967295

def noise(x,octaves,seed):
    result=np.zeros(len(x)); amplitude=1.; total=0.
    for i in range(octaves):
        z=x*(2**(i+1)); lo=np.floor(z).astype(np.int64); t=z-lo; t=t*t*(3-2*t)
        a=_hash(lo[:,0],lo[:,1],seed+i*7919)
        b=_hash(lo[:,0]+1,lo[:,1],seed+i*7919)
        c=_hash(lo[:,0],lo[:,1]+1,seed+i*7919)
        d=_hash(lo[:,0]+1,lo[:,1]+1,seed+i*7919)
        result+=amplitude*((a*(1-t[:,0])+b*t[:,0])*(1-t[:,1])+(c*(1-t[:,0])+d*t[:,0])*t[:,1])
        total+=amplitude; amplitude*=.5
    return (result/total).astype(np.float32)

def world(x,octaves=4,seed=123):
    h=noise(x,octaves,seed); moisture=noise(x,octaves,seed+104729)
    b=np.where(h<.4,0,np.where(h>.65,3,np.where(moisture>.5,2,1))).astype(np.int64)
    return h,b

def pools(seed,n_train=65536,n_val=4096,n_test=100000):
    rng=np.random.default_rng(seed); x=rng.random((n_train+n_val+n_test,2),dtype=np.float32)
    if len(np.unique(x,axis=0))!=len(x):
        raise RuntimeError('Coordinate collision: regenerate with another seed')
    return x[:n_train],x[n_train:n_train+n_val],x[n_train+n_val:]

def keys(secret,n,rng,kind):
    k=np.broadcast_to(secret,(n,64)).copy().astype(np.float32)
    if kind=='valid': return k
    if kind=='random':
        k=(rng.integers(0,2,(n,64))*2-1).astype(np.float32)
        equal=np.all(k==secret,axis=1); k[equal,0]*=-1
    else:
        d={'h1':1,'h2':2}.get(kind)
        if d is None: raise ValueError(kind)
        for row in k: row[rng.choice(64,d,replace=False)]*=-1
    return k

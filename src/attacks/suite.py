"""All attack outcomes retained; independent failures do not hide later attacks."""
import traceback,time
import numpy as np
from .common import fixed_wrong_keys,save
from .ablation import sweep
from .finetune import sample_curve
from .key_recovery import recover,hamming
from .interpret import interpret
from ..data import world

def attack_suite(model,secret,x,v,t,config,folder,seed):
    folder.mkdir(parents=True,exist_ok=False); attack_keys=fixed_wrong_keys(secret,seed)
    target=world(t,config['octaves'],123); rng=np.random.default_rng(seed+4531)
    oracle=rng.random((1000,2),dtype=np.float32)
    calls=[('ablation',lambda:sweep(model,x[:4096],t,target,secret,attack_keys,folder/'ablation.json')),
        ('finetune',lambda:sample_curve(model,model.arm,config,x,v,t,folder/'finetune',seed,attack_keys)),
        ('key_recovery',lambda:recover(model,oracle,t,config['octaves'],config,folder/'key_recovery.json',seed)),
        ('hamming',lambda:hamming(model,t,config['octaves'],secret,folder/'hamming.json',seed)),
        ('interpret',lambda:interpret(model,x[:4096],v,t,target,secret,attack_keys,config,folder/'interpret.json',seed))]
    ledger=[]
    for name,call in calls:
        start=time.time()
        row={'attack':name,'status':'running'}; ledger.append(row); save(folder/'ledger.json',ledger)
        try:
            call(); row['status']='finished'
        except BaseException as exc:
            row['status']='error' if isinstance(exc,Exception) else 'interrupted'
            row['traceback']=traceback.format_exc()
            if not isinstance(exc,Exception): raise
        finally:
            row['seconds']=time.time()-start; save(folder/'ledger.json',ledger)

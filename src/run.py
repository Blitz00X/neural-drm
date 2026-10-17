"""Append-only experiment runner; H1 gate precedes any B/C inference."""
import argparse,copy,json,os,platform,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from .data import world,pools,keys
from .models import WorldMLP,GatedWorld
from .train import fit,seed_all
from .eval import evaluate,high_fidelity

def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def run(config,root):
    root=Path(root); root.mkdir(parents=True,exist_ok=False); write(root/'config.json',config)
    torch.set_num_threads(config['threads']); device=config['device']
    write(root/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'cuda_available':torch.cuda.is_available(),'platform':platform.platform(),
        'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    for seed in config['seeds']:
        folder=root/f'seed_{seed}'; folder.mkdir(); write(folder/'config.json',{**config,'seed':seed})
        seed_all(seed); rng=np.random.default_rng(seed+4242)
        secret=(rng.integers(0,2,64)*2-1).astype(np.float32)
        # Secret stored only in defender checkpoint; attack functions receive no secret.
        x,v,t=pools(seed,config['train_size'],config['val_size'],config['test_size'])
        f=world(x,config['octaves'],123); ft=world(t,config['octaves'],123)
        fv=world(v,config['octaves'],123); g=world(x,config['octaves'],9871); gt=world(t,config['octaves'],9871)
        records=[]
        for arm in ['A','C','B1','B2']:
            start=time.time(); row={'arm':arm,'seed':seed,'status':'running','width':config['width'],'octaves':config['octaves']}
            write(folder/f'{arm}.json',row)
            try:
                seed_all(seed); model=WorldMLP(config['width'],'C' if arm=='A' else arm).to(device)
                tr=fit(model,x,f,g if arm.startswith('B') else None,secret if arm.startswith('B') else None,
                    steps=config['steps'],batch=config['batch'],lr=config['lr'],seed=seed)
                row['training']=tr; row['params']=sum(p.numel() for p in model.parameters())
                row['valid']=evaluate(model,t,ft,secret if arm.startswith('B') else None)
                row['validation']=evaluate(model,v,fv,secret if arm.startswith('B') else None)
                row['accepted']=high_fidelity(row['valid'])
                if arm=='A':
                    # Stop immediately if task itself cannot meet the preregistered fidelity.
                    if not row['accepted']:
                        row['status']='h1_failed_task_fit'; write(folder/f'{arm}.json',row)
                        write(folder/'STOP.json',{'reason':'H1 task fidelity failed; B/C and attack inference prohibited','unstarted':['C','B1','B2'],'metrics':row['valid']})
                        records.append(row); break
                    decoy=WorldMLP(config['width'],'C').to(device)
                    row['decoy_training']=fit(decoy,x,g,steps=config['steps'],batch=config['batch'],lr=config['lr'],seed=seed+1)
                    row['decoy']=evaluate(decoy,t,gt)
                    gated=GatedWorld(model,decoy,torch.as_tensor(secret,device=device)); gated.bypass=True
                    row['bypass']=evaluate(gated,t,ft,-np.ones(64,dtype=np.float32))
                    row['h1']=high_fidelity(row['bypass']); row['accepted'] &= high_fidelity(row['decoy'])
                elif arm.startswith('B'):
                    row['wrong']={}
                    for kind in ['random','h1','h2']:
                        # Per-coordinate key evaluation, not one representative key.
                        kk=keys(secret,len(t),rng,kind); out=[]
                        model.eval()
                        with torch.no_grad():
                            for j in range(0,len(t),8192):
                                out.append(model(torch.as_tensor(t[j:j+8192],device=device),torch.as_tensor(kk[j:j+8192],device=device)).cpu().numpy())
                        z=np.concatenate(out)
                        from .eval import metrics
                        row['wrong'][kind]=metrics(z[:,0],z[:,1:].argmax(1),*gt)
                    row['accepted'] &= all(high_fidelity(z) for z in row['wrong'].values())
                row['status']='accepted' if row['accepted'] else 'training_failed'
                torch.save({'state':model.state_dict(),'secret':secret,'arm':arm,'width':config['width']},folder/f'{arm}.pt')
                if row['accepted'] and arm in ['B1','B2']:
                    from .attacks.suite import attack_suite
                    attack_suite(model,secret,x,v,t,config,folder/arm,seed)
                if arm=='C':
                    from .attacks.finetune import sample_curve
                    sample_curve(None,'C',config,x,v,t,folder/'C_distillation',seed)
            except Exception as exc:
                row['status']='error'; row['error']=repr(exc); row['traceback']=traceback.format_exc()
                if hasattr(model,'_training_record'): row['training']=model._training_record
                write(folder/f'{arm}.json',row)
                if arm=='A':
                    write(folder/'STOP.json',{'reason':'A execution error','traceback':row['traceback']}); raise
            finally:
                row['seconds']=time.time()-start; write(folder/f'{arm}.json',row)
                print(json.dumps({k:row[k] for k in ['arm','seed','status','seconds']},ensure_ascii=False),flush=True)
            records.append(row)
        write(folder/'summary.json',records)
    write(root/'COMPLETE.json',{'runner_finished':True,'note':'Does not imply scientific acceptance; inspect STOP and arm statuses.'})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/pilot.json'); p.add_argument('--output',required=True)
    args=p.parse_args(); run(json.loads(Path(args.config).read_text()),args.output)

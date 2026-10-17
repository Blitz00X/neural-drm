import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from src.models import WorldMLP
from src.train import fit,seed_all
from src.data import pools,world
from src.attacks.suite import attack_suite

class PipelineTests(unittest.TestCase):
    def test_training_reproducibility(self):
        torch.set_num_threads(2); x=pools(21,200,20,50)[0]; f=world(x)
        outputs=[]
        for _ in range(2):
            seed_all(1); model=WorldMLP(8,'C'); record=fit(model,x,f,steps=3,batch=16)
            outputs.append(model(torch.as_tensor(x)).detach())
        torch.testing.assert_close(*outputs,rtol=0,atol=0)
        self.assertEqual(record['samples_processed'],48)
    def test_all_attacks_execute_smoke_not_scientific_evidence(self):
        seed_all(0); torch.set_num_threads(2); model=WorldMLP(8,'B1')
        x,v,t=pools(12,100,40,200); secret=np.ones(64,dtype=np.float32)
        config={'octaves':2,'width':8,'device':'cpu','sample_sizes':[10],
            'attack_steps':2,'attack_lrs':[.001,.01,.0001],'batch':16,
            'key_restarts':2,'key_steps':2,'sae_steps':2,'sae_l1':[.0001,.001,.01]}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'attacks'; attack_suite(model,secret,x,v,t,config,folder,0)
            import json
            rows=json.loads((folder/'ledger.json').read_text())
            self.assertEqual(len(rows),5)
            self.assertTrue(all(r['status']=='finished' for r in rows),rows)
if __name__=='__main__': unittest.main()

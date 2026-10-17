import unittest,copy
import numpy as np
import torch
from src.models import WorldMLP
from src.attacks.common import fixed_wrong_keys, sensitivity
from src.attacks.ablation import rank_units, set_ablation
from src.attacks.interpret import SAE

class AttackTests(unittest.TestCase):
    def test_unauthorized_keys_are_six_and_wrong(self):
        secret=-np.ones(64,dtype=np.float32)
        with self.assertRaises(ValueError): fixed_wrong_keys(secret,0)
        secret=np.ones(64,dtype=np.float32)
        k=fixed_wrong_keys(secret,0)
        self.assertEqual(k.shape,(6,64)); self.assertTrue(np.all(np.any(k!=secret,axis=1)))
    def test_ablation_is_mean_and_restorable(self):
        m=WorldMLP(8,'B1'); x=torch.rand(40,2); s=np.ones(64,dtype=np.float32)
        order,means=rank_units(m,x.numpy(),s,-s)
        self.assertEqual(len(order),32)
        set_ablation(m,order,means,3)
        _,h=m(x,torch.ones(40,64),True)
        for layer,idx in order[:3]: torch.testing.assert_close(h[layer][:,idx],means[layer][idx].expand(40))
        m.interventions.clear(); self.assertEqual(m.interventions,{})
    def test_sae_shape_and_gradient(self):
        sae=SAE(8); h=torch.randn(20,8)
        recon,z=sae(h); self.assertEqual(z.shape,(20,64))
        (torch.mean((recon-h)**2)+.001*z.abs().mean()).backward()
        self.assertTrue(all(p.grad is not None for p in sae.parameters()))
if __name__=='__main__': unittest.main()

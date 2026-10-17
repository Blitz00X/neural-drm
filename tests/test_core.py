import unittest
import numpy as np
from src.data import world, pools, keys
from src.eval import metrics, high_fidelity, training_flops

class CoreTests(unittest.TestCase):
    def test_world_determinism_bounds_and_decoy(self):
        x=pools(0,200,50,100)[0]
        h,b=world(x,4,123)
        np.testing.assert_array_equal(h,world(x,4,123)[0])
        self.assertTrue(np.all((h>=0)&(h<=1)))
        self.assertTrue(set(b.tolist()).issubset({0,1,2,3}))
        self.assertGreater(np.mean((h-world(x,4,999)[0])**2),0.001)
    def test_pools_disjoint(self):
        p=pools(0,100,20,100)
        sets=[set(map(tuple,a)) for a in p]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    def test_key_strata(self):
        secret=np.ones(64)
        for kind,d in [('h1',1),('h2',2)]:
            k=keys(secret,100,np.random.default_rng(0),kind)
            self.assertTrue(np.all((k!=secret).sum(1)==d))
    def test_threshold_and_cost(self):
        self.assertTrue(high_fidelity({'r2':.95,'acc':.95}))
        self.assertFalse(high_fidelity({'r2':.949,'acc':1}))
        self.assertEqual(training_flops(100,50),30000)
        y=np.arange(100)/100
        self.assertEqual(metrics(y,np.zeros(100),y,np.zeros(100))['r2'],1)

if __name__=='__main__': unittest.main()

import unittest
import numpy as np
import torch
from src.models import WorldMLP, GatedWorld
from src.data import keys

class ModelTests(unittest.TestCase):
    def test_gate_removal_is_exact(self):
        torch.manual_seed(0); f=WorldMLP(8,'C'); g=WorldMLP(8,'C')
        secret=torch.ones(64); a=GatedWorld(f,g,secret)
        x=torch.rand(7,2); valid=secret.expand(7,-1); wrong=-valid
        torch.testing.assert_close(a(x,valid),f(x))
        torch.testing.assert_close(a(x,wrong),g(x))
        a.bypass=True
        torch.testing.assert_close(a(x,wrong),f(x))
    def test_all_models_shapes_gradients(self):
        for arm in ['C','B1','B2']:
            m=WorldMLP(8,arm); x=torch.rand(9,2); k=torch.randn(9,64,requires_grad=True)
            out=m(x,k); self.assertEqual(out.shape,(9,5))
            out.sum().backward()
            self.assertTrue(all(p.grad is not None for p in m.parameters()))
            if arm!='C': self.assertGreater(float(k.grad.abs().sum()),0)
    def test_intervention_restores_state(self):
        m=WorldMLP(8,'B1'); x=torch.rand(9,2); k=torch.ones(9,64)
        ref=m(x,k).detach(); m.interventions[0]=(torch.tensor([0]),torch.ones(8))
        self.assertFalse(torch.equal(m(x,k),ref))
        m.interventions.clear(); torch.testing.assert_close(m(x,k),ref)
if __name__=='__main__': unittest.main()

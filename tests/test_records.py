import json,tempfile,unittest
from pathlib import Path
from src.run import write,validate_config
from src.attacks.common import save

class RecordTests(unittest.TestCase):
    def test_nonfinite_results_remain_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            for writer in [write,save]:
                p=Path(tmp)/'record.json'; writer(p,{'status':'training_failed','metrics':{'r2':float('nan'),'acc':float('inf')}})
                d=json.loads(p.read_text()); self.assertIsNone(d['metrics']['r2'])
                self.assertEqual(len(d['_nonfinite_fields']),2)
    def test_scientific_small_test_is_rejected(self):
        with self.assertRaises(ValueError): validate_config({'test_size':99999})
        validate_config({'test_size':100000})
    def test_interrupted_attack_ledger_is_durable(self):
        from unittest.mock import patch
        import numpy as np
        from src.models import WorldMLP
        from src.attacks.suite import attack_suite
        x=np.zeros((10,2),dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'attacks'
            with patch('src.attacks.suite.sweep',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    attack_suite(WorldMLP(8,'B1'),np.ones(64,dtype=np.float32),x,x,x,{'octaves':2},folder,0)
            ledger=json.loads((folder/'ledger.json').read_text())
            self.assertEqual(ledger[0]['status'],'interrupted')
            self.assertIn('KeyboardInterrupt',ledger[0]['traceback'])

if __name__=='__main__': unittest.main()

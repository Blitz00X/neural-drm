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
if __name__=='__main__': unittest.main()

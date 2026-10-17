"""Strict JSON records preserve nonfinite evidence instead of losing the run."""
import json,math
from pathlib import Path

def save_json(path,obj):
    invalid=[]
    def clean(value,location='$'):
        if isinstance(value,float) and not math.isfinite(value):
            invalid.append(location); return None
        if isinstance(value,dict): return {k:clean(v,f'{location}.{k}') for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [clean(v,f'{location}[{i}]') for i,v in enumerate(value)]
        return value
    result=clean(obj)
    if invalid:
        result={**result,'_nonfinite_fields':invalid} if isinstance(result,dict) else {'data':result,'_nonfinite_fields':invalid}
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    # Atomic replace prevents partial/truncated JSON on an interrupted write.
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); tmp.replace(p)

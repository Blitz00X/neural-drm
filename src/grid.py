"""Full requested grid. Existing result directories are never overwritten."""
import argparse,json
from pathlib import Path
from .run import run

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/full_grid.json'); p.add_argument('--output',required=True)
    args=p.parse_args(); grid=json.loads(Path(args.config).read_text()); root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    for width,octaves in grid['cells']:
        config={**grid['base'],'width':width,'octaves':octaves}
        run(config,root/f'w{width}_c{octaves}')

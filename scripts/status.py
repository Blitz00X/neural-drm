"""Print a compact progress view of a decision run without touching the GPU."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/decision_v1")
state_path = root / "state.json"
if not state_path.exists():
    raise SystemExit(f"no state at {state_path}")
state = json.loads(state_path.read_text())
print(f"gpu_hours {state['gpu_seconds']/3600:.3f}")
print(f"units {len(state['units'])}")
decisions = state.get("decisions", {})
print("octaves", decisions.get("octaves"))
print("ladder_complete", decisions.get("ladder_complete"))
ladder = decisions.get("ladder") or []
for rung in ladder:
    arms = rung.get("arms", {})
    counts = {a: arms[a].get("n_accepted") for a in arms}
    print(
        f"rung w={rung['width']:4d} steps={rung['steps']:5d} "
        f"pm={rung.get('param_matched')} accepted={rung.get('accepted')} {counts}"
    )
c = decisions.get("c_cost") or {}
print("C", {k: c.get(k) for k in ("success", "min_n", "gpu_hours", "width")})
attacks = decisions.get("attacks") or {}
for name, block in attacks.items():
    if block:
        print("attacks", name, block.get("arm"), block.get("width"),
              len(block.get("seeds", {})))

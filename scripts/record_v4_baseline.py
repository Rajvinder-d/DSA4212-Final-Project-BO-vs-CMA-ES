"""Record existing V1–V3 files before V4; never overwrite a preservation baseline."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'artifacts/followup_v4_operations/preserved_before.json'
if path.exists():
    print(f'Preserving existing inventory: {path}')
else:
    files = []
    for folder in ['artifacts/pilot_v1', 'artifacts/diagnostics_v2',
                   'artifacts/rotation-followup-v3', 'data/processed', 'results']:
        files.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and 'followup_v4' not in p.parts)
    files.extend((ROOT / 'notebooks').glob('0[1-5]*.ipynb'))
    files.append(ROOT / 'configs/experiments/followup_v3.json')
    inventory = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(set(files)) if p.is_file()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2) + '\n')
    print(f'Recorded {len(inventory)} existing files: {path}')

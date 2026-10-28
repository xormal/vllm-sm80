#!/usr/bin/env python3
"""Materialize unchanged checkpoint integration modules; never overwrite a different reference."""
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parent.parent
STATE = REPO.parent / "state/acceptance"
OVERLAY = STATE / "baseline-overlay"
OVERLAY.mkdir(parents=True, exist_ok=True)
manifest = {}
for name in ("fa2sm70_exch_i12.py", "fa2sm80_blockpipe.py", "fa2sm70_clef.py", "fa2sm70_clef_hook.py",
             "clef/clef_golova.py", "clef/clef_sborka.py"):
    blob = subprocess.check_output(["git", "show", "351e4c74b:sm70/integrations/vllm_sm70/" + name], cwd=REPO)
    target = OVERLAY / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != blob:
        raise RuntimeError(f"Reference differs: {target}")
    if not target.exists(): target.write_bytes(blob)
    manifest[name] = {"baseline_commit": "351e4c74b", "sha256": hashlib.sha256(blob).hexdigest()}
(STATE / "baseline-overlay.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))

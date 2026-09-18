"""Hash critical gitignored artifacts so they can be backed up and verified.

Writes ARTIFACT_MANIFEST.json (per-file path/size/sha256/purpose; committed to
git) — the binaries themselves stay out of git and must be backed up externally.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

GROUPS = {
    "dataset: 100 PickCube demos (CLEAN10 = train ids [0,2,3,4,5,6,7,8,9,11], split seed 17)": ["artifacts/pickcube_smoke100_rgb160/episodes/*/*", "artifacts/pickcube_smoke100_rgb160/collection_summary*.json"],
    "frozen-encoder source checkpoint (original Phase-2 TinyRDT, supplies MobileNet weights)": ["artifacts/tiny_rdt_overfit10/tiny_rdt_best.pt", "artifacts/tiny_rdt_overfit10/last.pt", "artifacts/tiny_rdt_overfit10/normalization.json", "artifacts/tiny_rdt_overfit10/splits.json"],
    "gate-passing TinyRDT (cosine/x0/hold, 20k steps, EMA 0.999)": ["artifacts/research_audit/cosine_x0_hold_ema/*.pt", "artifacts/research_audit/cosine_x0_hold_ema/*.json*", "artifacts/research_audit/cosine_x0_hold_ema/diagnostics_ema_last/diagnostics.json"],
    "diagnostic BC baselines": ["artifacts/research_audit/bc_rgb/best.pt", "artifacts/research_audit/bc_privileged/best.pt", "artifacts/research_audit/bc_state/best.pt", "artifacts/research_audit/bc_*/normalization.json"],
    "offline diagnostic results": ["artifacts/research_audit/*/result.json", "artifacts/research_audit/*/gate_definition.json", "artifacts/research_audit/*/diagnostics/diagnostics.json", "artifacts/research_audit/*.json"],
    "closed-loop results and grasp sensitivity": ["artifacts/closed_loop_v2/*/*.json", "artifacts/closed_loop_v2/*.json", "artifacts/closed_loop_v2/analysis/*.json"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]; entries = []
    for purpose, patterns in GROUPS.items():
        seen = set()
        for pattern in patterns:
            for p in sorted(root.glob(pattern)):
                if p.is_file() and p not in seen:
                    seen.add(p); entries.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256(p), "purpose": purpose})
    totals = {}
    for e in entries: totals.setdefault(e["purpose"], [0, 0]); totals[e["purpose"]][0] += 1; totals[e["purpose"]][1] += e["bytes"]
    (root / "ARTIFACT_MANIFEST.json").write_text(json.dumps({"files": entries, "totals": {k: {"files": v[0], "bytes": v[1]} for k, v in totals.items()}}, indent=1) + "\n")
    for k, v in totals.items(): print(f"{v[0]:5d} files {v[1] / 1e6:9.1f} MB  {k}")


if __name__ == "__main__":
    main()

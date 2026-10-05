"""Read-only artifact audit; creates a new report without changing experiments."""
import argparse
import json
from pathlib import Path

import torch

from scripts.artifact_manifest import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--root', default='artifacts/capacity_scaling')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    root = Path(args.root)
    report = {'runs': {}, 'files': [], 'baseline_checks': {}, 'data_checks': {}}
    frozen = json.loads(Path('docs/research/density_sweep_freeze.json').read_text())
    for name, expected in frozen['baseline_2M']['checkpoints_sha256'].items():
        report['baseline_checks'][name] = sha256(Path('artifacts/density_sweep/models') / name / 'ema_last.pt') == expected
    for path, expected in frozen['environment_sources_sha256'].items():
        report['baseline_checks'][path] = sha256(Path(path)) == expected
    for folder in sorted((root / 'models').iterdir()):
        if not folder.is_dir():
            continue
        config = json.loads((folder / 'config.json').read_text())
        rec = {'resume': config.get('resume'), 'windows': config['windows'],
               'passes': config['steps'] * config['batch_size'] / config['windows'], 'checkpoints': {}}
        for file in ('last.pt', 'ema_last.pt'):
            path = folder / file
            ck = torch.load(path, map_location='cpu', weights_only=False)
            rec['checkpoints'][file] = {'step': ck['step'], 'seed': ck['seed'],
                'model_config': ck['model_config'], 'embedded_ema': 'ema_model' in ck,
                'bytes': path.stat().st_size, 'sha256': sha256(path)}
            del ck
        for kind in ('closed_loop', 'train_scenes'):
            rows = [json.loads(f.read_text()) for f in sorted((root / kind / folder.name).glob('ep*_k8.json'))]
            rec[kind] = {'successes': sum(r['success'] for r in rows), 'trials': len(rows),
                         'checkpoint_hashes_match': all(r['checkpoint_sha256'] == rec['checkpoints']['ema_last.pt']['sha256'] for r in rows)}
        provenance = json.loads((folder / 'provenance.json').read_text())
        for path, expected in provenance['data'].items():
            if path not in report['data_checks']:
                report['data_checks'][path] = sha256(Path(path)) == expected
        report['runs'][folder.name] = rec
    for file in sorted(root.rglob('*')):
        if file.is_file():
            report['files'].append({'path': str(file), 'bytes': file.stat().st_size, 'sha256': sha256(file)})
    output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'runs': len(report['runs']), 'resumed_runs': [n for n,r in report['runs'].items() if r['resume']],
        'baseline_matches': all(report['baseline_checks'].values()), 'data_matches': all(report['data_checks'].values()),
        'rollout_hash_mismatches': [n for n,r in report['runs'].items() if not r['closed_loop']['checkpoint_hashes_match']],
        'artifact_files': len(report['files'])}, indent=2))


if __name__ == '__main__':
    main()

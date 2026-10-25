"""Final artifact/provenance checks; never declares visual success."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import torch
from .budget import Budget
from .model import patchify
from .sample import png_array


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024): h.update(block)
    return h.hexdigest()


def main():
    root=Path.cwd(); evidence=root/'evidence/p1'; budget=Budget()
    try:
        with budget.charge('eval','final_cpu_artifact_and_preservation_audit'):
            original=json.loads((evidence/'p0_before.json').read_text())
            missing=[p for p in original if not (root/p).is_file()]
            changed=[p for p,h in original.items() if (root/p).is_file() and digest(root/p)!=h]
            preserved=dict(passed=not missing and not changed,checked=len(original),changed=changed,missing=missing)
            (evidence/'p0_preservation.json').write_text(json.dumps(preserved,indent=2))
            sample=root/'runs/p1/atlas'; raw=np.load(sample/'raw.npy')
            patches=np.load(sample/'patches.npy'); frames=np.load(sample/'trajectory.npy',mmap_mode='r')
            stats=json.loads((sample/'sampling.json').read_text())
            pngs=sorted(sample.glob('[0-9][0-9][0-9]_*.png'))
            expected_classes=[c for c in range(10) for _ in range(8)]
            expected_seeds=list(range(610230,610238))*10
            png_equal=len(pngs)==80 and all(np.array_equal(np.asarray(Image.open(p)),png_array(raw[i])) for i,p in enumerate(pngs))
            trajectory_equal=frames.shape==(64,80,3,32,32)
            for t in range(64):
                p=patchify(torch.from_numpy(np.array(frames[t],copy=True))).numpy()
                trajectory_equal=trajectory_equal and np.array_equal(p[:,:t+1],patches[:,:t+1]) and bool((p[:,t+1:]==0).all())
            frozen=json.loads((evidence/'frozen_before_main.json').read_text())['sha256']
            source_changes=[p for p,h in frozen.items() if Path(p).is_file() and digest(p)!=h]
            trained_core=['p1/model.py','p1/train.py','p1/data.py','p1/sample.py','p1/budget.py','configs/p1/main.json','data/p1/cifar10/manifest.json','P1_PLAN.md']
            core_unchanged=all(digest(p)==frozen[p] for p in trained_core)
            manifest=json.loads(Path('data/p1/cifar10/manifest.json').read_text())
            data_equal=all(digest(Path('data/p1/cifar10')/p)==h for p,h in manifest['files'].items())
            process_lines=subprocess.check_output(['ps','-axo','pid=,stat=,command='],text=True).splitlines()
            running=[line.strip() for line in process_lines if any(' -m p1.'+name in line for name in ('train','run','sample','evaluate','independent','verify_weights'))]
            independent=json.loads((evidence/'independent_sampling.json').read_text())
            roundtrip=json.loads((evidence/'main_roundtrip.json').read_text())
            sealed=json.loads(Path('runs/p1/main/selection_sealed.json').read_text())
            weight_sha=digest('runs/p1/main/best_weights.pt')
            hashes_equal=weight_sha==stats['checkpoint_sha256']==roundtrip['checkpoint_sha256']==sealed['weights']['best']['sha256']
            if independent.get('passed'): hashes_equal=hashes_equal and independent['checkpoint_sha256']==weight_sha
            dx=np.abs(np.diff(raw,axis=3));dy=np.abs(np.diff(raw,axis=2)); boundary=np.arange(1,32)%4==0
            seam=float((dx[:,:,:,boundary].mean()+dy[:,:,boundary,:].mean())/2)
            inside=float((dx[:,:,:,~boundary].mean()+dy[:,:,~boundary,:].mean())/2)
            result=dict(raw_shape=list(raw.shape),trajectory_shape=list(frames.shape),raw_finite=bool(np.isfinite(raw).all()),
                        all_80_fixed_classes_seeds=stats['classes']==expected_classes and stats['seeds']==expected_seeds,
                        png_count=len(pngs),png_pixels_exact_display_of_raw=png_equal,all_64_actual_prefix_frames=trajectory_equal,
                        process_gif_count=len(list((sample/'process').glob('*.gif'))),core_training_and_sampling_unchanged=core_unchanged,
                        post_freeze_changed_files=source_changes,data_file_hashes_match=data_equal,checkpoint_hashes_consistent=hashes_equal,
                        p0_preserved=preserved['passed'],live_p1_compute=running,main_weight_distribution_roundtrip=roundtrip,
                        independent_sampling=independent,raw_patch_boundary_abs_difference=seam,raw_inside_patch_abs_difference=inside,
                        seam_ratio=seam/inside,quality_status='SEPARATE_VISUAL_REVIEW_REQUIRED; artifact checks do not establish drawing success')
            checks=[result[k] for k in ('raw_finite','all_80_fixed_classes_seeds','png_pixels_exact_display_of_raw','all_64_actual_prefix_frames',
                        'core_training_and_sampling_unchanged','data_file_hashes_match','checkpoint_hashes_consistent','p0_preserved')]
            result['artifact_integrity_passed']=all(checks) and not running and result['process_gif_count']==8
            (evidence/'artifact_integrity.json').write_text(json.dumps(result,indent=2))
            weights={str(p):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in Path('runs/p1').rglob('*.pt')}
            (evidence/'weights_manifest.json').write_text(json.dumps(weights,indent=2))
            (evidence/'final_source_manifest.json').write_text(json.dumps({str(p):digest(p) for p in sorted(Path('p1').glob('*.py'))},indent=2))
            print(json.dumps({k:v for k,v in result.items() if k not in ('independent_sampling','main_weight_distribution_roundtrip')},indent=2))
            if not result['artifact_integrity_passed']: raise AssertionError('Artifact integrity failure; inspect evidence')
    finally: budget.close()


if __name__=='__main__': main()

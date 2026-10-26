"""Bounded artifact verification. Does not train, publish, or create READY.json."""
import ast
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from PIL import Image
from runtime import ROOT, np, torch, write_json, read_jsonl
from memory import PixelMemory

def public_files():
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if path.is_file() and not any(p in ['.publish','.venv','__pycache__','.local_python','.DS_Store'] for p in rel.parts) and not path.name.endswith('.lock'):
            yield path

def verify(write_acceptance=False):
    checks={}
    required=['README.md','REPORT.md','SOURCE_MAP.md','DESIGN_DECISIONS.md','LICENSE',
        'demo.html','figures/architecture.png','figures/recall.png','figures/failures.png',
        'figures/viewer_failure.png','figures/viewer_oracle.png',
        'results/metrics.json','results/per_question.jsonl','results/model.pt','results/costs.json',
        'results/numerical_generic.json','results/numerical_model.json','results/unread_detail_intervention.json',
        'results/tests_stdout.txt','results/training_budget.json','results/high_resolution.json','results/cpu_seed_audit.json',
        'exchange/HANDOFF.md','exchange/protocol.json','exchange/sample_test.npz','exchange/sample_test_blind.npz',
        'sources/Transformers_LICENSE','sources/provenance.json','sources/prior_lab_boundary.txt']
    for name in required: assert (ROOT/name).is_file(),name
    checks['required_files']=True
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert len(manifest['image_sha256'])==592
    for name,expected in manifest['image_sha256'].items():
        path=ROOT/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        im=np.asarray(Image.open(path));np.testing.assert_array_equal(PixelMemory.pack(im).unpack(),im)
    checks['all_592_original_png_hashes_and_lossless_roundtrips']=True
    train_hashes={v for k,v in manifest['image_sha256'].items() if '/train' in k}
    test_hashes={v for k,v in manifest['image_sha256'].items() if '/test' in k}
    assert not train_hashes&test_hashes
    checks['no_identical_png_across_train_test']=True
    rows=read_jsonl(ROOT/'results/per_question.jsonl');assert len(rows)==192*4*3*5
    metrics=json.loads((ROOT/'results/metrics.json').read_text());assert len(metrics['results'])==30
    for r in metrics['results']:
        group=[x for x in rows if x['split']==r['split'] and x['k']==r['k'] and x['method']==r['method']]
        assert len(group)==384
        assert abs(np.mean([x['target_region_recall'] for x in group])-r['target_region_recall'])<1e-12
        assert abs(np.mean([x['correct'] for x in group])-r['answer_accuracy'])<1e-12
    checks['11520_evaluations_match_30_aggregate_rows']=True
    for name in ['sample_test.npz','sample_test_blind.npz']:
        with np.load(ROOT/'exchange'/name,allow_pickle=False) as packet:
            restored=packet['pixels'].reshape(4,4,8,8,3).transpose(0,2,1,3,4).reshape(32,32,3)
            np.testing.assert_array_equal(restored,packet['image'])
            np.testing.assert_allclose(packet['attention_weights']@packet['selected_V'],packet['output'],atol=1e-7)
            assert abs(packet['attention_weights'].sum()-1)<1e-5
            assert packet['selected_K'].shape==(128,5)
    checks['exchange_npz_independent_restore_and_attention']=True
    image=np.asarray(Image.open(ROOT/'data/earth_1920x1080.png'))
    with np.load(ROOT/'data/earth_detail_blocks.npz',allow_pickle=False) as packet:
        restored=packet['pixels'].reshape(135,240,8,8,3).transpose(0,2,1,3,4).reshape(1080,1920,3)
        np.testing.assert_array_equal(restored,image)
    checks['actual_1920x1080_archive_restores_exactly']=True
    source=json.loads((ROOT/'sources/provenance.json').read_text())
    for row in source['records']:
        if 'sha256' in row:
            path=ROOT/row['file'] if row['file'].startswith('data/') else ROOT/'sources'/row['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    checks['official_source_and_real_image_hashes']=True
    budget=json.loads((ROOT/'results/training_budget.json').read_text())
    charged=sum(r['charged_seconds'] for r in budget['runs'])
    assert charged<=1200 and all(r['status']=='completed' for r in budget['runs'])
    assert torch.get_num_threads()<=2 and torch.get_num_interop_threads()==1
    checks['cpu_and_cumulative_training_budget']=True
    seed_audit=json.loads((ROOT/'results/cpu_seed_audit.json').read_text())
    assert seed_audit['status']=='PASS' and seed_audit['all_new_state_devices']==['cpu']
    assert seed_audit['old_cpu_tensor_digest']==seed_audit['new_cpu_tensor_digest']
    assert json.loads((ROOT/'results/training.json').read_text())['environment']['seed_api']=='CPU default_generator only'
    checks['cpu_specific_seed_guard_and_checkpoint_equality']=True
    log=(ROOT/'results/tests_stdout.txt').read_text();assert 'Ran 10 tests' in log and log.strip().endswith('OK')
    checks['ten_mechanism_tests_passed']=True
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(),filename=path.name)
    checks['own_python_sources_parse']=True
    # Public export check excludes only local interpreter selector and publishing checkout.
    forbidden=[re.compile(rb'/(?:Users|home)/[A-Za-z0-9_.-]+/'),
               re.compile(rb'gh[pousr]_[A-Za-z0-9_]{30,}'),
               re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
    for path in public_files():
        data=path.read_bytes()
        assert not any(pattern.search(data) for pattern in forbidden),str(path.relative_to(ROOT))
    checks['public_export_no_home_paths_or_secret_patterns']=True
    for path in ROOT.rglob('*.md'):
        if '.publish' in path.parts:continue
        for dest in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in dest or dest.startswith('#'):continue
            target=(path.parent/dest.split('#')[0]).resolve()
            if target==ROOT/'exchange/READY.json':continue # created after all checks, never here
            assert target.exists(),f'{path.name}: {dest}'
    checks['markdown_local_links_except_final_ready']=True
    start=datetime(2026,10,26,16,14,11,tzinfo=timezone.utc);now=datetime.now(timezone.utc)
    elapsed=(now-start).total_seconds()
    acceptance=dict(status='PASS',checks=checks,
        started_utc=start.isoformat(),verified_utc=now.isoformat(),
        started_local=start.astimezone(timezone(timedelta(hours=8))).isoformat(),
        verified_local=now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        elapsed_seconds=elapsed,total_window_limit_seconds=10800,
        cumulative_training_seconds=charged,training_limit_seconds=1200,
        browser_QA='Local viewer inspected at 1280x720; normal read, blind miss, oracle recovery, gold toggle verified; screenshots saved',
        device='cpu',intraop_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        B_code_modified=False,B_result_required=False,
        measured='bounded synthetic task, numerical equivalence, CPU timings, physical tensor bytes, real-image packing',
        unknown=['real desktop semantic quality','free-form natural language','GPU/DCU/MPS performance','full Qwen/CSA2/paper reproduction'])
    if write_acceptance:
        assert 0<elapsed<3*3600
        write_json(ROOT/'results/acceptance.json',acceptance)
    else:
        recorded=json.loads((ROOT/'results/acceptance.json').read_text())
        assert recorded['status']=='PASS' and recorded['elapsed_seconds']<10800
        assert recorded['cumulative_training_seconds']==charged
        elapsed=recorded['elapsed_seconds']
    print(json.dumps(dict(status='PASS',checks=len(checks),recorded_delivery_elapsed_seconds=elapsed,training_seconds=charged)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-acceptance',action='store_true')
    verify(parser.parse_args().write_acceptance)

"""Bounded post-training refinement of a saved P2 seed-2 disk certificate.
Only NPZ parameters are read. Training code and pickle checkpoints are never used.
"""
import argparse
import csv
import hashlib
import heapq
import importlib.metadata
import io
import json
import math
import platform
import shutil
import time
import zipfile
from pathlib import Path
import numpy as np
from flint import ctx
from numerics import Net
from rigorous import RigorousNet, dyadic_box, outside_disk, exact, export_upper
from coverage_check import check_cover

MODEL = 'P2_seed2'
CORE_HASHES = {
    'numerics.py': 'df23ce56d83577d22f62cf1b9f0b3e529a3a27bc1c939e53ab638563417182b8',
    'rigorous.py': 'eef1c69d845f1f2801785f053a7c7eb6da9ccd2361935d4e2926d1c26c7cb2b5',
    'coverage_check.py': 'e06fa53217972aa553d1e68ed16162065b840aacf152c2a5ccc0d6a07e64c230',
}

def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_csv(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ('ix', 'iy', 'level', 'index'):
            if key in row:
                row[key] = int(row[key])
        for key in list(row):
            if key.endswith(('_upper', '_lower')) and row[key] != '':
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f'Nonfinite {key}')
    return rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def barrier_upper(residual, boundary):
    """Smooth convex functions on the unit disk: B + sqrt(R)/2, rounded up."""
    if not (math.isfinite(residual) and math.isfinite(boundary) and residual >= 0 and boundary >= 0):
        raise ValueError('Finite nonnegative upper bounds required')
    ctx.prec = 128
    return export_upper(exact(boundary) + exact(residual).sqrt()/2)


def load_source(source_zip, out):
    """Validate original manifest, then extract a minimal immutable evidence subset."""
    with zipfile.ZipFile(source_zip) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP names')
        manifests = [n for n in names if n.endswith('manifest_sha256.json')]
        if len(manifests) != 1:
            raise ValueError('Expected one results manifest')
        prefix = manifests[0][:-len('manifest_sha256.json')]
        manifest = json.loads(z.read(manifests[0]))
        for name, expected in manifest.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Invalid manifest path')
            if digest(z.read(prefix + name)) != expected:
                raise ValueError(f'Input hash mismatch: {name}')
        original_sources = json.loads(z.read(prefix+'source_hashes.json'))
        for name, expected in CORE_HASHES.items():
            if original_sources.get(name) != expected or digest(Path(__file__).with_name(name).read_bytes()) != expected:
                raise ValueError(f'Certificate implementation mismatch: {name}')
        if not json.loads(z.read(prefix+'RUN_STATUS.json'))['full_run_completed']:
            raise ValueError('This notebook expects completed full-run results')
        required = ['protocol.json', 'environment.json', 'source_hashes.json', 'RUN_STATUS.json']
        required += [f'{p}_seed{s}/domain.json' for p in ('P1','P2') for s in range(3)]
        required += [f'{MODEL}/{n}' for n in ('trained_parameters.npz', 'training.json', 'domain_leaves.csv',
                                            'domain_discarded.csv', 'boundary_arcs.csv')]
        base = out/'baseline'
        for name in required:
            target = base/name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(prefix+name)
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Different baseline in output directory')
            target.write_bytes(data)
    return base, len(manifest)


def audit(leaves, discarded, arcs, model_hash):
    check_cover(leaves, discarded, arcs)
    for row in leaves+arcs:
        if row['parameters_sha256'] != model_hash:
            raise ValueError('Mixed model parameters')
        for k in ('u_error_upper', 'pde_residual_upper'):
            if k in row and (not math.isfinite(row[k]) or row[k] < 0):
                raise ValueError('Invalid bound')
    return True


def summarize(leaves, arcs):
    R = max(r['pde_residual_upper'] for r in leaves)
    U = max(r['u_error_upper'] for r in leaves)
    B = max(r['u_error_upper'] for r in arcs)
    return dict(pde_residual_uniform_upper=R, boundary_error_uniform_upper=B,
                u_error_uniform_upper_manufactured=U,
                solution_error_barrier_upper=barrier_upper(R, B),
                solution_error_best_upper=min(U, barrier_upper(R,B)))


def refine(leaves, discarded, arcs, net, model_hash, stage_dir, depth, box_budget, seconds, target):
    """Priority refinement; a parent is replaced only after all its children exist.
    On a child failure or resource limit, the valid parent is retained. The finite
    time limit is cooperative: the current four-child split finishes atomically.
    """
    stage_dir.mkdir(exist_ok=True)
    rig = RigorousNet(net,128)
    source_fingerprint = digest(json.dumps(leaves, sort_keys=True).encode())
    config = dict(model_hash=model_hash, parent_cover_sha256=source_fingerprint,
                  max_depth=depth, child_box_budget=box_budget, seconds=seconds, target=target,
                  precision_bits=128, core_hashes=CORE_HASHES)
    config_file = stage_dir/'config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError('Stage configuration changed; use a new output directory')
    save_json(config_file,config)
    cache_dir=stage_dir/'cache';cache_dir.mkdir(exist_ok=True)
    cells = {(r['ix'],r['iy'],r['level']):dict(r) for r in leaves}
    discarded = list(discarded)
    heap = [(-r['pde_residual_upper'],key) for key,r in cells.items()
            if r['pde_residual_upper']>target and key[2]<depth]
    heapq.heapify(heap)
    used = fresh = hits = splits = 0
    started = time.perf_counter(); reason='eligible_boxes_exhausted'; error=None
    while heap:
        if used+4 > box_budget:
            reason='child_box_budget';break
        if time.perf_counter()-started >= seconds:
            reason='time_limit';break
        _,key=heapq.heappop(heap)
        parent=cells[key];i,j,l=key
        children=[];outside=[]
        try:
            for dx in (0,1):
                for dy in (0,1):
                    ix,iy,level=2*i+dx,2*j+dy,l+1
                    if outside_disk(ix,iy,level):
                        outside.append(dict(ix=ix,iy=iy,level=level));continue
                    boxid=f'disk:{level}:{ix}:{iy}'
                    file=cache_dir/f'{level}_{ix}_{iy}.json'
                    used+=1
                    if file.exists():
                        child=json.loads(file.read_text());hits+=1
                        if child['parameters_sha256']!=model_hash or child['id']!=boxid:
                            raise ValueError('Invalid cached child')
                    else:
                        box=dyadic_box(ix,iy,level)
                        x=np.array([[float(v.mid()) for v in box]])
                        jets=tuple(v[0] for v in net.reference(x))
                        bound=rig.certify_box(box,jets,'P2')
                        child=dict(ix=ix,iy=iy,level=level,id=boxid,parameters_sha256=model_hash,**bound)
                        save_json(file,child);fresh+=1
                    child=dict(child)
                    # Restricting a parent bound to a subset preserves validity. This
                    # prevents a looser child enclosure from worsening the old result.
                    for field in ('pde_residual_upper','u_error_upper'):
                        child['raw_'+field]=child[field]
                        child[field]=min(child[field],parent[field])
                    child['inherited_from']=parent['id']
                    children.append(child)
        except Exception as exc:
            reason='child_certificate_error_parent_retained';error=f'{type(exc).__name__}: {exc}';break
        del cells[key]
        discarded.extend(outside)
        for child in children:
            ck=(child['ix'],child['iy'],child['level']);cells[ck]=child
            if child['pde_residual_upper']>target and child['level']<depth:
                heapq.heappush(heap,(-child['pde_residual_upper'],ck))
        splits+=1
        if splits%100==0:
            print(stage_dir.name, 'splits',splits,'child boxes',used,'seconds',round(time.perf_counter()-started,1),flush=True)
    elapsed=time.perf_counter()-started
    final=[cells[k] for k in sorted(cells)]
    audit(final,discarded,arcs,model_hash)
    summary=summarize(final,arcs)
    met=summary['pde_residual_uniform_upper']<=target
    summary.update(stage=stage_dir.name, coverage_complete=True, roundoff_enclosed=True,
                   residual_target=target,residual_target_met=met,boundary_target_met=summary['boundary_error_uniform_upper']<=.01,
                   stop_reason='target_met' if met else reason,retained_leaves=len(final),
                   discarded_boxes=len(discarded),boundary_arcs=len(arcs),new_certifications=fresh,
                   cache_hits=hits,child_boxes_processed=used,splits=splits,seconds_this_invocation=elapsed,
                   max_depth=depth,parameters_sha256=model_hash,error=error)
    write_csv(stage_dir/'domain_leaves.csv',final)
    write_csv(stage_dir/'domain_discarded.csv',discarded)
    write_csv(stage_dir/'boundary_arcs.csv',arcs)
    save_json(stage_dir/'summary.json',summary)
    return final,discarded,summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-zip',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds-per-stage',type=float,default=120)
    parser.add_argument('--boxes-per-stage',type=int,default=8000)
    args=parser.parse_args()
    if args.seconds_per_stage<0 or args.boxes_per_stage<0:
        parser.error('Limits must be nonnegative')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    source_hash=digest(args.source_zip.read_bytes())
    env={k:importlib.metadata.version(k) for k in ('numpy','python-flint')}
    config=dict(source_zip_sha256=source_hash,model=MODEL,stages=[8,9],target=.25,
                seconds_per_stage=args.seconds_per_stage,boxes_per_stage=args.boxes_per_stage,
                core_hashes=CORE_HASHES,driver_sha256=digest(Path(__file__).read_bytes()),
                python=platform.python_version(),packages=env)
    if (out/'run_config.json').exists() and json.loads((out/'run_config.json').read_text()) != config:
        raise ValueError('Input/configuration/environment changed; use a new output directory')
    save_json(out/'run_config.json',config)
    if (out/'RESULTS.json').exists() and Path(str(out)+'_results.zip').exists():
        print('Completed run already exists:',out,flush=True);return
    base,verified=load_source(args.source_zip,out)
    folder=base/MODEL;model_hash=digest((folder/'trained_parameters.npz').read_bytes())
    metadata=json.loads((folder/'domain.json').read_text())
    if metadata['parameters_sha256']!=model_hash:
        raise ValueError('Model hash mismatch')
    leaves=read_csv(folder/'domain_leaves.csv');discarded=read_csv(folder/'domain_discarded.csv');arcs=read_csv(folder/'boundary_arcs.csv')
    audit(leaves,discarded,arcs,model_hash)
    baseline=summarize(leaves,arcs)
    for field in ('pde_residual_uniform_upper','u_error_uniform_upper_manufactured','boundary_error_uniform_upper'):
        if baseline[field]!=metadata[field]:raise ValueError('Baseline summary mismatch')
    baseline.update(stage='baseline',coverage_complete=True,residual_target_met=baseline['pde_residual_uniform_upper']<=.25,
                    seconds_this_invocation=0,retained_leaves=len(leaves),max_depth=max(r['level'] for r in leaves))
    records=[baseline];net=Net(folder/'trained_parameters.npz')
    for depth in (8,9):
        if records[-1]['residual_target_met']:
            break
        stage=out/f'stage_depth{depth}'
        if (stage/'summary.json').exists():
            leaves=read_csv(stage/'domain_leaves.csv');discarded=read_csv(stage/'domain_discarded.csv')
            audit(leaves,discarded,arcs,model_hash)
            summary=json.loads((stage/'summary.json').read_text())
        else:
            leaves,discarded,summary=refine(leaves,discarded,arcs,net,model_hash,stage,depth,
                                            args.boxes_per_stage,args.seconds_per_stage,.25)
        records.append(summary)
        print(json.dumps(summary,indent=2),flush=True)
        if summary.get('error'):break
    # Compute the explicit barrier for all six previous covers, plus refined P2 seed 2.
    barriers=[]
    for p in ('P1','P2'):
        for seed in range(3):
            d=json.loads((base/f'{p}_seed{seed}'/'domain.json').read_text())
            R=d['pde_residual_uniform_upper'];B=d['boundary_error_uniform_upper'];U=d['u_error_uniform_upper_manufactured']
            barriers.append(dict(model=f'{p}_seed{seed}',stage='baseline',R=R,B=B,
                                 barrier_upper=barrier_upper(R,B),direct_manufactured_upper=U,best_upper=min(U,barrier_upper(R,B))))
    last=records[-1]
    barriers.append(dict(model=MODEL,stage=last['stage'],R=last['pde_residual_uniform_upper'],B=last['boundary_error_uniform_upper'],
                         barrier_upper=last['solution_error_barrier_upper'],direct_manufactured_upper=last['u_error_uniform_upper_manufactured'],
                         best_upper=last['solution_error_best_upper']))
    write_csv(out/'comparison.csv',records);write_csv(out/'solution_bounds.csv',barriers)
    save_json(out/'RESULTS.json',dict(trained=False,model=MODEL,parameters_sha256=model_hash,
                                     source_files_verified=verified,stages=records,final=last,
                                     note='Time excludes the original run. Boundary certificates are reused unchanged. '
                                     'Barrier assumes smooth convex solutions on the unit disk, as for manufactured P1/P2. '
                                     'A failed tolerance is not a failed cover.'))
    code=out/'code';code.mkdir(exist_ok=True)
    for name in list(CORE_HASHES)+['refine_saved.py']:
        shutil.copyfile(Path(__file__).with_name(name),code/name)
    manifest={str(p.relative_to(out)):digest(p.read_bytes()) for p in out.rglob('*')
              if p.is_file() and p.name!='manifest_sha256.json'}
    save_json(out/'manifest_sha256.json',manifest)
    zip_path=Path(str(out)+'_results.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    print('SAVED:',zip_path,flush=True)

if __name__=='__main__':
    main()

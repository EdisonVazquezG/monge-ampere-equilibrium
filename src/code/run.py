#!/usr/bin/env python3
"""One-command pilot reproduction and four experiments; CPU float64 + Arb."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from collections import deque
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numerics import Net, target, det, adaptive
from rigorous import RigorousNet, point_box, dyadic_box, boundary_box, outside_disk
from coverage_check import check_cover


HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)


def load_config(profile, override=None):
    cfg = json.loads((HERE/'config_full.json').read_text())
    if profile == 'smoke':
        cfg.update(seeds=[0], benchmark_points=12, benchmark_repeats=2,
                   interval_points=3, rigorous_benchmark_points=2,
                   domain_max_depth=2, domain_max_boxes=100,
                   boundary_max_depth=3, boundary_max_arcs=32)
        cfg['pilot'].update(adam_steps=2, lbfgs_steps=2, batch_interior=8, batch_boundary=8,
                            lbfgs_interior=12, lbfgs_boundary=12, test_interior=16, test_boundary=16)
    if override:
        custom = json.loads(Path(override).read_text())
        pilot = custom.pop('pilot', {})
        cfg.update(custom); cfg['pilot'].update(pilot)
    cfg['profile'] = profile
    return cfg


def environment():
    return {'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
            'device': 'cpu', 'dtype': 'float64', 'torch_threads': torch.get_num_threads(),
            'packages': {p: importlib.metadata.version(p) for p in
                         ('torch', 'numpy', 'matplotlib', 'python-flint')},
            'torch_config': torch.__config__.show()}


def training(folder, problem, seed, cfg):
    # Imported only for training. Definitions copied verbatim from the upload.
    import original_pilots as old
    pc = old.PilotConfig(seed=seed, **cfg['pilot'])
    checkpoint = folder/'trained_parameters.npz'
    if (folder/'training.json').exists():
        metadata = json.loads((folder/'training.json').read_text())
        if metadata['config'] != asdict(pc) or metadata['problem'] != problem:
            raise RuntimeError('Training config changed; use a new output directory')
        if sha(checkpoint) != metadata['parameters_sha256']:
            raise RuntimeError('Saved effective parameters changed')
        print('Reusing trained model', folder.name, flush=True)
        return
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if problem == 'P1':
        model, history, before, seconds = old.train_pilot(pc)
        after = old.validate_backward(model, pc.depth, pc.boundary_weight)
    else:
        model, history, before, after, seconds = old.p2_train(pc)
    if not (before['pass'] and after['pass']):
        raise RuntimeError('Backward validation failed; no completed marker written')
    with torch.no_grad():
        arrays = {k: v.detach().cpu().numpy() for k, v in
                  zip(('W', 'A', 'b', 'w', 'a', 'c'), model.constrained())}
    np.savez(checkpoint, **arrays)
    torch.save({'config': asdict(pc), 'problem': problem, 'state_dict': model.state_dict()}, folder/'model_state.pt')
    csv_write(folder/'training_history.csv', history)
    write_json(folder/'training.json', {'problem': problem, 'config': asdict(pc), 'seconds': seconds,
                                       'backward_before': before, 'backward_after': after,
                                       'parameters_sha256': sha(checkpoint)})


def evaluation_points(seed, cfg):
    # Matches original torch CPU sampling and reserved seeds exactly.
    gen = torch.Generator().manual_seed(900000+seed)
    n = cfg['pilot']['test_interior']; nb = cfg['pilot']['test_boundary']
    r = torch.sqrt(torch.rand(n, generator=gen, dtype=torch.float64))
    t = 2*np.pi*torch.rand(n, generator=gen, dtype=torch.float64)
    x = torch.stack((r*torch.cos(t), r*torch.sin(t)), dim=1).numpy()
    t = 2*np.pi*(torch.arange(nb, dtype=torch.float64)+.371)/nb
    xb = torch.stack((torch.cos(t), torch.sin(t)), dim=1).numpy()
    return x, xb


def evaluate(folder, problem, seed, cfg):
    net = Net(folder/'trained_parameters.npz')
    x, xb = evaluation_points(seed, cfg)
    reference = net.reference(x); u, gradient, H = net.output(x, *reference)
    ub = net.output(xb, *net.reference(xb))[0]
    truth, f = target(x, problem); gb = target(xb, problem)[0]
    reference_bound = net.certificate(x, *reference)['det']
    dref = det(H)
    np.savez_compressed(folder/'evaluation.npz', x=x, boundary_x=xb, u=u, gradient=gradient,
                        H=H, det=dref, u_exact=truth, f=f, boundary_u=ub, boundary_g=gb,
                        reference_det_diagnostic=reference_bound)
    summary = {'u_rmse': float(np.sqrt(np.mean((u-truth)**2))),
               'u_relative_l2_sample': float(np.linalg.norm(u-truth)/np.linalg.norm(truth)),
               'u_max_abs_sample': float(abs(u-truth).max()),
               'pde_residual_rmse': float(np.sqrt(np.mean((dref-f)**2))),
               'pde_residual_max_abs_sample': float(abs(dref-f).max()),
               'boundary_rmse': float(np.sqrt(np.mean((ub-gb)**2))),
               'boundary_max_abs_sample': float(abs(ub-gb).max()),
               'min_hessian_eigenvalue_sample': float(np.linalg.eigvalsh(H).min()),
               'W_norm_infinity_float64': net.q,
               'reference_det_diagnostic_max': float(reference_bound.max()),
               'interpretation': 'Float64 sample diagnostics; not interval enclosures.'}
    rows = []; pointrows = []; depth_data = {}
    for depth in cfg['depths']:
        jets = net.unroll(x, depth); dhat = det(net.output(x, *jets)[2])
        cert = net.certificate(x, *jets)['det']; error = abs(dhat-dref)
        floor = reference_bound+cfg['numerical_comparison_floor']
        signal = error > floor
        # With a floating reference, this is a diagnostic count, not a proof.
        violations = error > cert+floor
        row = {'depth': depth, 'n': len(x), 'det_error_max': float(error.max()),
               'bound_max': float(cert.max()), 'bound_median': float(np.median(cert)),
               'diagnostic_violations': int(violations.sum()), 'ratio_signal_points': int(signal.sum()),
               'bound_error_ratio_median_above_floor': float(np.median(cert[signal]/error[signal])) if signal.any() else None,
               'fraction_bound_below_5pct_pde_residual': float(np.mean(cert <= .05*abs(dhat-f)))}
        rows.append(row)
        for i in range(len(x)):
            pointrows.append({'depth': depth, 'point': i, 'det_approx': dhat[i], 'det_reference': dref[i],
                              'det_error_observed': error[i], 'bound_float64': cert[i],
                              'reference_diagnostic': reference_bound[i], 'diagnostic_violation': bool(violations[i])})
        depth_data[f'det_L{depth}'] = dhat; depth_data[f'bound_L{depth}'] = cert
    np.savez_compressed(folder/'depth_curves.npz', **depth_data)
    csv_write(folder/'certificate_points_float64.csv', pointrows)
    csv_write(folder/'certificate_depth_summary.csv', rows)
    write_json(folder/'evaluation.json', summary)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for ax, values, label in zip(axes, [u, u-truth, dref-f], ['Potential', 'Solution error', 'PDE residual']):
        im = ax.scatter(x[:, 0], x[:, 1], c=values, s=7)
        ax.set_aspect('equal'); ax.set_title(label); fig.colorbar(im, ax=ax)
    fig.savefig(folder/'sample_maps.png', dpi=160); plt.close(fig)


def benchmark(folder, problem, seed, cfg):
    net = Net(folder/'trained_parameters.npz'); rig = RigorousNet(net, cfg['interval_precision_bits'])
    with np.load(folder/'evaluation.npz') as data:
        x = data['x'][:cfg['benchmark_points']]; dref = data['det'][:len(x)]
    rows = []; raw = []
    fixed_depth = 32
    def fixed(certify=False):
        jets = net.unroll(x, fixed_depth); d = det(net.output(x, *jets)[2])
        bounds = net.certificate(x, *jets)['det'] if certify else np.full(len(x), np.nan)
        return d, bounds, np.full(len(x), fixed_depth), None
    jobs = [('fixed32', None, lambda: fixed(False)), ('fixed32_with_float_certificate', None, lambda: fixed(True))]
    jobs += [('adaptive_float64', tol, lambda tol=tol: adaptive(net, x, tol, cfg['adaptive_checkpoints']))
             for tol in cfg['adaptive_tolerances']]
    # Warmup every method; interleave timing order deterministically to reduce drift.
    for _, _, fun in jobs:
        fun()
    timings = [[] for _ in jobs]; outputs = [None]*len(jobs)
    for repeat in range(cfg['benchmark_repeats']):
        order = np.random.default_rng(1234+repeat).permutation(len(jobs))
        for j in order:
            start = time.perf_counter(); out = jobs[j][2](); elapsed = time.perf_counter()-start
            timings[j].append(elapsed); outputs[j] = out
            raw.append({'method': jobs[j][0], 'tolerance': jobs[j][1], 'repeat': repeat, 'seconds': elapsed})
    for j, (method, tol, _) in enumerate(jobs):
        dhat, bounds, depths, accepted = outputs[j]
        err = abs(dhat-dref)
        rows.append({'method': method, 'tolerance': tol, 'points': len(x),
                     'median_seconds': float(np.median(timings[j])), 'min_seconds': min(timings[j]),
                     'max_seconds': max(timings[j]), 'det_error_max_vs_reference': float(err.max()),
                     'mean_depth': float(depths.mean()), 'max_depth': int(depths.max()),
                     'bound_max': float(np.max(bounds)) if np.isfinite(bounds).all() else None,
                     'accepted_fraction': float(accepted.mean()) if accepted is not None else None,
                     'accuracy_target_met_observed': bool(err.max() <= tol) if tol is not None else None,
                     'certificate_mode': 'none' if method == 'fixed32' else 'float64_not_roundoff_verified'})
    # Separate small, equally sized cohort: rigorous stopping with all Arb cost inside timer.
    xr = x[:cfg['rigorous_benchmark_points']]; rr = dref[:len(xr)]
    def rigorous_evaluate(adapt, tol):
        values = []; bounds = []; depths = []; passed = []
        for point in xr:
            jets = net.zero(1); previous = 0
            for depth in (cfg['adaptive_checkpoints'] if adapt else [32]):
                for _ in range(depth-previous):
                    jets = net.step(point[None], *jets)
                dhat = float(det(net.output(point[None], *jets)[2])[0])
                cert = rig.certify(point_box(point), tuple(v[0] for v in jets), problem, reported_det=dhat)
                previous = depth
                if cert['det_error_upper'] <= tol:
                    break
            values.append(dhat); bounds.append(cert['det_error_upper']); depths.append(depth)
            passed.append(cert['det_error_upper'] <= tol)
        return np.array(values), np.array(bounds), np.array(depths), np.array(passed)
    for tol in cfg['adaptive_tolerances']:
        for mode in (False, True):
            rigorous_evaluate(mode, tol)
            times = []
            for repeat in range(cfg['benchmark_repeats']):
                start = time.perf_counter(); d, b, depths, ok = rigorous_evaluate(mode, tol)
                times.append(time.perf_counter()-start)
                raw.append({'method': 'adaptive_arb' if mode else 'fixed32_arb', 'tolerance': tol,
                            'repeat': repeat, 'seconds': times[-1]})
            rows.append({'method': 'adaptive_arb' if mode else 'fixed32_arb', 'tolerance': tol,
                         'points': len(xr), 'median_seconds': float(np.median(times)),
                         'min_seconds': min(times), 'max_seconds': max(times),
                         'det_error_max_vs_reference': float(abs(d-rr).max()), 'mean_depth': float(depths.mean()),
                         'max_depth': int(depths.max()), 'bound_max': float(b.max()),
                         'accepted_fraction': float(ok.mean()), 'accuracy_target_met_observed': bool(abs(d-rr).max() <= tol),
                         'certificate_mode': 'Arb_roundoff_enclosed'})
    # Only compare rows sharing point count and precision target; no forced speedup claim.
    for row in rows:
        if row['method'] == 'adaptive_float64':
            row['speedup_vs_fixed32_same_cohort'] = rows[0]['median_seconds']/row['median_seconds']
            row['speedup_comparison_meets_target'] = bool(row['accepted_fraction'] == 1 and row['accuracy_target_met_observed'] and
                                                         rows[0]['det_error_max_vs_reference'] <= row['tolerance'])
        elif row['method'] == 'adaptive_arb':
            base = next(r for r in rows if r['method'] == 'fixed32_arb' and r['tolerance'] == row['tolerance'])
            row['speedup_vs_fixed32_same_cohort'] = base['median_seconds']/row['median_seconds']
            row['speedup_comparison_meets_target'] = row['accepted_fraction'] == base['accepted_fraction'] == 1
    csv_write(folder/'benchmark.csv', rows); csv_write(folder/'benchmark_raw.csv', raw)


def intervals(folder, problem, seed, cfg):
    net = Net(folder/'trained_parameters.npz'); rig = RigorousNet(net, cfg['interval_precision_bits'])
    with np.load(folder/'evaluation.npz') as data:
        count = cfg['interval_points']
        # -1 certifies all evaluation points; otherwise a deterministic, spread subset.
        ids = np.arange(len(data['x'])) if count == -1 else np.unique(np.linspace(0, len(data['x'])-1, min(count, len(data['x'])), dtype=int))
        x = data['x'][ids]; ref = data['det'][ids]
    rows = []; start = time.perf_counter()
    for depth in cfg['depths']:
        jets = net.unroll(x, depth); d = det(net.output(x, *jets)[2]); float_cert = net.certificate(x, *jets)['det']
        for k, point in enumerate(x):
            cert = rig.certify(point_box(point), tuple(v[k] for v in jets), problem, reported_det=d[k])
            rows.append({'depth': depth, 'point': int(ids[k]), 'reported_det_float64': d[k],
                         'observed_error_vs_float_reference': abs(d[k]-ref[k]),
                         'float64_diagnostic_bound': float_cert[k], **cert})
    csv_write(folder/'certificate_points_arb.csv', rows)
    write_json(folder/'intervals.json', {'precision_bits': cfg['interval_precision_bits'], 'certified_points': len(ids),
                                       'depths': cfg['depths'], 'certified_point_depth_pairs': len(rows),
                                       'seconds': time.perf_counter()-start,
                                       'meaning': 'Rigorous point enclosures for saved binary64 effective parameters; sampled points only.'})


def domain(folder, problem, seed, cfg):
    net = Net(folder/'trained_parameters.npz'); rig = RigorousNet(net, cfg['interval_precision_bits'])
    tag = hashlib.sha256(json.dumps({k: cfg[k] for k in cfg if k.startswith(('domain_', 'boundary_', 'interval_precision'))}, sort_keys=True).encode()).hexdigest()
    cache_file = folder/'domain_cache.jsonl'; cache = {}
    model_hash = sha(folder/'trained_parameters.npz')
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            try:
                row = json.loads(line)
                if row['settings'] == tag and row['parameters_sha256'] == model_hash:
                    cache[row['id']] = row
            except json.JSONDecodeError:
                pass  # an interrupted trailing line is never used as evidence
    start = time.perf_counter()
    # Repair interrupted final record before appending; prevents concatenated JSON records.
    if cache_file.exists() and cache_file.stat().st_size:
        with cache_file.open('ab') as f:
            f.write(b'\n')
    def certify(key, box, values_only):
        if key in cache:
            return cache[key]
        center = np.array([[float(v.mid()) for v in box]])
        jets = tuple(v[0] for v in net.reference(center))
        cert = rig.certify_box(box, jets, problem, values_only=values_only)
        row = {'id': key, 'settings': tag, 'parameters_sha256': model_hash, **cert}
        with cache_file.open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush()
        cache[key] = row
        return row
    leaves = []; pending = deque([(0, 0, 0)]); examined = 0; discarded = []
    while pending:
        ix, iy, level = pending.popleft()
        if outside_disk(ix, iy, level):
            discarded.append({'ix': ix, 'iy': iy, 'level': level}); continue
        key = f'disk:{level}:{ix}:{iy}'
        cert = certify(key, dyadic_box(ix, iy, level), False); examined += 1
        # The work budget restricts refinement, never drops an uncertified region.
        split = (cert['pde_residual_upper'] > cfg['domain_residual_target'] and
                 level < cfg['domain_max_depth'] and examined+len(pending)+4 <= cfg['domain_max_boxes'])
        if split:
            pending.extend((2*ix+dx, 2*iy+dy, level+1) for dx in (0, 1) for dy in (0, 1))
        else:
            leaves.append({'ix': ix, 'iy': iy, 'level': level, **cert})
        if examined % 250 == 0:
            print(folder.name, 'domain boxes', examined, 'pending', len(pending), flush=True)
    arcs = []; pending_arcs = deque([(0, 0)]); examined_arcs = 0
    while pending_arcs:
        index, level = pending_arcs.popleft()
        cert = certify(f'boundary:{level}:{index}', boundary_box(index, level), True); examined_arcs += 1
        split = (cert['u_error_upper'] > cfg['boundary_error_target'] and level < cfg['boundary_max_depth'] and
                 examined_arcs+len(pending_arcs)+2 <= cfg['boundary_max_arcs'])
        if split:
            pending_arcs.extend([(2*index, level+1), (2*index+1, level+1)])
        else:
            arcs.append({'index': index, 'level': level, **cert})
    # A finite bound for EVERY retained leaf and arc is required to reach this write.
    check_cover(leaves, discarded, arcs)
    rb = max(r['pde_residual_upper'] for r in leaves); ub = max(r['u_error_upper'] for r in leaves)
    bb = max(r['u_error_upper'] for r in arcs)
    csv_write(folder/'domain_leaves.csv', leaves); csv_write(folder/'boundary_arcs.csv', arcs)
    csv_write(folder/'domain_discarded.csv', discarded)
    write_json(folder/'domain.json', {'coverage_complete': True, 'roundoff_enclosed': True,
                                    'domain': 'closed unit disk', 'parameters_sha256': model_hash,
                                    'pde_residual_uniform_upper': rb, 'u_error_uniform_upper_manufactured': ub,
                                    'boundary_error_uniform_upper': bb,
                                    'residual_target_met': rb <= cfg['domain_residual_target'],
                                    'boundary_target_met': bb <= cfg['boundary_error_target'],
                                    'retained_leaves': len(leaves), 'examined_boxes': examined,
                                    'discarded_strictly_outside_boxes': len(discarded), 'boundary_arcs': len(arcs),
                                    'seconds_this_invocation': time.perf_counter()-start,
                                    'precision_bits': cfg['interval_precision_bits'],
                                    'method': 'Uniform interval residuals on dyadic boxes; trigonometric interval arcs on boundary.',
                                    'interpretation': 'A valid bound need not meet the requested tolerance. No PDE stability constant is inferred.'})


def report(root, cfg):
    # Regenerate the independent V1 checks and sharpness figure with original code.
    import check_bounds
    check_bounds.OUT = root
    check_bounds.RNG = np.random.default_rng(20261015)
    check_bounds.run()
    rows = []; certrows = []; benchrows = []; intervalrows = []; domainrows = []
    for problem in cfg['problems']:
        for seed in cfg['seeds']:
            folder = root/f'{problem}_seed{seed}'
            base = {'problem': problem, 'seed': seed}
            for name, out in [('evaluation.json', rows), ('intervals.json', intervalrows), ('domain.json', domainrows)]:
                if (folder/name).exists():
                    out.append({**base, **json.loads((folder/name).read_text())})
            for name, out in [('certificate_depth_summary.csv', certrows), ('benchmark.csv', benchrows)]:
                if (folder/name).exists():
                    with (folder/name).open() as f:
                        out.extend({**base, **r} for r in csv.DictReader(f))
    for name, data in [('pilots', rows), ('depth', certrows), ('benchmark', benchrows),
                       ('intervals', intervalrows), ('domain', domainrows)]:
        csv_write(root/f'table_{name}.csv', data)
    count = sum(int(r['n']) for r in certrows)
    violations = sum(int(r['diagnostic_violations']) for r in certrows)
    status = {'profile': cfg['profile'], 'trained_models': sum((root/f'{p}_seed{s}'/'training.json').exists()
               for p in cfg['problems'] for s in cfg['seeds']),
              'evaluated_models': len(rows), 'float64_point_depth_pairs': count,
              'diagnostic_violations': violations,
              'arb_point_depth_pairs': sum(r['certified_point_depth_pairs'] for r in intervalrows),
              'completed_domain_covers': len(domainrows),
              'full_pilot_pair_count_expected': 6*6*2048,
              'full_run_completed': cfg['profile'] == 'full' and len(rows) == 6 and len(domainrows) == 6 and count == 73728
                                    and len(intervalrows) == 6 and all((root/f'{p}_seed{s}'/'benchmark_complete.json').exists()
                                    for p in cfg['problems'] for s in cfg['seeds']),
              'note': 'Counts describe this execution, not reproduction of historical results by assertion.'}
    write_json(root/'RUN_STATUS.json', status)
    if certrows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for problem, ax in zip(cfg['problems'], axes):
            for seed in cfg['seeds']:
                rr = [r for r in certrows if r['problem'] == problem and r['seed'] == seed]
                if not rr: continue
                ax.semilogy([int(r['depth']) for r in rr], [max(float(r['bound_max']), 1e-18) for r in rr], '-o', label=f'bound, seed {seed}')
                ax.semilogy([int(r['depth']) for r in rr], [max(float(r['det_error_max']), 1e-18) for r in rr], '--', label=f'error, seed {seed}')
            ax.set_title(problem+' | float64 diagnostics'); ax.set_xlabel('Depth'); ax.legend(fontsize=7)
        fig.savefig(root/'evaluation_depth.pdf'); fig.savefig(root/'evaluation_depth.png', dpi=180); plt.close(fig)
    if rows:
        table = ['% Automatically generated from this run; sample metrics, not uniform bounds.',
                 '\\begin{tabular}{llrrr}', 'Problem & Seed & RMSE $u$ & Max. PDE residual & Max. boundary error \\\\', '\\hline']
        for r in rows:
            table.append(f"{r['problem']} & {r['seed']} & {r['u_rmse']:.4g} & {r['pde_residual_max_abs_sample']:.4g} & {r['boundary_max_abs_sample']:.4g} \\\\")
        table.append('\\end{tabular}')
        (root/'pilot_table.tex').write_text('\n'.join(table)+'\n')
    if benchrows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, modes, label in zip(axes,
                [['fixed32', 'fixed32_with_float_certificate', 'adaptive_float64'], ['fixed32_arb', 'adaptive_arb']],
                ['Float64 cohort', 'Arb cohort (smaller; compare within panel)']):
            for mode in modes:
                rr = [r for r in benchrows if r['method'] == mode]
                if not rr: continue
                ax.loglog([float(r['median_seconds'])/int(r['points']) for r in rr],
                          [max(float(r['det_error_max_vs_reference']), 1e-18) for r in rr],
                          'o', label=mode, alpha=.7)
            ax.set_title(label, fontsize=10); ax.set_xlabel('Median seconds / point'); ax.set_ylabel('Max error vs float reference')
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1e'))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.legend(fontsize=7)
        fig.savefig(root/'benchmark_tradeoff.pdf'); fig.savefig(root/'benchmark_tradeoff.png', dpi=160); plt.close(fig)
    if domainrows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        labels = [f"{r['problem']} s{r['seed']}" for r in domainrows]; pos = np.arange(len(labels))
        for ax, key, samplekey, title in zip(axes,
                ['pde_residual_uniform_upper', 'boundary_error_uniform_upper'],
                ['pde_residual_max_abs_sample', 'boundary_max_abs_sample'], ['PDE residual', 'Boundary error']):
            samples = [next(r[samplekey] for r in rows if r['problem'] == d['problem'] and r['seed'] == d['seed']) for d in domainrows]
            ax.bar(pos-.18, samples, width=.36, label='Sample maximum (float64)')
            ax.bar(pos+.18, [r[key] for r in domainrows], width=.36, label='Uniform upper bound (Arb)')
            ax.set_xticks(pos, labels); ax.set_yscale('log'); ax.set_title(title); ax.legend(fontsize=7)
        fig.savefig(root/'uniform_bounds.pdf'); fig.savefig(root/'uniform_bounds.png', dpi=160); plt.close(fig)
    # Historical notebook evidence is compared explicitly, not used as a target to tune toward.
    old = json.loads((HERE/'historical_baselines.json').read_text()); comparisons = []
    for r in rows:
        for metric, expected in old.get(f"{r['problem']}_seed{r['seed']}", {}).items():
            comparisons.append({'problem': r['problem'], 'seed': r['seed'], 'metric': metric,
                                'historical_notebook': expected, 'this_run': r[metric],
                                'absolute_difference': abs(r[metric]-expected)})
    csv_write(root/'historical_comparison.csv', comparisons)
    (root/'READ_RESULTS.txt').write_text(
        'RUN_STATUS.json: what actually completed. Smoke is never manuscript evidence.\n'
        'table_pilots.csv / pilot_table.tex: sample metrics.\n'
        'table_depth.csv / evaluation_depth.pdf: float64 certificate diagnostics.\n'
        'table_benchmark.csv: end-to-end CPU timing, with certificate cost where stated. Compare identical point counts.\n'
        'table_intervals.csv: certified point/depth counts; separate from float64 diagnostics.\n'
        'table_domain.csv: valid uniform bounds on disk and boundary; check target_met flags separately.\n'
        'historical_comparison.csv: discrepancies from saved notebook outputs, not a bitwise-reproduction assertion.\n'
        'The report supplied earlier used different sampled maxima; see METHODS.txt.\n'
        'Every JSON/CSV bound exported by rigorous.py is rounded outward to binary64. Preserve original files.\n')
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest_sha256.json'}
    write_json(root/'manifest_sha256.json', files)
    shutil.make_archive(str(root)+'_results', 'zip', root)
    print(json.dumps(status, indent=2), flush=True)
    print('Results:', str(root)+'_results.zip', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'full'], default='full')
    parser.add_argument('--stage', choices=['all', 'train', 'evaluate', 'benchmark', 'interval', 'domain', 'report'], default='all')
    parser.add_argument('--out', default='results_full')
    parser.add_argument('--config', help='JSON overrides; use a new output folder for a changed protocol')
    args = parser.parse_args(); cfg = load_config(args.profile, args.config)
    root = Path(args.out).resolve(); root.mkdir(parents=True, exist_ok=True)
    protocol = root/'protocol.json'
    source_hashes = {p.name: sha(p) for p in HERE.glob('*.py')}
    if protocol.exists() and json.loads(protocol.read_text()) != cfg:
        raise RuntimeError('Protocol differs from existing output. Choose a new --out directory.')
    sources = root/'source_hashes.json'
    if sources.exists() and json.loads(sources.read_text()) != source_hashes:
        raise RuntimeError('Source changed. Choose a new --out directory to preserve provenance.')
    torch.set_num_threads(1)
    write_json(protocol, cfg); write_json(sources, source_hashes)
    write_json(root/'environment.json', environment())
    (root/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    if args.stage == 'report':
        report(root, cfg); return
    functions = {'train': training, 'evaluate': evaluate, 'benchmark': benchmark, 'interval': intervals, 'domain': domain}
    for problem in cfg['problems']:
        for seed in cfg['seeds']:
            folder = root/f'{problem}_seed{seed}'; folder.mkdir(exist_ok=True)
            stages = list(functions) if args.stage == 'all' else [args.stage]
            for stage in stages:
                marker = folder/f'{stage}_complete.json'
                if marker.exists():
                    metadata = json.loads(marker.read_text())
                    if metadata['parameters_sha256'] != sha(folder/'trained_parameters.npz'):
                        raise RuntimeError('Model changed after completed stage')
                    print('Already complete:', folder.name, stage, flush=True); continue
                print('RUN', folder.name, stage, flush=True)
                functions[stage](folder, problem, seed, cfg)
                write_json(marker, {'parameters_sha256': sha(folder/'trained_parameters.npz'), 'stage': stage})
    report(root, cfg)


if __name__ == '__main__':
    main()

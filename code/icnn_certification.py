#!/usr/bin/env python3
"""X7: can an explicit ICNN be certified with the same Arb machinery, and at what cost?

Question. The equilibrium model has a certificate (Theorem 5.1, Proposition 6.1) built around the fixed
point. An explicit ICNN of similar size is a finite composition, so its jets can be enclosed directly
with ball arithmetic. This script certifies BOTH models with the same Arb precision, the same boxes
and the same priority-refinement protocol as X4, so that the comparison is not about the code path.

Parts
  A  pointwise: Arb cost per point and width of the determinant enclosure (ICNN vs equilibrium)
  B  uniform residual over the unit disk: R versus certified boxes and CPU time (ICNN vs equilibrium),
     together with the ratio R / (largest sampled residual), which separates looseness of the
     certificate from the quality of the trained model
  C  depth sweep of the explicit ICNN (more layers, same width): does the box enclosure degrade
     with depth, as natural interval extension typically does?
  D  naive a priori Hessian bound for the ICNN by layer recursion (analytic, no evaluation)

What this does NOT show: that no cleverer certificate exists for ICNNs. The ICNN enclosure here is the
natural interval extension of the exact jet recursion (with a tight enclosure of sigma(1-sigma)).

Usage (paths as in the Colab notebook):
    python icnn_certification.py --profile full \
        --extended-run ../results/extended_full --paper-run ../results/paper_full \
        --out ../results/X7_icnn_certification
    python icnn_certification.py --selftest --extended-run ... --paper-run ...
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import heapq
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flint import arb, ctx

import extended as ext            # registers P3/P4; reuses training, evaluation and equilibrium loading
import rigorous
from rigorous import (RigorousNet, point_box, dyadic_box, outside_disk, exact, export_upper,
                      softplus_ball, sigmoid_ball, dot)
from numerics import Net, det
from coverage_check import check_cover

write_json, csv_write = ext.write_json, ext.csv_write
PRECISION = 128

SETTINGS = {
    'full': {'problems': ['P1', 'P2', 'P3'], 'point_seeds': [0, 1, 2], 'points': 32,
             'cover_seed': 0, 'checkpoints': [100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
             'initial_level': 2, 'max_level': 11,
             'depth_layers': [3, 6, 12], 'depth_problem': 'P2', 'depth_seed': 0, 'depth_checkpoints': [1000, 4000]},
    'smoke': {'problems': ['P1', 'P2'], 'point_seeds': [0], 'points': 4,
              'cover_seed': 0, 'checkpoints': [50, 100, 200],
              'initial_level': 2, 'max_level': 7,
              'depth_layers': [3, 4], 'depth_problem': 'P2', 'depth_seed': 0, 'depth_checkpoints': [100, 200]},
}


# ----------------------------------------------------------------------------------------------
# Arb enclosure of the explicit ICNN jets
# ----------------------------------------------------------------------------------------------
def gamma_ball(d):
    """Enclosure of s(1-s) for s in the ball d, using monotonicity on each side of 1/2."""
    lo, hi = d.lower(), d.upper()
    g_lo, g_hi = lo*(1-lo), hi*(1-hi)
    half = arb(1)/2
    lower = g_lo.lower() if g_lo.lower() <= g_hi.lower() else g_hi.lower()
    if hi <= half or lo >= half:
        upper = g_lo.upper() if g_lo.upper() >= g_hi.upper() else g_hi.upper()
    else:
        upper = arb(1)/4
    return lower.union(upper)


class ICNNArb:
    """z1 = sp(A0 x + b0); z_{k+1} = sp(W_k z_k + A_k x + b_k); u = w.z + a.x + c  (binary64 parameters, exactly)."""

    def __init__(self, folder, precision=PRECISION):
        ctx.prec = precision
        with np.load(Path(folder)/'trained_parameters.npz') as f:
            P = {k: f[k] for k in f.files}
        self.P = P
        self.layers = sum(1 for k in P if k.startswith('A') and k[1:].isdigit())
        self.m = len(P['w'])
        conv = lambda M: [[exact(v) for v in row] for row in M]
        self.A = [conv(P[f'A{k}']) for k in range(self.layers)]
        self.b = [[exact(v) for v in P[f'b{k}']] for k in range(self.layers)]
        self.W = [None] + [conv(P[f'W{k}']) for k in range(1, self.layers)]
        self.w = [exact(v) for v in P['w']]; self.a = [exact(v) for v in P['a']]; self.c = exact(P['c'])
        self.precision = precision

    def jets(self, x):
        """Enclosures of u, grad u, and (H00, H01, H11) valid for every point of the box x."""
        ctx.prec = self.precision
        m = self.m
        z = J = K = None
        pairs = [(0, 0), (0, 1), (1, 1)]
        for k in range(self.layers):
            if k == 0:
                s = [dot(self.A[0][r], x)+self.b[0][r] for r in range(m)]
            else:
                s = [dot(self.W[k][r], z)+dot(self.A[k][r], x)+self.b[k][r] for r in range(m)]
            d = [sigmoid_ball(t) for t in s]; g = [gamma_ball(di) for di in d]
            znew = [softplus_ball(t) for t in s]
            if k == 0:
                V = [[self.A[0][r][i] for i in range(2)] for r in range(m)]
                Kn = [[g[r]*V[r][i]*V[r][j] for (i, j) in pairs] for r in range(m)]
            else:
                V = [[self.A[k][r][i]+dot(self.W[k][r], [J[t][i] for t in range(m)]) for i in range(2)] for r in range(m)]
                Kn = [[d[r]*dot(self.W[k][r], [K[t][c] for t in range(m)])+g[r]*V[r][i]*V[r][j]
                       for c, (i, j) in enumerate(pairs)] for r in range(m)]
            J = [[d[r]*V[r][i] for i in range(2)] for r in range(m)]
            K, z = Kn, znew
        u = dot(self.w, z)+dot(self.a, x)+self.c
        grad = [dot(self.w, [J[r][i] for r in range(m)])+self.a[i] for i in range(2)]
        H = [dot(self.w, [K[r][c] for r in range(m)]) for c in range(3)]
        return u, grad, H

    def residual_upper(self, x, problem):
        _, _, H = self.jets(x)
        _, f = rigorous.target_box(x, problem)
        determinant = H[0]*H[2]-H[1]*H[1]
        return export_upper((determinant-f).abs_upper()), determinant


# ----------------------------------------------------------------------------------------------
# Float64 reference jets of the ICNN (for tests)
# ----------------------------------------------------------------------------------------------
def icnn_jets_np(P, x):
    layers = sum(1 for k in P if k.startswith('A') and k[1:].isdigit()); n, m = len(x), len(P['w'])
    z = np.zeros((n, m)); J = np.zeros((n, m, 2)); K = np.zeros((n, m, 2, 2))
    for k in range(layers):
        W = P[f'W{k}'] if k > 0 else np.zeros((m, m))
        s = z @ W.T + x @ P[f'A{k}'].T + P[f'b{k}']
        d = np.exp(-np.logaddexp(0, -s))
        V = np.einsum('ab,nbi->nai', W, J) + P[f'A{k}']
        K = d[..., None, None]*np.einsum('ab,nbij->naij', W, K) + (d*(1-d))[..., None, None]*V[..., :, None]*V[..., None, :]
        J = d[..., None]*V; z = np.logaddexp(0, s)
    return (z @ P['w'] + x @ P['a'] + P['c'], np.einsum('m,nmi->ni', P['w'], J) + P['a'],
            np.einsum('m,nmij->nij', P['w'], K))


def naive_hessian_bound(P):
    """Entrywise bound |H_ij| <= ||w||_1 h_L by layer recursion (|sigma'|<=1, sigma(1-sigma)<=1/4).

    g_1 = a_0, h_1 = a_0^2/4;  U_k = ||W_k||_inf g_k + a_k,  g_{k+1} = U_k,  h_{k+1} = ||W_k||_inf h_k + U_k^2/4,
    with a_k the largest column sup-norm of A_k. It uses neither the cancellation nor any structure of W:
    it is a plain bound, not claimed to be the best possible for an ICNN."""
    layers = sum(1 for k in P if k.startswith('A') and k[1:].isdigit())
    acol = lambda A: float(np.abs(A).max())
    g, h = acol(P['A0']), acol(P['A0'])**2/4
    for k in range(1, layers):
        q = float(np.abs(P[f'W{k}']).sum(1).max()); U = q*g + acol(P[f'A{k}'])
        g, h = U, q*h + U*U/4
    return float(np.abs(P['w']).sum()*h)


# ----------------------------------------------------------------------------------------------
# Generic priority cover (same logic as X4)
# ----------------------------------------------------------------------------------------------
def priority_cover(certify, level0, max_level, checkpoints, tag):
    start = time.perf_counter(); evaluated = 0

    def counted(ix, iy, level):
        nonlocal evaluated
        evaluated += 1
        return certify(ix, iy, level)

    n0 = 2**level0; cells, discarded = {}, []
    for ix in range(n0):
        for iy in range(n0):
            if outside_disk(ix, iy, level0):
                discarded.append({'ix': ix, 'iy': iy, 'level': level0})
            else:
                cells[(ix, iy, level0)] = counted(ix, iy, level0)
    heap = [(-r, k) for k, r in cells.items()]; heapq.heapify(heap)
    checkpoints = sorted(checkpoints); nxt = 0; curve = []

    def snapshot():
        row = {'evaluated_boxes': evaluated, 'retained_boxes': len(cells), 'max_level': max(k[2] for k in cells),
               'R_uniform': max(cells.values()), 'seconds': time.perf_counter()-start}
        curve.append(row); print(tag, row, flush=True)

    while heap and nxt < len(checkpoints):
        if evaluated >= checkpoints[nxt]:
            snapshot(); nxt += 1; continue
        negR, key = heapq.heappop(heap)
        if key not in cells or -negR != cells[key]:
            continue
        ix, iy, level = key
        if level >= max_level:
            continue
        parent = cells.pop(key)
        for dx in (0, 1):
            for dy in (0, 1):
                c = (2*ix+dx, 2*iy+dy, level+1)
                if outside_disk(*c):
                    discarded.append({'ix': c[0], 'iy': c[1], 'level': c[2]}); continue
                cells[c] = min(counted(*c), parent); heapq.heappush(heap, (-cells[c], c))   # parent bound stays valid
    if not curve or curve[-1]['evaluated_boxes'] != evaluated:
        snapshot()
    check_cover([{'ix': k[0], 'iy': k[1], 'level': k[2]} for k in cells], discarded, [{'index': 0, 'level': 0}])
    return curve


def eq_certify_factory(net, rig, problem, counter=None):
    def certify(ix, iy, level):
        box = dyadic_box(ix, iy, level)
        centre = np.array([[float(v.mid()) for v in box]])
        jets = tuple(v[0] for v in net.reference(centre))
        return rig.certify_box(box, jets, problem)['pde_residual_upper']
    return certify


def icnn_certify_factory(icnn, problem):
    def certify(ix, iy, level):
        return icnn.residual_upper(dyadic_box(ix, iy, level), problem)[0]
    return certify


# ----------------------------------------------------------------------------------------------
# Parts
# ----------------------------------------------------------------------------------------------
def icnn_folder(extended_run, problem, seed):
    return Path(extended_run)/'X2_explicit_icnn'/f'icnn_{problem}_seed{seed}'


def part_a_pointwise(cfg, S, args, out):
    marker = out/'A_pointwise_complete.json'
    if marker.exists():
        return json.loads(marker.read_text())['rows']
    rows = []
    for problem in S['problems']:
        for seed in S['point_seeds']:
            ef = ext.equilibrium_folder(Path(args.extended_run), cfg, problem, seed)
            net = Net(ef/'trained_parameters.npz'); rig = RigorousNet(net, PRECISION)
            icnn = ICNNArb(icnn_folder(args.extended_run, problem, seed))
            x, _ = ext.run.evaluation_points(seed, ext.paper_cfg(cfg)); x = x[:S['points']]
            # ICNN
            t0 = time.perf_counter(); half = []; res = []
            for xi in x:
                r, dd = icnn.residual_upper(point_box(xi), problem)
                half.append(float(dd.rad())); res.append(r)
            t_icnn = (time.perf_counter()-t0)/len(x)
            # equilibrium (certificate only; the converged float64 state is computed first and timed apart)
            t0 = time.perf_counter(); refs = [tuple(v[0] for v in net.reference(xi[None])) for xi in x]
            t_ref = (time.perf_counter()-t0)/len(x)
            t0 = time.perf_counter(); eq = [rig.certify(point_box(xi), j, problem) for xi, j in zip(x, refs)]
            t_eq = (time.perf_counter()-t0)/len(x)
            rows.append({'problem': problem, 'seed': seed, 'points': len(x),
                         'icnn_arb_seconds_per_point': t_icnn, 'icnn_det_enclosure_radius_max': max(half),
                         'icnn_residual_upper_max': max(res),
                         'eq_arb_certificate_seconds_per_point': t_eq, 'eq_float_reference_seconds_per_point': t_ref,
                         'eq_det_error_upper_max': max(e['det_error_upper'] for e in eq),
                         'eq_residual_upper_max': max(e['pde_residual_upper'] for e in eq)})
            print('A', rows[-1], flush=True)
    write_json(marker, {'rows': rows}); csv_write(out/'table_X7A_pointwise.csv', rows)
    return rows


def part_b_cover(cfg, S, args, out):
    marker = out/'B_cover_complete.json'
    if marker.exists():
        return json.loads(marker.read_text())['rows']
    rows = []; seed = S['cover_seed']
    for problem in S['problems']:
        ef = ext.equilibrium_folder(Path(args.extended_run), cfg, problem, seed)
        net, _, _, eq_eval = ext.evaluate_equilibrium(ef, problem, seed, cfg)
        rig = RigorousNet(net, PRECISION)
        icf = icnn_folder(args.extended_run, problem, seed)
        ic_eval = ext.evaluate_explicit(icf, problem, seed, cfg); icnn = ICNNArb(icf)
        for model, certify, sample in [
                ('equilibrium', eq_certify_factory(net, rig, problem), eq_eval['pde_residual_max_abs_sample']),
                ('icnn', icnn_certify_factory(icnn, problem), ic_eval['pde_residual_max_abs_sample'])]:
            curve = priority_cover(certify, S['initial_level'], S['max_level'], S['checkpoints'], f'B {problem} {model}')
            for r in curve:
                rows.append({'problem': problem, 'seed': seed, 'model': model, 'sample_residual_max': sample,
                             'R_over_sample_max': r['R_uniform']/sample, **r})
    write_json(marker, {'rows': rows}); csv_write(out/'table_X7B_cover.csv', rows)
    return rows


def part_c_depth(cfg, S, args, out):
    marker = out/'C_depth_complete.json'
    if marker.exists():
        return json.loads(marker.read_text())['rows']
    rows = []; problem, seed = S['depth_problem'], S['depth_seed']
    for layers in S['depth_layers']:
        folder = out/'icnn_depth'/f'L{layers}_{problem}_seed{seed}'
        pc = dict(cfg['pilot'])
        meta = ext.train(folder, problem, seed, pc, ext.ExplicitICNN(8, layers, seed))
        ev = ext.evaluate_explicit(folder, problem, seed, cfg)
        icnn = ICNNArb(folder)
        curve = priority_cover(icnn_certify_factory(icnn, problem), S['initial_level'], S['max_level'],
                               S['depth_checkpoints'], f'C L={layers}')
        with np.load(folder/'trained_parameters.npz') as f:
            P = {k: f[k] for k in f.files}
        t0 = time.perf_counter(); icnn.residual_upper(dyadic_box(8, 8, 4), problem); t_box = time.perf_counter()-t0
        for r in curve:
            rows.append({'layers': layers, 'parameters': meta['trainable_parameters'], 'problem': problem, 'seed': seed,
                         'u_relative_l2_sample': ev['u_relative_l2_sample'],
                         'sample_residual_max': ev['pde_residual_max_abs_sample'],
                         'R_over_sample_max': r['R_uniform']/ev['pde_residual_max_abs_sample'],
                         'one_box_seconds': t_box, 'naive_hessian_bound': naive_hessian_bound(P), **r})
    write_json(marker, {'rows': rows}); csv_write(out/'table_X7C_depth.csv', rows)
    return rows


def part_d_apriori(cfg, S, args, out):
    rows = []
    sources = [(f'icnn_{p}_seed{s}', icnn_folder(args.extended_run, p, s), p, s) for p in S['problems'] for s in S['point_seeds']]
    sources += [(f'icnn_depth_L{L}', out/'icnn_depth'/f'L{L}_{S["depth_problem"]}_seed{S["depth_seed"]}', S['depth_problem'], S['depth_seed'])
                for L in S['depth_layers']]
    for name, folder, problem, seed in sources:
        if not (folder/'trained_parameters.npz').exists():
            continue
        with np.load(folder/'trained_parameters.npz') as f:
            P = {k: f[k] for k in f.files}
        x, _ = ext.run.evaluation_points(seed, ext.paper_cfg(cfg))
        H = icnn_jets_np(P, x)[2]; observed = float(np.abs(H).max())
        bound = naive_hessian_bound(P)
        if observed > bound*(1+1e-12):
            raise RuntimeError(f'Naive a priori bound violated for {name}: {observed} > {bound}')
        rows.append({'model': name, 'layers': sum(1 for k in P if k.startswith('A') and k[1:].isdigit()),
                     'max_row_sum_W': max([float(np.abs(P[f'W{k}']).sum(1).max()) for k in range(1, 99) if f'W{k}' in P] or [0.0]),
                     'observed_max_abs_hessian_entry': observed, 'naive_apriori_bound': bound, 'bound_over_observed': bound/observed})
    csv_write(out/'table_X7D_apriori.csv', rows)
    return rows


def figures(out, rowsB, rowsC):
    if rowsB:
        problems = sorted(set(r['problem'] for r in rowsB))
        fig, axes = plt.subplots(1, len(problems), figsize=(4*len(problems), 3.6), constrained_layout=True, squeeze=False)
        for ax, p in zip(axes[0], problems):
            for model, style in [('equilibrium', 'o-'), ('icnn', 's--')]:
                rr = [r for r in rowsB if r['problem'] == p and r['model'] == model]
                ax.loglog([r['evaluated_boxes'] for r in rr], [r['R_uniform'] for r in rr], style, label=model)
            ax.axhline(0.25, color='k', ls=':', lw=.8); ax.set_title(p); ax.set_xlabel('Certified boxes')
            ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
        axes[0][0].set_ylabel('Uniform residual bound $R$')
        fig.savefig(out/'X7B_cover.pdf'); fig.savefig(out/'X7B_cover.png', dpi=160); plt.close(fig)
    if rowsC:
        fig, ax = plt.subplots(figsize=(4.5, 3.6), constrained_layout=True)
        for cp in sorted(set(r['evaluated_boxes'] for r in rowsC)):
            rr = [r for r in rowsC if r['evaluated_boxes'] == cp]
            ax.semilogy([r['layers'] for r in rr], [r['R_over_sample_max'] for r in rr], 'o-', label=f'{cp} boxes')
        ax.set_xlabel('ICNN layers'); ax.set_ylabel('$R$ / largest sampled residual'); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
        fig.savefig(out/'X7C_depth.pdf'); fig.savefig(out/'X7C_depth.png', dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------------------
def selftest(cfg, args):
    """(1) Arb point enclosure contains the float64 jets; (2) a box enclosure contains float64 jets at random
    points of the box; (3) numpy jets agree with the PyTorch ExplicitICNN; (4) the naive a priori bound holds."""
    folder = icnn_folder(args.extended_run, 'P2', 0)
    icnn = ICNNArb(folder); P = icnn.P
    rng = np.random.default_rng(1)
    pts = rng.uniform(-.7, .7, size=(6, 2))
    u_np, g_np, H_np = icnn_jets_np(P, pts)
    for i, p in enumerate(pts):
        u, g, H = icnn.jets(point_box(p))
        assert abs(float(u.mid())-u_np[i]) <= float(u.rad())+1e-12*(1+abs(u_np[i])), 'u point'
        for c, (a, b) in enumerate([(0, 0), (0, 1), (1, 1)]):
            assert abs(float(H[c].mid())-H_np[i, a, b]) <= float(H[c].rad())+1e-10*(1+abs(H_np[i, a, b])), 'H point'
    box = dyadic_box(5, 9, 4)
    lo = [float(v.mid()-v.rad()) for v in box]; hi = [float(v.mid()+v.rad()) for v in box]
    u, g, H = icnn.jets(box); _, f = rigorous.target_box(box, 'P2')
    samples = rng.uniform(lo, hi, size=(200, 2)); _, _, Hs = icnn_jets_np(P, samples)
    for c, (a, b) in enumerate([(0, 0), (0, 1), (1, 1)]):
        assert Hs[:, a, b].min() >= float(H[c].mid()-H[c].rad())-1e-12 and Hs[:, a, b].max() <= float(H[c].mid()+H[c].rad())+1e-12, 'H box'
    R, _ = icnn.residual_upper(box, 'P2')
    fs = ext.PROBLEMS['P2'].f_np(samples)
    assert np.abs(det(Hs)-fs).max() <= R+1e-12, 'residual box'
    state = torch.load(folder/'model_state.pt')['state_dict']
    layers = icnn.layers; model = ext.ExplicitICNN(icnn.m, layers, 0); model.load_state_dict(state)
    xt = torch.tensor(pts, dtype=torch.float64)
    with torch.no_grad():
        ut, gt, Ht, _ = model.jets(xt)
    assert np.allclose(ut.numpy(), u_np, atol=1e-11) and np.allclose(Ht.numpy(), H_np, atol=1e-10), 'torch vs numpy'
    obs = np.abs(icnn_jets_np(P, rng.uniform(-1, 1, size=(500, 2)))[2]).max()
    assert obs <= naive_hessian_bound(P), 'naive bound'
    print('selftest passed: box R =', R, '| observed |H| max =', obs, '| naive bound =', naive_hessian_bound(P))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', choices=['smoke', 'full'], default='full')
    parser.add_argument('--extended-run', required=True, help='Completed extended.py output (X2 ICNNs, X3 P3 models)')
    parser.add_argument('--paper-run', required=True, help='Completed run.py output (P1/P2 equilibrium models)')
    parser.add_argument('--out', default='results_X7_icnn_certification')
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--parts', nargs='+', default=['A', 'B', 'C', 'D'], choices=['A', 'B', 'C', 'D'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = ext.load_config(args.profile); cfg['paper_run'] = str(Path(args.paper_run).resolve())
    if args.selftest:
        selftest(cfg, args); return
    S = SETTINGS[args.profile]; out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    write_json(out/'settings.json', {'profile': args.profile, 'settings': S, 'precision_bits': PRECISION})
    rowsB = rowsC = []
    if 'A' in args.parts: part_a_pointwise(cfg, S, args, out)
    if 'B' in args.parts: rowsB = part_b_cover(cfg, S, args, out)
    if 'C' in args.parts: rowsC = part_c_depth(cfg, S, args, out)
    if 'D' in args.parts: part_d_apriori(cfg, S, args, out)
    figures(out, rowsB, rowsC)
    print('Done:', out)


if __name__ == '__main__':
    main()

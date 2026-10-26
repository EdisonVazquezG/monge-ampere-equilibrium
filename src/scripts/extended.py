#!/usr/bin/env python3
"""Extended experiments for the convex equilibrium Monge-Ampere paper.

X1  near-critical contraction factor q -> 1 (curvature constants, accuracy, certification cost)
X2  explicit ICNN of similar size trained with the same protocol
X3  harder manufactured problems: P3 (radial exponential) and P4 (degenerate right-hand side)
X4  tightness of the uniform residual certificate under adaptive refinement
X5  finite-difference Newton baseline on the same problems
X6  certification cost as the network width grows

The certified modules (numerics.py, rigorous.py, coverage_check.py) are used unchanged, so the
paper reproduction in run.py keeps its provenance hashes. New problems are registered by
extending the target functions at import time (see `register_problems`).

Usage:
    python extended.py --profile smoke --out results_extended_smoke
    python extended.py --profile full  --out results_extended --paper-run results_full
    python extended.py --profile full  --out results_extended --experiments X1 X5
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import copy
import csv
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numerics
import rigorous
import run
import original_pilots as old
from numerics import Net, det
from rigorous import RigorousNet, point_box, dyadic_box, boundary_box, outside_disk, exact, export_upper
from coverage_check import check_cover

HERE = Path(__file__).resolve().parent
DTYPE = torch.float64
write_json, csv_write, sha = run.write_json, run.csv_write, run.sha


# ----------------------------------------------------------------------------------------------
# Problems
# ----------------------------------------------------------------------------------------------
class Problem:
    """Manufactured solution u* and right-hand side f = det D^2 u*, in NumPy, PyTorch and Arb."""

    def __init__(self, name, description, u_np, f_np, u_t, f_t, box):
        self.name, self.description = name, description
        self.u_np, self.f_np, self.u_t, self.f_t, self.box = u_np, f_np, u_t, f_t, box


def _arb_sq(v):
    return v*v


PROBLEMS = {
    'P1': Problem('P1', 'u = x1^2 + x2^2/4, f = 1',
                  lambda x: x[:, 0]**2 + x[:, 1]**2/4, lambda x: np.ones(len(x)),
                  lambda x: x[:, 0]**2 + 0.25*x[:, 1]**2, lambda x: torch.ones(len(x), dtype=x.dtype),
                  None),
    'P2': Problem('P2', 'u = |x|^2/2 + 0.1(x1^4+x2^4), f = (1+1.2x1^2)(1+1.2x2^2)',
                  lambda x: .5*(x*x).sum(1) + .1*(x**4).sum(1),
                  lambda x: (1 + 1.2*x[:, 0]**2)*(1 + 1.2*x[:, 1]**2),
                  old.p2_u, old.p2_f, None),
    'P3': Problem('P3', 'u = exp(|x|^2/2), f = (1+|x|^2) exp(|x|^2)  [Nystrom-Vestberg, Ex. 4.1]',
                  lambda x: np.exp((x*x).sum(1)/2),
                  lambda x: (1 + (x*x).sum(1))*np.exp((x*x).sum(1)),
                  lambda x: torch.exp((x*x).sum(1)/2),
                  lambda x: (1 + (x*x).sum(1))*torch.exp((x*x).sum(1)),
                  lambda x: ((_arb_sq(x[0]) + _arb_sq(x[1]))/2).exp(),
                  ),
    'P4': Problem('P4', 'u = x1^4/12 + x2^2/2, f = x1^2  (degenerate on x1 = 0)',
                  lambda x: x[:, 0]**4/12 + x[:, 1]**2/2,
                  lambda x: x[:, 0]**2,
                  lambda x: x[:, 0]**4/12 + 0.5*x[:, 1]**2,
                  lambda x: x[:, 0]**2,
                  None),
}


def _box_P3(x):
    r2 = _arb_sq(x[0]) + _arb_sq(x[1])
    return (r2/2).exp(), (1 + r2)*r2.exp()


def _box_P4(x):
    x2 = _arb_sq(x[0])
    return _arb_sq(x2)/12 + _arb_sq(x[1])/2, x2


def register_problems():
    """Extend the target functions used by numerics/rigorous/run with P3 and P4.

    P1 and P2 are delegated to the original, unchanged implementations.
    """
    if getattr(rigorous, '_extended_problems', False):
        return
    original_target, original_box = numerics.target, rigorous.target_box

    def target(x, problem):
        if problem in ('P1', 'P2'):
            return original_target(x, problem)
        p = PROBLEMS[problem]
        return p.u_np(x), p.f_np(x)

    def target_box(x, problem):
        if problem in ('P1', 'P2'):
            return original_box(x, problem)
        return {'P3': _box_P3, 'P4': _box_P4}[problem](x)

    numerics.target = target
    rigorous.target_box = target_box   # looked up at call time inside RigorousNet.certify
    run.target = target                # used by run.evaluate
    rigorous._extended_problems = True


register_problems()


def verify_problems():
    """Check f = det D^2 u* with autograd for every registered problem."""
    x = torch.tensor([[0., 0.], [.3, .4], [-.7, .2], [.1, -.9]], dtype=DTYPE)
    report = {}
    for name, p in PROBLEMS.items():
        H = torch.stack([torch.autograd.functional.hessian(lambda t: p.u_t(t[None])[0], xi) for xi in x])
        err = float((old.determinant(H) - p.f_t(x)).abs().max())
        report[name] = err
        if err > 1e-11:
            raise AssertionError(f'Manufactured data inconsistent for {name}: {err}')
    return report


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
class FixedRowSumPilot(old.ConvexPilot):
    """Equilibrium network with every row sum of W equal to q, so ||W||_inf = q exactly.

    Used for the near-critical study: the contraction gap 1 - q is then the actual gap.
    """

    def __init__(self, width=12, q=0.7, seed=0):
        super().__init__(width, q, seed)
        del self.row_gate   # unused here; removing it keeps the parameter count honest

    def constrained(self):
        W = self.q*torch.softmax(self.raw_W, dim=1)
        w = old.sp(self.raw_w)/self.width
        return W, self.A, self.b, w, self.a, self.c


class ExplicitICNN(nn.Module):
    """Explicit input convex network: z1 = sp(A0 x + b0), z_{k+1} = sp(W_k z_k + A_k x + b_k).

    W_k = softplus(V_k)/width >= 0 and w = softplus(v)/width >= 0, as in the equilibrium model.
    With width 8 and 3 hidden layers it has 211 trainable parameters (equilibrium m = 12: 207).
    """
    is_explicit = True

    def __init__(self, width=8, layers=3, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.width, self.layers = width, layers
        self.A = nn.ParameterList([nn.Parameter(1.5*torch.randn(width, 2, generator=gen, dtype=DTYPE))
                                   for _ in range(layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(width, dtype=DTYPE)) for _ in range(layers)])
        self.V = nn.ParameterList([nn.Parameter(0.1*torch.randn(width, width, generator=gen, dtype=DTYPE))
                                   for _ in range(layers-1)])
        self.raw_w = nn.Parameter(torch.full((width,), float(np.log(np.expm1(1.0))), dtype=DTYPE))
        self.a = nn.Parameter(torch.zeros(2, dtype=DTYPE))
        self.c = nn.Parameter(torch.zeros((), dtype=DTYPE))

    def weights(self):
        return [old.sp(V)/self.width for V in self.V], old.sp(self.raw_w)/self.width

    def value_unrolled(self, x, depth=None):
        Ws, w = self.weights()
        z = old.sp(x @ self.A[0].T + self.b[0])
        for W, A, b in zip(Ws, self.A[1:], self.b[1:]):
            z = old.sp(z @ W.T + x @ A.T + b)
        return z @ w + x @ self.a + self.c

    def jets(self, x, depth=None):
        Ws, w = self.weights()
        n = len(x)
        z = x.new_zeros(n, self.width); J = x.new_zeros(n, self.width, 2); K = x.new_zeros(n, self.width, 2, 2)
        for k in range(self.layers):
            W = Ws[k-1] if k > 0 else torch.zeros(self.width, self.width, dtype=x.dtype)
            s = z @ W.T + x @ self.A[k].T + self.b[k]
            d = torch.sigmoid(s)
            V = torch.einsum('ab,nbi->nai', W, J) + self.A[k][None]
            K = d[:, :, None, None]*torch.einsum('ab,nbij->naij', W, K) + \
                (d*(1-d))[:, :, None, None]*V[:, :, :, None]*V[:, :, None, :]
            J = d[:, :, None]*V
            z = old.sp(s)
        return (z @ w + x @ self.a + self.c, torch.einsum('m,nmi->ni', w, J) + self.a,
                torch.einsum('m,nmij->nij', w, K), z)

    def export(self):
        Ws, w = self.weights()
        arrays = {'w': w, 'a': self.a, 'c': self.c}
        for k in range(self.layers):
            arrays[f'A{k}'] = self.A[k]; arrays[f'b{k}'] = self.b[k]
        for k, W in enumerate(Ws, start=1):
            arrays[f'W{k}'] = W
        return {k: v.detach().numpy() for k, v in arrays.items()}


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


# ----------------------------------------------------------------------------------------------
# Training (same protocol as the paper: Adam with step decay, then L-BFGS on fixed points)
# ----------------------------------------------------------------------------------------------
def strong_loss(model, xi, xb, depth, boundary_weight, problem):
    p = PROBLEMS[problem]
    _, _, H, _ = model.jets(xi, depth)
    loss_pde = ((old.determinant(H) - p.f_t(xi))**2).mean()
    loss_bc = ((model.value_unrolled(xb, depth) - p.u_t(xb))**2).mean()
    return loss_pde + boundary_weight*loss_bc, loss_pde, loss_bc


def check_backward(model, depth, boundary_weight, problem):
    """Input Hessian vs autograd, and parameter gradient vs a central finite difference."""
    gen = torch.Generator().manual_seed(8123)
    xi, xb = old.disk_points(2, gen), old.boundary_points(5, gen)
    _, _, H, _ = model.jets(xi, depth)
    Href = torch.stack([torch.autograd.functional.hessian(
        lambda t: model.value_unrolled(t[None], depth)[0], p.clone()) for p in xi])
    hessian_error = float((H - Href).abs().max())
    params = list(model.parameters())
    loss = strong_loss(model, xi, xb, depth, boundary_weight, problem)[0]
    g = torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss, params)])  # all parameters are used
    v = torch.randn(g.numel(), generator=gen, dtype=DTYPE); v = v/v.norm()
    tangent = float(g @ v); originals = [p.detach().clone() for p in params]
    estimates = []
    try:
        for eps in (1e-5, 1e-6):
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    offset = 0
                    for p, o in zip(params, originals):
                        p.copy_(o + sign*eps*v[offset:offset+p.numel()].reshape_as(p)); offset += p.numel()
                    values.append(float(strong_loss(model, xi, xb, depth, boundary_weight, problem)[0]))
            estimates.append(abs((values[0]-values[1])/(2*eps) - tangent))
    finally:
        with torch.no_grad():
            for p, o in zip(params, originals):
                p.copy_(o)
    scale = max(1.0, float(Href.abs().max()))
    ok = hessian_error <= 1e-9*scale and min(estimates) <= 1e-6 + 1e-4*abs(tangent)
    return {'input_hessian_max_error': hessian_error, 'directional_derivative': tangent,
            'finite_difference_errors': estimates, 'pass': bool(ok)}


def train(folder, problem, seed, pc, model):
    """Train `model` with the paper protocol; saves parameters, checkpoint and history."""
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'training.json').exists():
        print('Reusing trained model', folder, flush=True)
        return json.loads((folder/'training.json').read_text())
    torch.manual_seed(seed); torch.set_num_threads(1)
    depth, bw = pc['depth'], pc['boundary_weight']
    before = check_backward(model, depth, bw, problem)
    if not before['pass']:
        raise RuntimeError('Initial backward check failed: ' + json.dumps(before))
    gen = torch.Generator().manual_seed(10000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=pc['learning_rate'])
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(pc['adam_steps']*.65), int(pc['adam_steps']*.9)], gamma=.35)
    history = []; start = time.perf_counter()

    def record(phase, step, loss, pde, bc):
        history.append({'phase': phase, 'step': step, 'total': float(loss), 'pde_mse': float(pde),
                        'boundary_mse': float(bc), 'elapsed_seconds': time.perf_counter()-start})

    for step in range(1, pc['adam_steps']+1):
        xi = old.disk_points(pc['batch_interior'], gen); xb = old.boundary_points(pc['batch_boundary'], gen)
        opt.zero_grad(set_to_none=True)
        loss, pde, bc = strong_loss(model, xi, xb, depth, bw, problem)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at Adam step {step}')
        loss.backward(); opt.step(); sched.step()
        if step == 1 or step % pc['log_every'] == 0 or step == pc['adam_steps']:
            record('Adam', step, loss.detach(), pde.detach(), bc.detach())
            print(f'{folder.name} Adam {step}: PDE={float(pde):.3e} boundary={float(bc):.3e}', flush=True)
    xi = old.disk_points(pc['lbfgs_interior'], gen); xb = old.boundary_points(pc['lbfgs_boundary'])
    if pc['lbfgs_steps'] > 0:
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=pc['lbfgs_steps'],
                                  max_eval=2*pc['lbfgs_steps'], history_size=30, line_search_fn='strong_wolfe',
                                  tolerance_grad=1e-10, tolerance_change=1e-12)

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss, pde, bc = strong_loss(model, xi, xb, depth, bw, problem)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite L-BFGS loss')
            loss.backward()
            return loss
        lbfgs.step(closure)
    with torch.no_grad():
        loss, pde, bc = strong_loss(model, xi, xb, depth, bw, problem)
    record('final_fixed_training_set', 0, loss, pde, bc)
    seconds = time.perf_counter() - start
    after = check_backward(model, depth, bw, problem)
    if getattr(model, 'is_explicit', False):
        arrays = model.export()
    else:
        with torch.no_grad():
            arrays = {k: v.detach().numpy() for k, v in zip(('W', 'A', 'b', 'w', 'a', 'c'), model.constrained())}
    np.savez(folder/'trained_parameters.npz', **arrays)
    torch.save({'problem': problem, 'config': pc, 'state_dict': model.state_dict()}, folder/'model_state.pt')
    csv_write(folder/'training_history.csv', history)
    meta = {'problem': problem, 'seed': seed, 'config': pc, 'seconds': seconds,
            'trainable_parameters': count_parameters(model), 'architecture': type(model).__name__,
            'backward_before': before, 'backward_after': after,
            'parameters_sha256': sha(folder/'trained_parameters.npz')}
    write_json(folder/'training.json', meta)
    return meta


def depth_for_gap(q, tolerance):
    """Number of unrolled iterations so that q^L / (1-q) <= tolerance."""
    return int(math.ceil(math.log(tolerance*(1-q))/math.log(q)))


def done(folder, stage):
    return (folder/f'{stage}_complete.json').exists()


def mark(folder, stage, **extra):
    write_json(folder/f'{stage}_complete.json', {'stage': stage, **extra})


def paper_cfg(cfg, pilot=None, **overrides):
    """Configuration dictionary in the format expected by run.evaluate/intervals/domain."""
    base = run.load_config('full')
    base.update({k: cfg[k] for k in ('depths', 'interval_precision_bits') if k in cfg})
    base['pilot'].update(cfg['pilot'] if pilot is None else pilot)
    base.update(overrides)
    base['profile'] = cfg['profile']
    return base


# ----------------------------------------------------------------------------------------------
# Equilibrium analysis helpers
# ----------------------------------------------------------------------------------------------
def curvature_analysis(net, x, reference):
    """Sampled Hessian norms vs the two branches of Theorem 4.2 and the cancellation S*gamma <= 1."""
    z = reference[0]
    s = z @ net.W.T + x @ net.A.T + net.b
    d = 1/(1 + np.exp(-s)); gamma = d*(1-d)
    B = np.eye(net.m)[None] - d[..., None]*net.W
    Sgamma = np.linalg.solve(B, gamma[..., None])[..., 0]
    H = net.output(x, *reference)[2]
    delta = 1 - net.q
    wnorm = float(np.abs(net.w).sum()); KA = float(np.linalg.norm(net.A, axis=1).max())
    return {'q_effective': net.q, 'delta': delta, 'w_norm_1': wnorm, 'K_A': KA,
            'hessian_norm_max_sample': float(np.linalg.norm(H, ord=2, axis=(1, 2)).max()),
            'C2_quadratic_branch': wnorm*KA**2/delta**2,
            'C2_cubic_branch': wnorm*KA**2/(4*delta**3),
            'C2_theorem': wnorm*KA**2*min(delta**-2, 1/(4*delta**3)),
            'S_gamma_max_sample': float(Sgamma.max())}


def evaluate_equilibrium(folder, problem, seed, cfg, max_iter=50000):
    """Float64 evaluation on the reserved test points; works for any q < 1."""
    net = Net(folder/'trained_parameters.npz')
    x, xb = run.evaluation_points(seed, paper_cfg(cfg))
    ref = net.reference(x, max_iter=max_iter); u, g, H = net.output(x, *ref)
    ub = net.output(xb, *net.reference(xb, max_iter=max_iter))[0]
    p = PROBLEMS[problem]; truth, f = p.u_np(x), p.f_np(x); gb = p.u_np(xb)
    dref = det(H)
    out = {'u_relative_l2_sample': float(np.linalg.norm(u-truth)/np.linalg.norm(truth)),
           'u_max_abs_sample': float(abs(u-truth).max()),
           'pde_residual_max_abs_sample': float(abs(dref-f).max()),
           'pde_residual_rmse': float(np.sqrt(np.mean((dref-f)**2))),
           'boundary_max_abs_sample': float(abs(ub-gb).max()),
           'min_hessian_eigenvalue_sample': float(np.linalg.eigvalsh(H).min())}
    out.update(curvature_analysis(net, x, ref))
    return net, x, ref, out


def evaluate_explicit(folder, problem, seed, cfg):
    """Float64 evaluation of an explicit ICNN (exact up to rounding: no fixed point to solve)."""
    with np.load(folder/'trained_parameters.npz') as f:
        P = {k: f[k] for k in f.files}
    layers = sum(1 for k in P if k.startswith('A'))
    x, xb = run.evaluation_points(seed, paper_cfg(cfg))

    def jets(x):
        n, m = len(x), len(P['w'])
        z = np.zeros((n, m)); J = np.zeros((n, m, 2)); K = np.zeros((n, m, 2, 2))
        for k in range(layers):
            W = P[f'W{k}'] if k > 0 else np.zeros((m, m))
            s = z @ W.T + x @ P[f'A{k}'].T + P[f'b{k}']
            d = np.exp(-np.logaddexp(0, -s))
            V = np.einsum('ab,nbi->nai', W, J) + P[f'A{k}']
            K = d[..., None, None]*np.einsum('ab,nbij->naij', W, K) + (d*(1-d))[..., None, None]*V[..., :, None]*V[..., None, :]
            J = d[..., None]*V; z = np.logaddexp(0, s)
        return z @ P['w'] + x @ P['a'] + P['c'], np.einsum('m,nmij->nij', P['w'], K)

    u, H = jets(x); ub = jets(xb)[0]
    p = PROBLEMS[problem]; truth, f = p.u_np(x), p.f_np(x); gb = p.u_np(xb); dref = det(H)
    xt = x[:512]
    times = []
    for _ in range(5):
        t0 = time.perf_counter(); jets(xt); times.append(time.perf_counter()-t0)
    return {'u_relative_l2_sample': float(np.linalg.norm(u-truth)/np.linalg.norm(truth)),
            'u_max_abs_sample': float(abs(u-truth).max()),
            'pde_residual_max_abs_sample': float(abs(dref-f).max()),
            'pde_residual_rmse': float(np.sqrt(np.mean((dref-f)**2))),
            'boundary_max_abs_sample': float(abs(ub-gb).max()),
            'min_hessian_eigenvalue_sample': float(np.linalg.eigvalsh(H).min()),
            'eval_seconds_per_point_median': float(np.median(times))/len(xt)}


def timing_equilibrium(net, x, depth, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); jets = net.unroll(x, depth); net.output(x, *jets); times.append(time.perf_counter()-t0)
    return float(np.median(times))/len(x)


def geometric_checkpoints(max_depth):
    out, d = [], 4
    while d < max_depth:
        out.append(d); d = int(math.ceil(d*1.5))
    return out + [max_depth]


# ----------------------------------------------------------------------------------------------
# X1: near-critical q
# ----------------------------------------------------------------------------------------------
def x1_near_critical(root, cfg):
    e = cfg['X1']; out = root/'X1_near_critical'; rows = []
    for q in e['q_values']:
        L = depth_for_gap(q, e['unroll_tolerance'])
        for seed in e['seeds']:
            folder = out/f"{e['problem']}_q{q}_seed{seed}"
            pc = dict(cfg['pilot'], q=q, depth=L)
            train(folder, e['problem'], seed, pc, FixedRowSumPilot(pc['width'], q, seed))
            if not done(folder, 'x1'):
                net, x, ref, metrics = evaluate_equilibrium(folder, e['problem'], seed, cfg)
                # Certified evaluation cost: float64 adaptive stopping, then Arb on a small cohort.
                xs = x[:e['timing_points']]; checkpoints = geometric_checkpoints(e['max_depth_factor']*L)
                t0 = time.perf_counter()
                d, bound, depths, accepted = numerics.adaptive(net, xs, e['det_tolerance'], checkpoints)
                metrics.update({'unrolled_training_depth': L,
                                'adaptive_seconds_per_point': (time.perf_counter()-t0)/len(xs),
                                'adaptive_mean_depth': float(depths.mean()),
                                'adaptive_accepted_fraction': float(accepted.mean()),
                                'fixed_depth_seconds_per_point': timing_equilibrium(net, xs, int(depths.max()))})
                rig = RigorousNet(net, cfg['interval_precision_bits']); xa = x[:e['arb_points']]
                jets = net.unroll(xa, int(depths.max())); dh = det(net.output(xa, *jets)[2])
                t0 = time.perf_counter(); certs = []
                for k, pt in enumerate(xa):
                    certs.append(rig.certify(point_box(pt), tuple(v[k] for v in jets), e['problem'], reported_det=dh[k]))
                metrics.update({'arb_seconds_per_point': (time.perf_counter()-t0)/len(xa),
                                'arb_det_error_upper_max': max(c['det_error_upper'] for c in certs),
                                'arb_pde_residual_upper_max': max(c['pde_residual_upper'] for c in certs)})
                if e.get('domain_max_boxes', 0) > 0:
                    dc = paper_cfg(cfg, domain_max_depth=e['domain_max_depth'], domain_max_boxes=e['domain_max_boxes'],
                                   boundary_max_depth=e['boundary_max_depth'], boundary_max_arcs=e['boundary_max_arcs'])
                    run.domain(folder, e['problem'], seed, dc)
                    dom = json.loads((folder/'domain.json').read_text())
                    metrics.update({'domain_residual_upper': dom['pde_residual_uniform_upper'],
                                    'domain_boundary_upper': dom['boundary_error_uniform_upper'],
                                    'domain_seconds': dom['seconds_this_invocation'],
                                    'domain_boxes': dom['retained_leaves']})
                write_json(folder/'x1_metrics.json', metrics); mark(folder, 'x1')
            rows.append({'q': q, 'seed': seed, **json.loads((folder/'x1_metrics.json').read_text())})
    csv_write(out/'table_X1.csv', rows)
    if rows:
        qs = sorted(set(r['q'] for r in rows)); delta = np.array([1-q for q in qs])
        mean = lambda key: np.array([np.mean([r[key] for r in rows if r['q'] == q]) for q in qs])
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        axes[0].loglog(delta, mean('C2_cubic_branch'), 's--', label='cubic branch $\\|w\\|_1K_A^2/(4\\delta^3)$')
        axes[0].loglog(delta, mean('C2_quadratic_branch'), 'o-', label='quadratic branch $\\|w\\|_1K_A^2/\\delta^2$')
        axes[0].loglog(delta, mean('hessian_norm_max_sample'), '^:', label='observed max $\\|D^2u_\\theta\\|_2$')
        axes[0].set_title('Curvature bounds vs observed'); axes[0].set_xlabel('$\\delta = 1-q$'); axes[0].legend(fontsize=7)
        axes[1].loglog(delta, 100*mean('u_relative_l2_sample'), 'o-', label='relative $\\ell^2$ error of $u$ (%)')
        axes[1].loglog(delta, mean('pde_residual_max_abs_sample'), 's--', label='max sampled PDE residual')
        axes[1].set_title('Accuracy'); axes[1].set_xlabel('$\\delta$'); axes[1].legend(fontsize=7)
        axes[2].loglog(delta, mean('adaptive_mean_depth'), 'o-', label='mean adaptive depth')
        axes[2].loglog(delta, 1e3*mean('arb_seconds_per_point'), 's--', label='Arb ms / point')
        axes[2].set_title('Certified evaluation cost'); axes[2].set_xlabel('$\\delta$'); axes[2].legend(fontsize=7)
        for ax in axes:
            ax.invert_xaxis(); ax.grid(alpha=.3, which='both')
        fig.savefig(out/'X1_near_critical.pdf'); fig.savefig(out/'X1_near_critical.png', dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------------------------
# X3: harder problems (full paper pipeline on P3, P4)
# ----------------------------------------------------------------------------------------------
def x3_harder_problems(root, cfg):
    e = cfg['X3']; out = root/'X3_harder_problems'; rows = []
    pcfg = paper_cfg(cfg, interval_points=e['interval_points'], domain_max_depth=e['domain_max_depth'],
                     domain_max_boxes=e['domain_max_boxes'], boundary_max_depth=e['boundary_max_depth'],
                     boundary_max_arcs=e['boundary_max_arcs'], domain_residual_target=e['domain_residual_target'],
                     boundary_error_target=e['boundary_error_target'])
    for problem in e['problems']:
        for seed in e['seeds']:
            folder = out/f'{problem}_seed{seed}'
            pc = dict(cfg['pilot'])
            train(folder, problem, seed, pc, old.ConvexPilot(pc['width'], pc['q'], seed))
            for stage, fun in [('evaluate', run.evaluate), ('interval', run.intervals), ('domain', run.domain)]:
                if not done(folder, stage):
                    print('X3', folder.name, stage, flush=True)
                    fun(folder, problem, seed, pcfg); mark(folder, stage, parameters_sha256=sha(folder/'trained_parameters.npz'))
            ev = json.loads((folder/'evaluation.json').read_text()); dom = json.loads((folder/'domain.json').read_text())
            rows.append({'problem': problem, 'seed': seed, **{k: ev[k] for k in (
                'u_relative_l2_sample', 'u_max_abs_sample', 'pde_residual_max_abs_sample', 'boundary_max_abs_sample',
                'min_hessian_eigenvalue_sample', 'W_norm_infinity_float64')},
                'R_uniform': dom['pde_residual_uniform_upper'], 'B_uniform': dom['boundary_error_uniform_upper'],
                'U_direct_uniform': dom['u_error_uniform_upper_manufactured'],
                'barrier_upper': barrier(dom['pde_residual_uniform_upper'], dom['boundary_error_uniform_upper']),
                'boxes': dom['retained_leaves'], 'domain_seconds': dom['seconds_this_invocation']})
    csv_write(out/'table_X3.csv', rows)


def barrier(R, B):
    import flint
    flint.ctx.prec = 128
    return export_upper(exact(B) + exact(R).sqrt()/2)


# ----------------------------------------------------------------------------------------------
# X2: explicit ICNN comparison
# ----------------------------------------------------------------------------------------------
def equilibrium_folder(root, cfg, problem, seed):
    candidates = []
    if cfg.get('paper_run'):
        candidates.append(Path(cfg['paper_run'])/f'{problem}_seed{seed}')
    candidates.append(root/'X3_harder_problems'/f'{problem}_seed{seed}')
    candidates.append(root/'X2_explicit_icnn'/f'equilibrium_{problem}_seed{seed}')
    for c in candidates:
        if (c/'trained_parameters.npz').exists() and (c/'training.json').exists():
            return c
    return candidates[-1]


def x2_explicit_icnn(root, cfg):
    e = cfg['X2']; out = root/'X2_explicit_icnn'; rows = []
    for problem in e['problems']:
        for seed in e['seeds']:
            ef = equilibrium_folder(root, cfg, problem, seed)
            if not (ef/'training.json').exists():   # train the equilibrium baseline if no earlier run exists
                pc = dict(cfg['pilot']); train(ef, problem, seed, pc, old.ConvexPilot(pc['width'], pc['q'], seed))
            folder = out/f'icnn_{problem}_seed{seed}'
            pc = dict(cfg['pilot'])
            meta = train(folder, problem, seed, pc, ExplicitICNN(e['width'], e['layers'], seed))
            explicit = evaluate_explicit(folder, problem, seed, cfg)
            net, x, ref, eq = evaluate_equilibrium(ef, problem, seed, cfg)
            eq_meta = json.loads((ef/'training.json').read_text())
            xs = x[:512]
            rows.append({'problem': problem, 'seed': seed,
                         'equilibrium_parameters': eq_meta.get('trainable_parameters', 207),
                         'icnn_parameters': meta['trainable_parameters'],
                         'equilibrium_u_relative_l2': eq['u_relative_l2_sample'],
                         'icnn_u_relative_l2': explicit['u_relative_l2_sample'],
                         'equilibrium_pde_residual_max': eq['pde_residual_max_abs_sample'],
                         'icnn_pde_residual_max': explicit['pde_residual_max_abs_sample'],
                         'equilibrium_boundary_max': eq['boundary_max_abs_sample'],
                         'icnn_boundary_max': explicit['boundary_max_abs_sample'],
                         'equilibrium_eval_s_per_point_L32': timing_equilibrium(net, xs, 32),
                         'icnn_eval_s_per_point': explicit['eval_seconds_per_point_median'],
                         'equilibrium_training_seconds': eq_meta.get('seconds'),
                         'icnn_training_seconds': meta['seconds'],
                         'equilibrium_source': str(ef)})
    csv_write(out/'table_X2.csv', rows)
    if rows:
        labels = [f"{r['problem']}/{r['seed']}" for r in rows]; pos = np.arange(len(rows))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, (k1, k2, title) in zip(axes, [('equilibrium_u_relative_l2', 'icnn_u_relative_l2', 'Relative $\\ell^2$ error of $u$'),
                                             ('equilibrium_pde_residual_max', 'icnn_pde_residual_max', 'Max sampled PDE residual')]):
            ax.bar(pos-.18, [r[k1] for r in rows], .36, label='equilibrium (m=12)')
            ax.bar(pos+.18, [r[k2] for r in rows], .36, label='explicit ICNN (3x8)')
            ax.set_xticks(pos, labels, rotation=45); ax.set_yscale('log'); ax.set_title(title); ax.legend(fontsize=7)
        fig.savefig(out/'X2_explicit_icnn.pdf'); fig.savefig(out/'X2_explicit_icnn.png', dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------------------------
# X4: tightness of the uniform residual certificate under refinement
# ----------------------------------------------------------------------------------------------
def x4_tightness(root, cfg):
    e = cfg['X4']; out = root/'X4_tightness'; out.mkdir(parents=True, exist_ok=True); rows = []
    for problem, seed in [tuple(m) for m in e['models']]:
        source = equilibrium_folder(root, cfg, problem, seed)
        if not (source/'trained_parameters.npz').exists():
            print('X4: no trained model for', problem, seed, '- run the paper reproduction or X3 first', flush=True)
            continue
        tag = f'{problem}_seed{seed}'; marker = out/f'{tag}_complete.json'
        if marker.exists():
            rows.extend(json.loads(marker.read_text())['rows']); continue
        net = Net(source/'trained_parameters.npz'); rig = RigorousNet(net, cfg['interval_precision_bits'])
        start = time.perf_counter(); evaluated = 0

        def certify(ix, iy, level):
            nonlocal evaluated
            box = dyadic_box(ix, iy, level)
            centre = np.array([[float(v.mid()) for v in box]])
            jets = tuple(v[0] for v in net.reference(centre))
            evaluated += 1
            return rig.certify_box(box, jets, problem)['pde_residual_upper']

        level0 = e['initial_level']; n0 = 2**level0
        cells, discarded = {}, []
        for ix in range(n0):
            for iy in range(n0):
                if outside_disk(ix, iy, level0):
                    discarded.append({'ix': ix, 'iy': iy, 'level': level0})
                else:
                    cells[(ix, iy, level0)] = certify(ix, iy, level0)
        heap = [(-r, k) for k, r in cells.items()]; heapq.heapify(heap)
        checkpoints = sorted(e['box_checkpoints']); nxt = 0; curve = []

        def snapshot():
            R = max(cells.values())
            curve.append({'problem': problem, 'seed': seed, 'evaluated_boxes': evaluated, 'retained_boxes': len(cells),
                          'max_level': max(k[2] for k in cells), 'R_uniform': R,
                          'seconds': time.perf_counter()-start})
            print('X4', tag, curve[-1], flush=True)

        while heap and nxt < len(checkpoints):
            if evaluated >= checkpoints[nxt]:
                snapshot(); nxt += 1; continue
            negR, key = heapq.heappop(heap)
            if key not in cells or -negR != cells[key]:
                continue
            ix, iy, level = key
            if level >= e['max_level']:
                continue
            parent = cells.pop(key)
            for dx in (0, 1):
                for dy in (0, 1):
                    c = (2*ix+dx, 2*iy+dy, level+1)
                    if outside_disk(*c):
                        discarded.append({'ix': c[0], 'iy': c[1], 'level': c[2]}); continue
                    # A parent bound remains valid on each child; keep the tighter one.
                    cells[c] = min(certify(*c), parent); heapq.heappush(heap, (-cells[c], c))
        snapshot()
        arcs = [{'index': 0, 'level': 0}]
        check_cover([{'ix': k[0], 'iy': k[1], 'level': k[2]} for k in cells], discarded, arcs)
        R = curve[-1]['R_uniform']
        for row in curve:
            row['R_over_final'] = row['R_uniform']/R
        write_json(marker, {'rows': curve}); rows.extend(curve)
    csv_write(out/'table_X4.csv', rows)
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for key in sorted(set((r['problem'], r['seed']) for r in rows)):
            rr = [r for r in rows if (r['problem'], r['seed']) == key]
            axes[0].loglog([r['evaluated_boxes'] for r in rr], [r['R_uniform'] for r in rr], 'o-', label=f'{key[0]}/{key[1]}')
            axes[1].loglog([r['seconds'] for r in rr], [r['R_uniform'] for r in rr], 'o-', label=f'{key[0]}/{key[1]}')
        axes[0].set_xlabel('Certified boxes'); axes[1].set_xlabel('Seconds (CPU, one thread)')
        for ax in axes:
            ax.set_ylabel('Uniform residual bound $R$'); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
        fig.savefig(out/'X4_tightness.pdf'); fig.savefig(out/'X4_tightness.png', dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------------------------
# X5: finite-difference Newton baseline
# ----------------------------------------------------------------------------------------------
def fd_newton(problem, N, max_newton=30, max_bfo=5000, tol=1e-12):
    """Standard centred differences for det D^2u = f on the unit disk (BFO iteration + Newton polish).

    Unknowns are grid nodes strictly inside the disk. Stencil nodes outside the disk take the
    exact manufactured values (an exact extension of g), which favours this baseline.
    """
    import scipy.sparse as sp_
    import scipy.sparse.linalg as spla
    p = PROBLEMS[problem]; h = 2.0/N
    grid = np.linspace(-1, 1, N+1); X, Y = np.meshgrid(grid, grid, indexing='ij')
    inside = X**2 + Y**2 < 1 - 1e-14
    idx = -np.ones(X.shape, dtype=int); idx[inside] = np.arange(inside.sum())
    pts = np.stack([X[inside], Y[inside]], 1); n = len(pts)
    exact_all = p.u_np(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)
    f = p.f_np(pts); ii, jj = np.nonzero(inside)
    offsets = {'E': (1, 0), 'W': (-1, 0), 'N': (0, 1), 'S': (0, -1), 'NE': (1, 1), 'NW': (-1, 1), 'SE': (1, -1), 'SW': (-1, -1)}
    nb = {}
    for key, (di, dj) in offsets.items():
        ni, nj = ii+di, jj+dj
        if (ni < 0).any() or (ni > N).any() or (nj < 0).any() or (nj > N).any():
            raise RuntimeError('Stencil leaves the grid')
        nb[key] = (idx[ni, nj], exact_all[ni, nj])

    def values(u, key):
        k, ex = nb[key]
        return np.where(k >= 0, u[np.maximum(k, 0)], ex)

    def derivatives(u):
        uxx = (values(u, 'E') - 2*u + values(u, 'W'))/h**2
        uyy = (values(u, 'N') - 2*u + values(u, 'S'))/h**2
        uxy = (values(u, 'NE') - values(u, 'NW') - values(u, 'SE') + values(u, 'SW'))/(4*h**2)
        return uxx, uyy, uxy

    def jac(coeffs):
        rows, cols, data = [], [], []
        def add(key, c):
            k = nb[key][0]; m = k >= 0
            rows.append(np.nonzero(m)[0]); cols.append(k[m]); data.append(c[m])
        cxx, cyy, cxy = coeffs
        rows.append(np.arange(n)); cols.append(np.arange(n)); data.append(-2*cxx/h**2 - 2*cyy/h**2)
        for key in ('E', 'W'): add(key, cxx/h**2)
        for key in ('N', 'S'): add(key, cyy/h**2)
        for key, s in (('NE', 1), ('SW', 1), ('NW', -1), ('SE', -1)): add(key, s*cxy/(4*h**2))
        return sp_.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))

    t0 = time.perf_counter()
    # Stage 1: Benamou-Froese-Oberman iteration  Delta u_{k+1} = sqrt((uxx-uyy)^2 + 4 uxy^2 + 4 f).
    # Its fixed points satisfy det = f on the convex branch, so it selects the convex discrete solution
    # (plain Newton from a Poisson guess can converge to a spurious non-convex discrete solution).
    ones, zeros = np.ones(n), np.zeros(n)
    lap = spla.splu(jac((ones, ones, zeros)).tocsc())
    boundary = np.zeros(n)
    for key in ('E', 'W', 'N', 'S'):
        k, ex = nb[key]; boundary = boundary + np.where(k < 0, ex/h**2, 0)
    u = lap.solve(2*np.sqrt(np.maximum(f, 0)) - boundary)
    bfo_iterations = 0
    for bfo_iterations in range(1, max_bfo+1):
        uxx, uyy, uxy = derivatives(u)
        new = lap.solve(np.sqrt((uxx-uyy)**2 + 4*uxy**2 + 4*np.maximum(f, 0)) - boundary)
        change = np.abs(new-u).max(); u = new
        if change <= 1e-13:
            break
    # Stage 2: Newton polish, kept only if it stays on the convex branch and reduces the residual.
    def residual(v):
        a_, b_, c_ = derivatives(v); return np.abs(a_*b_ - c_**2 - f).max(), a_, b_, c_
    best, it = u.copy(), 0
    r_best = residual(u)[0]
    for it in range(1, max_newton+1):
        r, uxx, uyy, uxy = residual(u)
        if r <= tol:
            break
        cand = u - spla.spsolve(jac((uyy, uxx, -2*uxy)).tocsc(), uxx*uyy - uxy**2 - f)
        rc, cxx, cyy, _ = residual(cand)
        if not (rc < r and (cxx + cyy).min() >= 0):
            break
        u = cand
        if rc < r_best:
            best, r_best = cand.copy(), rc
    u = best
    seconds = time.perf_counter() - t0
    err = np.abs(u - p.u_np(pts))
    uxx, uyy, uxy = derivatives(u)
    return {'problem': problem, 'N': N, 'h': h, 'unknowns': n, 'bfo_iterations': bfo_iterations, 'newton_iterations': it,
            'final_discrete_residual': float(np.abs(uxx*uyy-uxy**2-f).max()),
            'u_max_abs_error_nodes': float(err.max()),
            'u_relative_l2_nodes': float(np.linalg.norm(err)/np.linalg.norm(p.u_np(pts))),
            'seconds': seconds}


def x5_fd_baseline(root, cfg):
    e = cfg['X5']; out = root/'X5_fd_baseline'; out.mkdir(parents=True, exist_ok=True); rows = []
    for problem in e['problems']:
        for N in e['grids']:
            try:
                rows.append(fd_newton(problem, N))
            except Exception as exc:   # report, never hide, a failed baseline run
                rows.append({'problem': problem, 'N': N, 'error': f'{type(exc).__name__}: {exc}'})
            print('X5', rows[-1], flush=True)
    csv_write(out/'table_X5.csv', rows)
    ok = [r for r in rows if 'error' not in r]
    if ok:
        fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
        for problem in e['problems']:
            rr = [r for r in ok if r['problem'] == problem]
            if rr:
                ax.loglog([r['h'] for r in rr], [r['u_max_abs_error_nodes'] for r in rr], 'o-', label=problem)
        ax.set_xlabel('$h$'); ax.set_ylabel('max nodal error'); ax.set_title('Finite-difference Newton baseline')
        ax.grid(alpha=.3, which='both'); ax.legend()
        fig.savefig(out/'X5_fd_baseline.pdf'); fig.savefig(out/'X5_fd_baseline.png', dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------------------------
# X6: certification cost vs width
# ----------------------------------------------------------------------------------------------
def x6_width_scaling(root, cfg):
    e = cfg['X6']; out = root/'X6_width_scaling'; rows = []
    for width in e['widths']:
        for seed in e['seeds']:
            folder = out/f"{e['problem']}_m{width}_seed{seed}"
            pc = dict(cfg['pilot'], width=width)
            meta = train(folder, e['problem'], seed, pc, old.ConvexPilot(width, pc['q'], seed))
            if not done(folder, 'x6'):
                net, x, ref, metrics = evaluate_equilibrium(folder, e['problem'], seed, cfg)
                xs = x[:e['timing_points']]
                t0 = time.perf_counter()
                for _ in range(3):
                    jets = net.unroll(xs, 32); net.certificate(xs, *jets)
                metrics['float64_seconds_per_point'] = (time.perf_counter()-t0)/(3*len(xs))
                rig = RigorousNet(net, cfg['interval_precision_bits']); xa = x[:e['arb_points']]
                jets = net.unroll(xa, 32); dh = det(net.output(xa, *jets)[2]); t0 = time.perf_counter()
                for k, pt in enumerate(xa):
                    rig.certify(point_box(pt), tuple(v[k] for v in jets), e['problem'], reported_det=dh[k])
                metrics['arb_point_seconds'] = (time.perf_counter()-t0)/len(xa)
                boxes = [dyadic_box(i, j, e['box_level']) for i, j in e['box_indices']]; t0 = time.perf_counter()
                for box in boxes:
                    centre = np.array([[float(v.mid()) for v in box]])
                    rig.certify_box(box, tuple(v[0] for v in net.reference(centre)), e['problem'])
                metrics['arb_box_seconds'] = (time.perf_counter()-t0)/len(boxes)
                metrics.update({'width': width, 'trainable_parameters': meta['trainable_parameters'],
                                'training_seconds': meta['seconds']})
                write_json(folder/'x6_metrics.json', metrics); mark(folder, 'x6')
            rows.append({'seed': seed, **json.loads((folder/'x6_metrics.json').read_text())})
    csv_write(out/'table_X6.csv', rows)
    if rows:
        ws = sorted(set(r['width'] for r in rows))
        mean = lambda k: [np.mean([r[k] for r in rows if r['width'] == w]) for w in ws]
        fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
        for key, label in [('float64_seconds_per_point', 'float64 L=32 + estimator'),
                           ('arb_point_seconds', 'Arb point certificate'), ('arb_box_seconds', 'Arb box certificate')]:
            ax.loglog(ws, mean(key), 'o-', label=label)
        ax.set_xlabel('width $m$'); ax.set_ylabel('seconds per point / box'); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
        fig.savefig(out/'X6_width_scaling.pdf'); fig.savefig(out/'X6_width_scaling.png', dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------------------------
EXPERIMENTS = {'X1': x1_near_critical, 'X3': x3_harder_problems, 'X2': x2_explicit_icnn,
               'X4': x4_tightness, 'X5': x5_fd_baseline, 'X6': x6_width_scaling}


def load_config(profile, override=None):
    cfg = json.loads((HERE/'config_extended.json').read_text())
    if profile == 'smoke':
        smoke = cfg.pop('smoke')
        for key, value in smoke.items():
            if isinstance(value, dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    else:
        cfg.pop('smoke')
    if override:
        custom = json.loads(Path(override).read_text())
        for key, value in custom.items():
            if isinstance(value, dict) and key in cfg:
                cfg[key].update(value)
            else:
                cfg[key] = value
    cfg['profile'] = profile
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', choices=['smoke', 'full'], default='full')
    parser.add_argument('--out', default='results_extended')
    parser.add_argument('--paper-run', help='Folder of a completed run.py output, reused for P1/P2 models')
    parser.add_argument('--experiments', nargs='+', default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument('--config', help='JSON overrides')
    args = parser.parse_args()
    cfg = load_config(args.profile, args.config)
    if args.paper_run:
        cfg['paper_run'] = str(Path(args.paper_run).resolve())
    root = Path(args.out).resolve(); root.mkdir(parents=True, exist_ok=True)
    protocol = root/'protocol_extended.json'
    if protocol.exists() and json.loads(protocol.read_text()) != cfg:
        raise RuntimeError('Configuration differs from the existing output; choose a new --out folder.')
    write_json(protocol, cfg); write_json(root/'environment.json', run.environment())
    write_json(root/'problem_checks.json', verify_problems())
    torch.set_num_threads(1)
    status = {}
    for name in args.experiments:
        print(f'===== {name} =====', flush=True); t0 = time.perf_counter()
        EXPERIMENTS[name](root, cfg)
        status[name] = {'completed': True, 'seconds_this_invocation': time.perf_counter()-t0}
        write_json(root/'STATUS_extended.json', status)
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()

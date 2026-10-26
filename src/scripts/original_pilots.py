"""Original pilot definitions from cells 4,6,8,10,20,21 (zero-based) of the E3 notebook.

Code is unchanged; only comments and messages were translated into English."""

# Original cell 4
from pathlib import Path
from dataclasses import dataclass, asdict
import csv
import json
import platform
import time
import shutil
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)  # Small matrices: avoid multithreading overhead.
DTYPE = torch.float64


@dataclass
class PilotConfig:
    seed: int = 0
    width: int = 12
    q: float = 0.7
    depth: int = 32
    adam_steps: int = 1200
    learning_rate: float = 0.02
    batch_interior: int = 128
    batch_boundary: int = 64
    boundary_weight: float = 10.0
    lbfgs_steps: int = 100
    lbfgs_interior: int = 512
    lbfgs_boundary: int = 128
    test_interior: int = 2048
    test_boundary: int = 512
    log_every: int = 100
    state_atol: float = 1e-12


def sp(t):
    return torch.logaddexp(torch.zeros_like(t), t)


class ConvexPilot(nn.Module):
    def __init__(self, width=12, q=0.7, seed=0):
        super().__init__()
        if width < 2 or not 0 < q < 1:
            raise ValueError("width >= 2 and 0 < q < 1 are required.")
        self.width, self.q = width, q
        gen = torch.Generator().manual_seed(seed)
        self.raw_W = nn.Parameter(0.1 * torch.randn(width, width, generator=gen, dtype=DTYPE))
        self.row_gate = nn.Parameter(torch.zeros(width, dtype=DTYPE))
        # Random initialization; the known interior solution is not used.
        self.A = nn.Parameter(1.5 * torch.randn(width, 2, generator=gen, dtype=DTYPE))
        self.b = nn.Parameter(torch.zeros(width, dtype=DTYPE))
        self.raw_w = nn.Parameter(torch.full((width,), float(np.log(np.expm1(1.0))), dtype=DTYPE))
        self.a = nn.Parameter(torch.zeros(2, dtype=DTYPE))
        self.c = nn.Parameter(torch.zeros((), dtype=DTYPE))

    def constrained(self):
        # Row sum i = q*sigmoid(row_gate_i) < q.
        W = self.q * torch.sigmoid(self.row_gate)[:, None] * torch.softmax(self.raw_W, dim=1)
        w = sp(self.raw_w) / self.width
        return W, self.A, self.b, w, self.a, self.c

    def value_unrolled(self, x, depth):
        W, A, b, w, a, c = self.constrained()
        z = x.new_zeros((len(x), self.width))
        for _ in range(depth):
            z = sp(z @ W.T + x @ A.T + b)
        return z @ w + x @ a + c

    def jets(self, x, depth):
        """EXACT value, gradient and Hessian of the truncated network (up to rounding).

        The chain rule is propagated through the iterations. There is no detach or
        no_grad: autograd can differentiate the resulting loss with respect to theta.
        """
        W, A, b, w, a, c = self.constrained()
        n, m = len(x), self.width
        z = x.new_zeros(n, m)
        J = x.new_zeros(n, m, 2)
        K = x.new_zeros(n, m, 2, 2)
        for _ in range(depth):
            s = z @ W.T + x @ A.T + b
            d = torch.sigmoid(s)
            V = torch.einsum("ab,nbi->nai", W, J) + A[None]
            WK = torch.einsum("ab,nbij->naij", W, K)
            K = d[:, :, None, None] * WK + (d*(1-d))[:, :, None, None] * V[:, :, :, None] * V[:, :, None, :]
            J = d[:, :, None] * V
            z = sp(s)
        u = z @ w + x @ a + c
        gradient = torch.einsum("m,nmi->ni", w, J) + a
        hessian = torch.einsum("m,nmij->nij", w, K)
        return u, gradient, hessian, z


# Original cell 6
def disk_points(n, gen):
    r = torch.sqrt(torch.rand(n, generator=gen, dtype=DTYPE))
    t = 2*np.pi*torch.rand(n, generator=gen, dtype=DTYPE)
    return torch.stack((r*torch.cos(t), r*torch.sin(t)), dim=1)


def boundary_points(n, gen=None):
    t = (2*np.pi*torch.rand(n, generator=gen, dtype=DTYPE) if gen is not None
         else 2*np.pi*torch.arange(n, dtype=DTYPE)/n)
    return torch.stack((torch.cos(t), torch.sin(t)), dim=1)


def boundary_g(x):
    return x[:, 0]**2 + 0.25*x[:, 1]**2


def exact_u_for_evaluation(x):
    # Called only in the final evaluation, never in the interior loss.
    return x[:, 0]**2 + 0.25*x[:, 1]**2


def determinant(H):
    return H[:, 0, 0]*H[:, 1, 1] - H[:, 0, 1]*H[:, 1, 0]


def strong_loss(model, interior, boundary, depth, boundary_weight):
    _, _, H, _ = model.jets(interior, depth)
    loss_pde = ((determinant(H)-1.0)**2).mean()
    loss_bc = ((model.value_unrolled(boundary, depth)-boundary_g(boundary))**2).mean()
    return loss_pde + boundary_weight*loss_bc, loss_pde, loss_bc


def validate_backward(model, depth, boundary_weight=10.0):
    """Check input derivatives and the parameter backward pass against independent routes.

    Small checks: 2 interior points, 5 boundary points, 3 directions
    in raw parameters and several central finite-difference steps.
    """
    gen = torch.Generator().manual_seed(8123)
    xi, xb = disk_points(2, gen), boundary_points(5, gen)
    u, g, H, _ = model.jets(xi, depth)
    ref_u, ref_g, ref_H = [], [], []
    for point in xi:
        point = point.clone().requires_grad_(True)
        fun = lambda t: model.value_unrolled(t[None], depth)[0]
        value = fun(point)
        ref_u.append(value)
        ref_g.append(torch.autograd.grad(value, point, create_graph=True)[0])
        ref_H.append(torch.autograd.functional.hessian(fun, point, create_graph=True))
    ur, gr, Hr = torch.stack(ref_u), torch.stack(ref_g), torch.stack(ref_H)
    input_errors = {key: float((a-b).abs().max().detach()) for key, a, b in
                    zip(("u", "gradient", "hessian"), (u,g,H), (ur,gr,Hr))}
    loss, _, _ = strong_loss(model, xi, xb, depth, boundary_weight)
    reference_loss = ((determinant(Hr)-1)**2).mean() + boundary_weight*((model.value_unrolled(xb,depth)-boundary_g(xb))**2).mean()
    params = list(model.parameters())
    gradient = torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss, params)])
    reference_gradient = torch.cat([v.reshape(-1) for v in torch.autograd.grad(reference_loss, params)])
    grad_diff = float((gradient-reference_gradient).abs().max())
    # rtol+atol, without dividing by a near-zero gradient.
    gradient_ok = bool(torch.allclose(gradient, reference_gradient, atol=1e-8, rtol=1e-6))
    originals = [p.detach().clone() for p in params]
    directions = []
    try:
        for direction_index in range(3):
            v = torch.randn(gradient.numel(), generator=gen, dtype=DTYPE)
            v = v / v.norm()
            tangent = float(torch.dot(gradient, v))
            estimates = []
            for eps in (1e-4, 1e-5, 1e-6):
                losses = []
                for sign in (1, -1):
                    offset = 0
                    with torch.no_grad():
                        for param, original in zip(params, originals):
                            block = v[offset:offset+param.numel()].reshape_as(param)
                            param.copy_(original + sign*eps*block)
                            offset += param.numel()
                        losses.append(float(strong_loss(model,xi,xb,depth,boundary_weight)[0]))
                estimate = (losses[0]-losses[1])/(2*eps)
                error = abs(estimate-tangent)
                estimates.append({"epsilon":eps,"finite_difference":estimate,"abs_error":error,
                                  "pass":error <= 1e-6 + 1e-4*abs(tangent)})
            directions.append({"direction":direction_index,"autograd":tangent,"estimates":estimates,
                               "pass":sum(r["pass"] for r in estimates)>=2})
    finally:
        with torch.no_grad():
            for param, original in zip(params, originals):
                param.copy_(original)
    passed = max(input_errors.values()) < 1e-9 and gradient_ok and all(r["pass"] for r in directions)
    return {"input_derivative_max_errors":input_errors,"parameter_gradient_max_error":grad_diff,
            "parameter_gradient_reference_pass":gradient_ok,"directional_finite_differences":directions,
            "pass":passed}


# Original cell 8
def train_pilot(cfg):
    model = ConvexPilot(cfg.width, cfg.q, cfg.seed)
    validation_before = validate_backward(model, cfg.depth, cfg.boundary_weight)
    if not validation_before["pass"]:
        raise RuntimeError("Initial backward validation failed: " + json.dumps(validation_before))
    print("Initial backward: PASS", flush=True)
    gen = torch.Generator().manual_seed(10000+cfg.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(cfg.adam_steps*.65),int(cfg.adam_steps*.9)],gamma=.35)
    history = []
    start = time.perf_counter()
    model.train()
    for step in range(1, cfg.adam_steps+1):
        xi = disk_points(cfg.batch_interior,gen)
        xb = boundary_points(cfg.batch_boundary,gen)
        optimizer.zero_grad(set_to_none=True)
        loss, pde, bc = strong_loss(model,xi,xb,cfg.depth,cfg.boundary_weight)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at Adam step {step}.")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"Non-finite gradient at Adam step {step}.")
        optimizer.step()
        scheduler.step()
        if step == 1 or step % cfg.log_every == 0 or step == cfg.adam_steps:
            row={"phase":"Adam","step":step,"total":float(loss.detach()),"pde_mse":float(pde.detach()),
                 "boundary_mse":float(bc.detach()),"elapsed_seconds":time.perf_counter()-start}
            history.append(row)
            print(f"Adam {step:4d} | PDE MSE={row['pde_mse']:.3e} | boundary MSE={row['boundary_mse']:.3e}",flush=True)
    # Fresh data, kept fixed only so that the L-BFGS line search evaluates the same function.
    xi = disk_points(cfg.lbfgs_interior,gen)
    xb = boundary_points(cfg.lbfgs_boundary)
    if cfg.lbfgs_steps > 0:
        opt_lbfgs = torch.optim.LBFGS(model.parameters(),lr=1,max_iter=cfg.lbfgs_steps,
            max_eval=cfg.lbfgs_steps*2,history_size=30,line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,tolerance_change=1e-12)
        closure_calls = 0
        def closure():
            nonlocal closure_calls
            opt_lbfgs.zero_grad(set_to_none=True)
            loss,pde,bc=strong_loss(model,xi,xb,cfg.depth,cfg.boundary_weight)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite loss in L-BFGS.")
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Non-finite gradient in L-BFGS.")
            closure_calls += 1
            if closure_calls == 1 or closure_calls % 20 == 0:
                row={"phase":"LBFGS_closure","step":closure_calls,"total":float(loss.detach()),
                     "pde_mse":float(pde.detach()),"boundary_mse":float(bc.detach()),"elapsed_seconds":time.perf_counter()-start}
                history.append(row)
                print(f"L-BFGS evaluation {closure_calls:3d} | PDE MSE={row['pde_mse']:.3e} | boundary MSE={row['boundary_mse']:.3e}",flush=True)
            return loss
        opt_lbfgs.step(closure)
    with torch.no_grad():
        loss,pde,bc=strong_loss(model,xi,xb,cfg.depth,cfg.boundary_weight)
    history.append({"phase":"final_fixed_training_set","step":0,"total":float(loss),"pde_mse":float(pde),
                    "boundary_mse":float(bc),"elapsed_seconds":time.perf_counter()-start})
    model.eval()
    return model, history, validation_before, time.perf_counter()-start


# Original cell 10
@torch.no_grad()
def equilibrium_evaluation(model,x,atol=1e-12,max_iter=2000):
    W,A,b,w,a,c=model.constrained()
    z=x.new_zeros(len(x),model.width)
    for L in range(max_iter+1):
        next_z=sp(z@W.T+x@A.T+b)
        tau=float((next_z-z).abs().max())
        if tau/(1-model.q)<=atol:
            break
        z=next_z
    else:
        raise RuntimeError("The equilibrium evaluation did not reach the tolerance.")
    s=z@W.T+x@A.T+b
    d=torch.sigmoid(s)
    B=torch.eye(model.width,dtype=DTYPE)[None]-d[:,:,None]*W[None]
    rhs_J=d[:,:,None]*A[None]
    J=torch.linalg.solve(B,rhs_J)
    V=torch.einsum("ab,nbi->nai",W,J)+A[None]
    rhs_K=(d*(1-d))[:,:,None,None]*V[:,:,:,None]*V[:,:,None,:]
    K=torch.linalg.solve(B,rhs_K.reshape(len(x),model.width,4)).reshape(len(x),model.width,2,2)
    return (z@w+x@a+c,torch.einsum("m,nmi->ni",w,J)+a,torch.einsum("m,nmij->nij",w,K),
            {"iterations":L,"max_state_residual":tau,"state_bound":tau/(1-model.q),
             "output_bound":float(w.abs().sum())*tau/(1-model.q),
             "linear_J_residual":float((B@J-rhs_J).abs().max()),
             "linear_K_residual":float((B@K.reshape(len(x),model.width,4)-rhs_K.reshape(len(x),model.width,4)).abs().max())})


@torch.no_grad()
def evaluate_pilot(model,cfg):
    # Reserved seeds; these samples are not used by Adam or L-BFGS.
    gen=torch.Generator().manual_seed(900000+cfg.seed)
    x=disk_points(cfg.test_interior,gen)
    # Angular offset avoids reusing the L-BFGS boundary grid.
    t=2*np.pi*(torch.arange(cfg.test_boundary,dtype=DTYPE)+0.371)/cfg.test_boundary
    xb=torch.stack((torch.cos(t),torch.sin(t)),dim=1)
    values,grads,hessians=[],[],[]
    infos=[]
    for block in x.split(256):
        u,g,H,info=equilibrium_evaluation(model,block,cfg.state_atol)
        values.append(u);grads.append(g);hessians.append(H);infos.append(info)
    u,g,H=torch.cat(values),torch.cat(grads),torch.cat(hessians)
    ub,_,_,boundary_info=equilibrium_evaluation(model,xb,cfg.state_atol)
    target=exact_u_for_evaluation(x)
    target_gradient=torch.stack((2*x[:,0],0.5*x[:,1]),dim=1)
    det=determinant(H)
    error=u-target
    bc_error=ub-boundary_g(xb)
    W,A,b,w,a,c=model.constrained()
    metrics={
        "u_rmse":float(torch.sqrt((error**2).mean())),
        "u_relative_l2_sample":float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(target)),
        "u_max_abs_sample":float(error.abs().max()),
        "gradient_rmse_euclidean":float(torch.sqrt(((g-target_gradient)**2).sum(1).mean())),
        "pde_residual_rmse":float(torch.sqrt(((det-1)**2).mean())),
        "pde_residual_max_abs_sample":float((det-1).abs().max()),
        "boundary_rmse":float(torch.sqrt((bc_error**2).mean())),
        "boundary_max_abs_sample":float(bc_error.abs().max()),
        "min_hessian_eigenvalue_sample":float(torch.linalg.eigvalsh((H+H.transpose(1,2))/2).min()),
        "W_norm_infinity":float(W.abs().sum(1).max()),
        "max_equilibrium_iterations":max(v["iterations"] for v in infos+[boundary_info]),
        "max_equilibrium_state_bound":max(v["state_bound"] for v in infos+[boundary_info]),
        "max_equilibrium_output_bound":max(v["output_bound"] for v in infos+[boundary_info]),
        "max_linear_J_residual":max(v["linear_J_residual"] for v in infos+[boundary_info]),
        "max_linear_K_residual":max(v["linear_K_residual"] for v in infos+[boundary_info]),
    }
    # Derivatives and determinant are compared directly; their errors are not inferred from the state residual.
    truncation=[]
    for depth in (cfg.depth,2*cfg.depth):
        ul,gl,Hl,zl=model.jets(x[:128],depth)
        next_z=sp(zl@W.T+x[:128]@A.T+b)
        truncation.append({"depth":depth,"u_max_difference_to_equilibrium":float((ul-u[:128]).abs().max()),
            "gradient_max_difference_to_equilibrium":float((gl-g[:128]).abs().max()),
            "hessian_max_difference_to_equilibrium":float((Hl-H[:128]).abs().max()),
            "det_max_difference_to_equilibrium":float((determinant(Hl)-det[:128]).abs().max()),
            "state_residual_bound":float((next_z-zl).abs().max())/(1-model.q)})
    data={"x":x.numpy(),"u":u.numpy(),"u_exact":target.numpy(),"gradient":g.numpy(),
          "hessian":H.numpy(),"det_hessian":det.numpy(),"boundary_x":xb.numpy(),
          "boundary_u":ub.numpy(),"boundary_g":boundary_g(xb).numpy()}
    return metrics,truncation,data


def depth_gradient_check(model,cfg):
    gen=torch.Generator().manual_seed(71000+cfg.seed)
    xi,xb=disk_points(8,gen),boundary_points(8,gen)
    vectors=[]
    losses=[]
    for depth in (cfg.depth,2*cfg.depth):
        loss,_,_=strong_loss(model,xi,xb,depth,cfg.boundary_weight)
        vectors.append(torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss,tuple(model.parameters()))]))
        losses.append(float(loss.detach()))
    return {"loss_L":losses[0],"loss_2L":losses[1],
            "gradient_max_abs_difference":float((vectors[0]-vectors[1]).abs().max()),
            "gradient_relative_l2_difference":float((vectors[0]-vectors[1]).norm()/vectors[1].norm().clamp_min(1e-14)),
            "note":"Numerical stability when doubling L; not a general error bound for the implicit backward pass."}


# Original cell 20
from pathlib import Path
from dataclasses import asdict, replace
from datetime import datetime
import json, csv, shutil, time, platform
import numpy as np
import torch
import matplotlib.pyplot as plt

def p2_u(x):
    return 0.5*(x**2).sum(dim=1) + 0.1*(x**4).sum(dim=1)

def p2_f(x):
    return (1 + 1.2*x[:, 0]**2)*(1 + 1.2*x[:, 1]**2)

def p2_loss(model, xi, xb, cfg):
    _, _, H, _ = model.jets(xi, cfg.depth)

    loss_pde = ((determinant(H) - p2_f(xi))**2).mean()

    # During training, p2_u is used only on the boundary.
    loss_bc = (
        (model.value_unrolled(xb, cfg.depth) - p2_u(xb))**2
    ).mean()

    return loss_pde + cfg.boundary_weight*loss_bc, loss_pde, loss_bc


# Check that the manufactured data correspond to the stated solution.
check_x = torch.tensor(
    [[0., 0.], [.3, .4], [-.7, .2]],
    dtype=DTYPE,
)

check_H = torch.stack([
    torch.autograd.functional.hessian(
        lambda t: p2_u(t[None])[0], x
    )
    for x in check_x
])

assert torch.allclose(
    determinant(check_H),
    p2_f(check_x),
    atol=1e-12,
    rtol=1e-12,
)

print("Problem 2 data: verified.")

# Original cell 21
def p2_check_backward(model, cfg):
    gen = torch.Generator().manual_seed(8123)
    xi = disk_points(2, gen)
    xb = boundary_points(5, gen)

    _, _, H, _ = model.jets(xi, cfg.depth)

    # Reference: Hessian computed directly from the output.
    H_reference = torch.stack([
        torch.autograd.functional.hessian(
            lambda t: model.value_unrolled(t[None], cfg.depth)[0],
            x.clone().requires_grad_(True),
            create_graph=True,
        )
        for x in xi
    ])

    loss = p2_loss(model, xi, xb, cfg)[0]

    reference_loss = (
        (determinant(H_reference) - p2_f(xi))**2
    ).mean()

    reference_loss = reference_loss + cfg.boundary_weight * (
        (model.value_unrolled(xb, cfg.depth) - p2_u(xb))**2
    ).mean()

    params = tuple(model.parameters())

    def flatten(values):
        return torch.cat([v.reshape(-1) for v in values])

    gradient = flatten(torch.autograd.grad(loss, params))
    reference_gradient = flatten(
        torch.autograd.grad(reference_loss, params)
    )

    # Additional check by finite differences.
    direction = torch.randn(
        gradient.numel(), generator=gen, dtype=DTYPE
    )
    direction = direction / direction.norm()
    predicted = float(gradient @ direction)

    originals = [p.detach().clone() for p in params]
    finite_differences = []

    try:
        for epsilon in (1e-4, 1e-5, 1e-6):
            values = []

            for sign in (1, -1):
                offset = 0

                with torch.no_grad():
                    for parameter, original in zip(params, originals):
                        block = direction[
                            offset:offset + parameter.numel()
                        ].reshape_as(parameter)

                        parameter.copy_(
                            original + sign*epsilon*block
                        )
                        offset += parameter.numel()

                    values.append(
                        float(p2_loss(model, xi, xb, cfg)[0])
                    )

            estimate = (values[0] - values[1]) / (2*epsilon)
            error = abs(estimate - predicted)

            finite_differences.append({
                "epsilon": epsilon,
                "finite_difference": estimate,
                "autograd": predicted,
                "abs_error": error,
                "pass": error <= 1e-6 + 1e-4*abs(predicted),
            })
    finally:
        with torch.no_grad():
            for parameter, original in zip(params, originals):
                parameter.copy_(original)

    result = {
        "hessian_max_difference": float(
            (H - H_reference).abs().max().detach()
        ),
        "parameter_gradient_max_difference": float(
            (gradient - reference_gradient).abs().max()
        ),
        "finite_differences": finite_differences,
    }

    result["pass"] = bool(
        torch.allclose(H, H_reference, atol=1e-9, rtol=1e-7)
        and torch.allclose(
            gradient, reference_gradient, atol=1e-8, rtol=1e-6
        )
        and sum(r["pass"] for r in finite_differences) >= 2
    )

    return result


def p2_train(cfg):
    model = ConvexPilot(cfg.width, cfg.q, cfg.seed)

    before = p2_check_backward(model, cfg)

    if not before["pass"]:
        raise RuntimeError(
            "Initial backward failed:\n" + json.dumps(before, indent=2)
        )

    print("Initial backward: PASS", flush=True)

    gen = torch.Generator().manual_seed(10000 + cfg.seed)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(cfg.adam_steps*0.65),
            int(cfg.adam_steps*0.9),
        ],
        gamma=0.35,
    )

    history = []
    start = time.perf_counter()

    def record(phase, step, loss, loss_pde, loss_bc):
        row = {
            "phase": phase,
            "step": step,
            "total": float(loss.detach()),
            "pde_mse": float(loss_pde.detach()),
            "boundary_mse": float(loss_bc.detach()),
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(row)

        print(
            f"{phase} {step}: "
            f"PDE={row['pde_mse']:.3e}, "
            f"boundary={row['boundary_mse']:.3e}",
            flush=True,
        )

    def backward_checked(loss):
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss.")

        loss.backward()

        if not all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise RuntimeError("Non-finite gradient.")

    # Adam: fresh points at each update.
    for step in range(1, cfg.adam_steps + 1):
        xi = disk_points(cfg.batch_interior, gen)
        xb = boundary_points(cfg.batch_boundary, gen)

        optimizer.zero_grad(set_to_none=True)

        loss, loss_pde, loss_bc = p2_loss(model, xi, xb, cfg)

        backward_checked(loss)
        optimizer.step()
        scheduler.step()

        if (
            step == 1
            or step % cfg.log_every == 0
            or step == cfg.adam_steps
        ):
            record("Adam", step, loss, loss_pde, loss_bc)

    # L-BFGS: fixed points during the line search.
    xi = disk_points(cfg.lbfgs_interior, gen)
    xb = boundary_points(cfg.lbfgs_boundary)

    if cfg.lbfgs_steps > 0:
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1,
            max_iter=cfg.lbfgs_steps,
            max_eval=2*cfg.lbfgs_steps,
            history_size=30,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
        )

        calls = 0

        def closure():
            nonlocal calls

            optimizer.zero_grad(set_to_none=True)
            loss, loss_pde, loss_bc = p2_loss(model, xi, xb, cfg)

            backward_checked(loss)
            calls += 1

            if calls == 1 or calls % 20 == 0:
                record(
                    "LBFGS_closure", calls, loss, loss_pde, loss_bc
                )

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        loss, loss_pde, loss_bc = p2_loss(model, xi, xb, cfg)

    record("final", 0, loss, loss_pde, loss_bc)
    seconds = time.perf_counter() - start

    after = p2_check_backward(model, cfg)

    return model, history, before, after, seconds

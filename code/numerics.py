"""Float64 evaluation. These diagnostics do NOT enclose roundoff."""
import numpy as np


def target(x, problem):
    if problem == 'P1':
        return x[:, 0]**2 + x[:, 1]**2/4, np.ones(len(x))
    return .5*(x*x).sum(1)+.1*(x**4).sum(1), (1+1.2*x[:, 0]**2)*(1+1.2*x[:, 1]**2)


class Net:
    def __init__(self, path=None, parameters=None):
        if parameters is None:
            with np.load(path, allow_pickle=False) as f:
                parameters = {k: f[k] for k in ('W', 'A', 'b', 'w', 'a', 'c')}
        for k, v in parameters.items():
            setattr(self, k, np.asarray(v, dtype=np.float64))
        self.m = len(self.w)
        assert self.W.shape == (self.m, self.m) and self.A.shape == (self.m, 2)
        assert self.b.shape == self.w.shape == (self.m,) and self.a.shape == (2,)
        assert all(np.isfinite(v).all() for v in parameters.values())
        assert np.min(self.W) >= 0 and np.min(self.w) >= 0
        self.q = float(self.W.sum(1).max())
        assert 0 <= self.q < 1

    def zero(self, n):
        return (np.zeros((n, self.m)), np.zeros((n, self.m, 2)),
                np.zeros((n, self.m, 2, 2)))

    def step(self, x, z, J, K):
        s = z@self.W.T + x@self.A.T + self.b
        d = np.exp(-np.logaddexp(0, -s))
        V = np.einsum('ab,nbi->nai', self.W, J)+self.A
        Kn = d[..., None, None]*np.einsum('ab,nbij->naij', self.W, K)
        Kn += (d*(1-d))[..., None, None]*V[..., :, None]*V[..., None, :]
        Kn = (Kn+Kn.swapaxes(-1, -2))/2
        return np.logaddexp(0, s), d[..., None]*V, Kn

    def unroll(self, x, depth):
        jets = self.zero(len(x))
        for _ in range(depth):
            jets = self.step(x, *jets)
        return jets

    def output(self, x, z, J, K):
        return (z@self.w+x@self.a+self.c,
                np.einsum('m,nmi->ni', self.w, J)+self.a,
                np.einsum('m,nmij->nij', self.w, K))

    def reference(self, x, tol=1e-13, max_iter=3000):
        z = np.zeros((len(x), self.m))
        for _ in range(max_iter):
            zn = np.logaddexp(0, z@self.W.T+x@self.A.T+self.b)
            if np.max(abs(zn-z))/(1-self.q) <= tol:
                break
            z = zn
        else:
            raise RuntimeError('Reference state did not converge')
        d = np.exp(-np.logaddexp(0, -(z@self.W.T+x@self.A.T+self.b)))
        B = np.eye(self.m)[None]-d[..., None]*self.W
        J = np.linalg.solve(B, d[..., None]*self.A)
        V = self.A+np.einsum('ab,nbi->nai', self.W, J)
        rhs = (d*(1-d))[..., None, None]*V[..., :, None]*V[..., None, :]
        K = np.linalg.solve(B, rhs.reshape(-1, self.m, 4)).reshape(-1, self.m, 2, 2)
        K = (K+K.swapaxes(-1, -2))/2
        return z, J, K

    def certificate(self, x, z, J, K):
        s = z@self.W.T+x@self.A.T+self.b
        d = np.exp(-np.logaddexp(0, -s))
        gamma = d*(1-d)
        B = np.eye(self.m)[None]-d[..., None]*self.W
        V = self.A+np.einsum('ab,nbi->nai', self.W, J)
        U = abs(V).max(1)
        rz = abs(z-np.logaddexp(0, s)).max(1)
        rj = abs(B@J-d[..., None]*self.A).max(1)
        rhs = gamma[..., None, None]*V[..., :, None]*V[..., None, :]
        rk = abs((B@K.reshape(-1, self.m, 4)).reshape(-1, self.m, 2, 2)-rhs).max(1)
        Ez = rz/(1-self.q)
        alpha = self.q*Ez/4
        Ej = (rj+alpha[:, None]*U)/(1-self.q)
        Delta = self.q*Ej
        Q = alpha[:, None, None]*U[..., :, None]*U[..., None, :]
        Q += (Delta[..., :, None]*U[..., None, :]+U[..., :, None]*Delta[..., None, :]
              +Delta[..., :, None]*Delta[..., None, :])/4
        Ek = (rk+Q+alpha[:, None, None]*self.q*abs(K).max(1))/(1-self.q)
        EH = np.sum(abs(self.w))*np.sqrt((Ek*Ek).sum((1, 2)))
        H = self.output(x, z, J, K)[2]
        E_det = 2*np.linalg.norm(H, ord=2, axis=(1, 2))*EH+EH*EH
        return {'state': Ez, 'first': Ej, 'second': Ek, 'hessian': EH, 'det': E_det}


def det(H):
    return H[:, 0, 0]*H[:, 1, 1]-H[:, 0, 1]*H[:, 1, 0]


def adaptive(net, x, tolerance, checkpoints):
    """Per-point stopping; includes certificate, indexing and compaction costs."""
    active = np.arange(len(x)); jets = net.zero(len(x)); previous = 0
    result = np.full(len(x), np.nan); bounds = result.copy()
    depths = np.zeros(len(x), dtype=int); accepted = np.zeros(len(x), dtype=bool)
    for depth in checkpoints:
        for _ in range(depth-previous):
            jets = net.step(x[active], *jets)
        cert = net.certificate(x[active], *jets)['det']
        good = cert <= tolerance
        take = good if depth != checkpoints[-1] else np.ones(len(active), dtype=bool)
        ids = active[take]
        result[ids] = det(net.output(x[active], *jets)[2])[take]
        bounds[ids] = cert[take]; depths[ids] = depth; accepted[ids] = good[take]
        keep = ~take
        active = active[keep]; jets = tuple(v[keep] for v in jets)
        previous = depth
        if not len(active):
            break
    return result, bounds, depths, accepted

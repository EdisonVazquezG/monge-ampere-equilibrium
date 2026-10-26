"""Independent numerical oracles and enclosure/coverage invariants.
Run: python test_certification.py. These tests complement mathematical review.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import unittest
import numpy as np
import mpmath as mp
from flint import arb, ctx
from numerics import Net, det
from rigorous import RigorousNet, exact, export_upper, point_box, dyadic_box, softplus_ball, sigmoid_ball
from coverage_check import check_cover


class CertificationTests(unittest.TestCase):
    def test_outward_export(self):
        ctx.prec = 160
        for v in [arb(1)/3, arb(2)/7, arb('1e-200'), arb('1e200'), arb(0)]:
            self.assertTrue(exact(export_upper(v)) >= v.upper())

    def test_monotone_functions_wide_ball(self):
        ctx.prec = 128
        wide = arb(0, 100)
        for v in [-100, -3, 0, 3, 100]:
            self.assertTrue(softplus_ball(wide).contains((1+arb(v).exp()).log()))
            self.assertTrue(sigmoid_ball(wide).contains(1/(1+(-arb(v)).exp())))

    def test_closed_form_separable(self):
        net = Net(parameters={'W': np.zeros((2, 2)), 'A': np.eye(2), 'b': np.zeros(2),
                              'w': np.ones(2), 'a': np.zeros(2), 'c': np.array(0.)})
        rig = RigorousNet(net)
        for x in [np.array([a, b]) for a in [-1., 0., 1.] for b in [-1., 0., 1.]]:
            jets = tuple(v[0] for v in net.unroll(x[None], 4))
            result = rig.certify(point_box(x), jets, 'P1')
            d = [1/(1+(-exact(v)).exp()) for v in x]
            analytic_det = d[0]*(1-d[0])*d[1]*(1-d[1])
            self.assertTrue(exact(result['det_lower']) <= analytic_det.lower())
            self.assertTrue(exact(result['det_upper']) >= analytic_det.upper())
        # Entire sign-changing box: must not produce NaN (Arb generic power pitfall).
        result = rig.certify_box(dyadic_box(0, 0, 0), tuple(v[0] for v in net.reference(np.zeros((1, 2)))), 'P2')
        self.assertTrue(np.isfinite(result['pde_residual_upper']))

    def test_coupled_against_80_digit_oracle(self):
        mp.mp.dps = 80
        rng = np.random.default_rng(7231)
        for q in [.2, .7, .95]:
            W = rng.uniform(.01, 1, (3, 3)); W *= q/W.sum(1)[:, None]
            arrays = dict(W=W, A=rng.normal(0, .6, (3, 2)), b=rng.normal(0, .2, 3),
                          w=rng.uniform(.2, 1, 3), a=rng.normal(0, .1, 2), c=np.array(.125))
            net = Net(parameters=arrays); rig = RigorousNet(net)
            for x in rng.uniform(-.7, .7, (2, 2)):
                z, J, K = [v[0] for v in net.unroll(x[None], 8)]
                # Stress all three residual terms, not just the state residual.
                J += rng.normal(0, 1e-4, J.shape)
                noise = rng.normal(0, 1e-4, K.shape); K += (noise+noise.swapaxes(-1, -2))/2
                result = rig.certify(point_box(x), (z, J, K), 'P2')
                wm = mp.matrix(W.tolist()); am = mp.matrix(arrays['A'].tolist())
                bm = mp.matrix(arrays['b'].tolist()); xx = mp.matrix(x.tolist()); zm = mp.zeros(3, 1)
                for _ in range(10000):
                    ss = wm*zm+am*xx+bm
                    zn = mp.matrix([mp.log(1+mp.exp(t)) for t in ss])
                    if max(abs(zn[i]-zm[i]) for i in range(3)) < mp.mpf('1e-70'):
                        zm = zn; break
                    zm = zn
                else:
                    self.fail('High precision reference failed')
                ss = wm*zm+am*xx+bm; ds = [1/(1+mp.exp(-t)) for t in ss]
                B = mp.eye(3)-mp.diag(ds)*wm
                JJ = B**-1*mp.diag(ds)*am; V = wm*JJ+am
                H = mp.zeros(2, 2)
                for i in range(2):
                    for j in range(2):
                        kk = mp.lu_solve(B, mp.matrix([ds[r]*(1-ds[r])*V[r, i]*V[r, j] for r in range(3)]))
                        H[i, j] = sum(mp.mpf(float(arrays['w'][r]))*kk[r] for r in range(3))
                dd = mp.det(H)
                self.assertLessEqual(mp.mpf(result['det_lower']), dd)
                self.assertGreaterEqual(mp.mpf(result['det_upper']), dd)

    def test_coverage_and_reject_gap(self):
        leaves = [dict(ix=i, iy=j, level=1) for i in [0, 1] for j in [0, 1]]
        arcs = [dict(index=i, level=2) for i in range(4)]
        self.assertTrue(check_cover(leaves, [], arcs))
        with self.assertRaises(AssertionError):
            check_cover(leaves[:-1], [], arcs)

    def test_float_jets_match_original_autograd(self):
        import torch
        import original_pilots as original
        model = original.ConvexPilot(seed=6)
        arrays = {k: v.detach().numpy() for k, v in zip(('W', 'A', 'b', 'w', 'a', 'c'), model.constrained())}
        net = Net(parameters=arrays)
        x = np.array([[.1, -.2], [.3, .6]])
        _, _, H, _ = model.jets(torch.tensor(x), 8)
        Hn = net.output(x, *net.unroll(x, 8))[2]
        np.testing.assert_allclose(Hn, H.detach().numpy(), atol=1e-12, rtol=1e-12)
        for k, point in enumerate(x):
            h = torch.autograd.functional.hessian(lambda t: model.value_unrolled(t[None], 8)[0], torch.tensor(point))
            np.testing.assert_allclose(Hn[k], h.numpy(), atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)

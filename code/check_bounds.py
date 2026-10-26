"""Independent float64 consistency checks, not interval verification or pilot reproduction.
Run: python3 check_bounds.py
Requires NumPy and Matplotlib. Seed and every sample are fixed below.
"""
from pathlib import Path
import json
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
RNG=np.random.default_rng(20261015)

def softplus(s): return np.logaddexp(0.,s)
def sigmoid(s): return np.exp(-np.logaddexp(0.,-s))

def equilibrium(W,A,b,x,tol=1e-13):
    z=np.zeros(len(b))
    for it in range(10000):
        zn=softplus(W@z+A@x+b)
        z=zn
        rho=np.max(np.abs(z-softplus(W@z+A@x+b)))
        if rho<=tol: return z,rho
    raise RuntimeError('Equilibrium did not converge')

def jets(W,A,b,z,x):
    s=W@z+A@x+b; d=sigmoid(s); gamma=d*(1-d)
    B=np.eye(len(z))-d[:,None]*W
    J=np.linalg.solve(B,d[:,None]*A)
    V=A+W@J
    K=np.zeros((len(z),2,2))
    for i in range(2):
        for j in range(i,2):
            K[:,i,j]=K[:,j,i]=np.linalg.solve(B,gamma*V[:,i]*V[:,j])
    return J,K

def certificate(W,A,b,w,x,z,J,K,q):
    delta=1-q;s=W@z+A@x+b;d=sigmoid(s);g=d*(1-d)
    B=np.eye(len(z))-d[:,None]*W;V=A+W@J
    rz=np.max(abs(z-softplus(s))); Ez=rz/delta;alpha=q*Ez/4
    U=np.max(abs(V),axis=0)
    ri=np.max(abs(B@J-d[:,None]*A),axis=0)
    Ei=(ri+alpha*U)/delta;Delta=q*Ei
    Eij=np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            rij=np.max(abs(B@K[:,i,j]-g*V[:,i]*V[:,j]))
            Q=alpha*U[i]*U[j]+(Delta[i]*U[j]+U[i]*Delta[j]+Delta[i]*Delta[j])/4
            Eij[i,j]=(rij+Q+alpha*q*np.max(abs(K[:,i,j])))/delta
    H=np.einsum('m,mij->ij',w,K)
    EH=np.sum(w)*np.linalg.norm(Eij,'fro')
    Edet=2*np.linalg.norm(H,2)*EH+EH*EH
    return Ez,Ei,Eij,EH,Edet,H

def run():
    maxima={k:0. for k in ['state','first_derivative','second_derivative','hessian','determinant','curvature','cancellation']}
    violations={k:0 for k in maxima}; reference_bound=0.;factor_error=0.
    for q in [.3,.7,.95]:
        for sample in range(40):
            m=8;W=RNG.uniform(.05,1.,(m,m));W=q*W/W.sum(axis=1,keepdims=True)
            A=RNG.normal(0,.5,(m,2));b=RNG.normal(0,.5,m);w=RNG.uniform(.1,1,m)
            x=RNG.uniform(-1,1,2)
            z,rho=equilibrium(W,A,b,x)
            reference_bound=max(reference_bound,rho/(1-q))
            J,K=jets(W,A,b,z,x);H=np.einsum('m,mij->ij',w,K)
            s=W@z+A@x+b;d=sigmoid(s);g=d*(1-d);B=np.eye(m)-d[:,None]*W
            V=np.linalg.solve(np.eye(m)-W*d[None,:],A)
            beta=np.linalg.solve(B.T,w)*g;Hfact=V.T@(beta[:,None]*V)
            factor_error=max(factor_error,np.linalg.norm(H-Hfact,2)/max(1.,np.linalg.norm(H,2)))
            C2=w.sum()*np.max(np.linalg.norm(A,axis=1))**2*min((1-q)**-2,1/(4*(1-q)**3))
            zhat=np.zeros(m)
            for _ in range([4,8,12,16][sample%4]): zhat=softplus(W@zhat+A@x+b)
            Jh,Kh=jets(W,A,b,zhat,x)
            eta=10.**RNG.uniform(-7,-3)
            Jh=Jh+RNG.normal(0,eta,Jh.shape)
            noise=RNG.normal(0,eta,Kh.shape);Kh=Kh+(noise+noise.transpose(0,2,1))/2
            Ez,Ei,Eij,EH,Edet,Hh=certificate(W,A,b,w,x,zhat,Jh,Kh,q)
            ratios={'state':np.max(abs(z-zhat))/Ez,
                    'first_derivative':np.max(np.max(abs(J-Jh),axis=0)/Ei),
                    'second_derivative':np.max(np.max(abs(K-Kh),axis=0)/Eij),
                    'hessian':np.linalg.norm(H-Hh,2)/EH,
                    'determinant':abs(np.linalg.det(H)-np.linalg.det(Hh))/Edet,
                    'curvature':np.linalg.norm(H,2)/C2,
                    'cancellation':np.max(np.linalg.solve(B,g))}
            for key,val in ratios.items():
                maxima[key]=max(maxima[key],float(val))
                violations[key]+=int(val>1.+1e-8)
    results={'seed':20261015,'networks':120,'width':8,'q_values':[.3,.7,.95],
             'max_actual_to_bound':maxima,'violations':violations,
             'max_reference_state_residual_bound':float(reference_bound),
             'max_hessian_factorization_scaled_error':float(factor_error),
             'python_version':platform.python_version(),'numpy_version':np.__version__,'matplotlib_version':matplotlib.__version__,
             'arithmetic':'float64; reference equilibria and certificates are not interval enclosed'}
    (OUT/'independent_checks.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=[r'\begin{tabular}{lrr}',r'\toprule Quantity & Maximum ratio & Violations \\',r'\midrule']
    labels={'state':'State error','first_derivative':'First derivative error','second_derivative':'Second derivative error','hessian':'Hessian error','determinant':'Determinant error','curvature':'Curvature bound','cancellation':r'Cancellation $S\gamma\le\mathbf 1$'}
    for k in maxima: lines.append(f'{labels[k]} & {maxima[k]:.5f} & {violations[k]} '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (OUT/'check_table.tex').write_text('\n'.join(lines)+'\n')
    eps=np.logspace(-4,-.3,200);curv=(1-eps)/(eps**2*(2-eps)**3)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.35))
    ax=axes[0];ax.loglog(eps,curv,color='#155f83',lw=2,label='Exact curvature')
    ax.loglog(eps,eps**-2,color='#800000',lw=1.8,label=r'Quadratic bound $\varepsilon^{-2}$')
    ax.loglog(eps,1/(4*eps**3),'--',color='#777777',label=r'Cubic bound $(4\varepsilon^3)^{-1}$')
    ax.set_xlabel(r'$\varepsilon=1-q$');ax.set_ylabel(r'$u_\varepsilon^{\prime\prime}(0)$ or upper bound')
    ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.15)
    ax=axes[1];ax.semilogx(eps,eps**2*curv,color='#155f83',lw=2)
    ax.axhline(.125,color='#800000',ls='--',label=r'Limit $1/8$');ax.set_ylim(.11,.15)
    ax.set_xlabel(r'$\varepsilon=1-q$');ax.set_ylabel(r'$\varepsilon^2u_\varepsilon^{\prime\prime}(0)$')
    ax.legend(frameon=False);ax.grid(alpha=.15)
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'sharpness.pdf');plt.close(fig)
    print(json.dumps(results,indent=2))

if __name__=='__main__': run()

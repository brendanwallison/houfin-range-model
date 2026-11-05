"""Small matrix-free curvature helpers, independent of the population model."""
import numpy as np


def lanczos(operator,dimension,steps,seed=20261101):
    """Fully reorthogonalized Lanczos; report Ritz estimates and residuals.

    Extremal Ritz values are not a full eigenvalue count or a certified global
    condition number. This remains true when the observed Ritz values are positive.
    """
    rng=np.random.default_rng(seed);q=rng.normal(size=dimension);q/=np.linalg.norm(q)
    basis=[];alpha=[];beta=[];previous=np.zeros(dimension);last_beta=0.
    for i in range(min(steps,dimension)):
        basis.append(q.copy());w=np.array(operator(q),dtype=float,copy=True)
        if not np.isfinite(w).all():raise ValueError('nonfinite Hessian-vector product')
        if i:w-=last_beta*previous
        value=float(q@w);alpha.append(value);w-=value*q
        Q=np.asarray(basis)
        for _ in range(2):w-=Q.T@(Q@w)
        b=float(np.linalg.norm(w));last_beta=b
        if b<1e-12*max(abs(value),1.):break
        beta.append(b);previous,q=q,w/b
    T=np.diag(alpha)
    if len(alpha)>1:T+=np.diag(beta[:len(alpha)-1],1)+np.diag(beta[:len(alpha)-1],-1)
    values,vectors=np.linalg.eigh(T);Q=np.asarray(basis).T
    residuals=last_beta*np.abs(vectors[-1])
    return dict(values=values,vectors=Q@vectors,residuals=residuals,
                orthogonality_error=float(np.max(np.abs(Q.T@Q-np.eye(len(alpha))))),steps=len(alpha))


def spectrum_report(result):
    ev=result['values'];pos=ev[ev>0]
    return {'ritz_min':float(ev[0]),'ritz_max':float(ev[-1]),
        'ritz_values':ev.tolist(),'ritz_residual_norms':result['residuals'].tolist(),
        'observed_negative_ritz_values':int((ev<0).sum()),
        'positive_ritz_ratio':float(pos.max()/pos.min()) if len(pos)>1 else None,
        'steps':result['steps'],'orthogonality_error':result['orthogonality_error'],
        'scope':'Krylov Ritz estimates, not full-spectrum inertia or a certified condition number; no eigenvalue clipping used'}

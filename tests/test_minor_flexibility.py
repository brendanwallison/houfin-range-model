from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/exp'))
import numpy as np
from minor_flexibility import active_indices,normalize_cell_variance,iid_log_innovations,regional_log_innovations


def test_dispatch_covers_distinct_hypotheses_before_replicate_seeds():
    from dispatch_order import dispatch_order
    names=['0609_step44_allrho99_s42.sh','0609_step44_kprivate10_s41.sh',
           '0609_step45_regional_s010_s41.sh','0609_step45_cell_iid_s010_s41.sh',
           '0608_step44_shared_s43.sh','0612_step46_uniform05_s41.sh']
    assert sorted(names,key=dispatch_order)==[names[i] for i in [4,3,1,2,0,5]]


def test_active_innovations_do_not_leak_into_forecast_or_earlier_years():
    years=np.arange(1960,2026)
    idx,active=active_indices(years,1966,2025,[2021,2022,2023,2024,2025])
    assert len(idx)==55 and years[idx[0]]==1966 and years[idx[-1]]==2020
    raw=np.arange(55*3,dtype=float).reshape(55,3)/100
    eps=iid_log_innovations(raw,idx,len(years),.01)
    assert np.array_equal(eps[active==0],np.zeros((11,3)))
    np.testing.assert_allclose(eps[idx],.01*raw-.00005)
    state=np.array([0.,3.,8.])
    assert np.all((state*np.exp(eps))[:,0]==0)


def test_regional_rows_have_same_cell_variance_as_iid():
    rng=np.random.default_rng(194)
    S=normalize_cell_variance(rng.normal(size=(9,4)))
    np.testing.assert_allclose((S*S).sum(1),1.)
    raw=rng.normal(size=(100000,4));active=np.ones(len(raw));sigma=.02
    eps=regional_log_innovations(raw,active,S,sigma)
    np.testing.assert_allclose(np.var(eps,axis=0),sigma**2,rtol=.025)
    np.testing.assert_allclose(np.mean(np.exp(eps),axis=0),1.,atol=.0003)


def test_private_K_prior_covariance_is_strong_but_not_exact_identity():
    rng=np.random.default_rng(42);shared=rng.normal(size=150000);private=rng.normal(size=150000)
    for tau in [.1,.25]:
        K=shared+tau*private
        np.testing.assert_allclose(np.corrcoef(shared,K)[0,1],1/np.sqrt(1+tau*tau),atol=.001)
        np.testing.assert_allclose(np.var(K),1+tau*tau,rtol=.015)


def test_iid_prior_mean_is_one_and_no_dense_identity_required():
    rng=np.random.default_rng(3);raw=rng.normal(size=(100000,3))
    eps=iid_log_innovations(raw,np.arange(len(raw)),len(raw),.02)
    np.testing.assert_allclose(np.mean(np.exp(eps),axis=0),1.,atol=.0003)
    assert np.max(np.abs(np.corrcoef(eps.T)-np.eye(3)))<.015


def test_process_multiplier_origin_accounting_against_dense_matrix():
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from contribution_budget import attributed_step
    A=np.array([[.7,.1],[.2,.8]]);S=np.diag([.4,.5]);J=np.array([[.2,.1],[.1,.2]])
    a=np.array([4.,9.]);j=np.array([7.,2.]);sa=np.array([.6,.8]);sj=np.array([.3,.4]);f=np.array([2.,3.])
    mult=np.array([.98,1.04])
    def move(aa,jj):return jnp.array(A)@aa,jnp.array(S)@jj,jnp.array(J)@jj
    b=attributed_step(move,jnp.array(a),jnp.array(j),jnp.array(sa*mult),jnp.array(sj*mult),jnp.array(f),
       tuple(jnp.array(np.diag(k)) for k in [A,S,J]),jnp.array(mult))
    origin_matrix=np.diag(mult)@(np.diag(sa)@A@np.diag(a)+(np.diag(sj)@S+J)@np.diag(j))
    np.testing.assert_allclose(b['breeder_contribution'],origin_matrix.sum(0),atol=1e-12)
    np.testing.assert_allclose(b['next_breeders'],origin_matrix.sum(1),atol=1e-12)

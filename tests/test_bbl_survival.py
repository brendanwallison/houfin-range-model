import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit
from scipy.special import xlogy

from research.exp.bbl_survival import marked_transition, marked_recovery_distribution, recovery_probabilities, negative_loglik, prepare


def test_mover_survival_replaces_resident_survival():
    t = marked_transition(np.array([0.,1.]), np.array([.2,.3]),
                          np.array([[0.,1.],[1.,0.]]), np.array([.8,.9]))
    np.testing.assert_allclose(t, [[.2,0],[.8,0]])


def test_marked_probability_distribution_not_survival_at_mean_location():
    t = marked_transition(np.array([.5,.5]), np.array([.2,.3]),
                          np.array([[.25,.75],[.5,.5]]), np.array([.8,.4]))
    np.testing.assert_allclose(t.sum(1), [.35,.45])
    assert np.all(t.sum(1) <= 1)
    with pytest.raises(ValueError):
        marked_transition(np.array([.5,.5]),np.array([.2,.3]),np.ones((2,2)),np.ones(2))


def test_recovery_probabilities_match_closed_form_and_censoring():
    s,r,n=.6,.02,24
    p=recovery_probabilities(s,np.full(n,r),n)
    np.testing.assert_allclose(p.sum(1),1)
    assert p[3,2] == 0
    assert p[0,0] == pytest.approx((1-s**(1/24))*r)
    assert p[0,1] == pytest.approx(s**(1/24)*(1-s**(1/12))*r)
    assert p[0,-1] == pytest.approx(1-r*(1-s**((n-.5)/12)))
    assert p[-1,-1] == pytest.approx(1-r*(1-s**(1/24)))


def test_marked_cohort_propagates_locations_without_births_or_double_survival():
    transition=np.array([[.2,.4],[.1,.7]])
    p=marked_recovery_distribution([1.,0.],np.array([transition,transition]),[.1,.2])
    # Interval 1: alive .6. Interval 2: [.2,.4] @ T = [.08,.36], alive .44.
    np.testing.assert_allclose(p,[.04,.032,.928])
    assert p.sum()==pytest.approx(1)


def test_marked_cohort_matches_scalar_monthly_recovery_model():
    s,r,n=.57,.025,18
    transitions=np.array([[[s**(1/24)]]]+[[[s**(1/12)]]]*(n-1))
    np.testing.assert_allclose(marked_recovery_distribution([1.],transitions,np.full(n,r)),
                               recovery_probabilities(s,np.full(n,r),n)[0],rtol=1e-12)


def test_reporting_is_conditional_on_death():
    p = recovery_probabilities(.8,np.full(12,.1),12)
    assert p[0,:-1].sum() < .021  # NOT 10% of bands recovered
    counts=np.zeros_like(p); counts[0,-1]=100
    assert np.isfinite(negative_loglik([0,-5],counts,np.zeros(12,dtype=int)))


def test_sparse_likelihood_agrees_with_independent_dense_probabilities():
    era=np.repeat(np.arange(2),12)
    theta=np.array([.3,-4,-3])
    p=recovery_probabilities(expit(theta[0]),expit(theta[1:])[era],24)
    rng=np.random.default_rng(7)
    counts=np.array([rng.multinomial(2000,row) for row in p])
    assert negative_loglik(theta,counts,era)==pytest.approx(-xlogy(counts,p).sum(),rel=1e-12)


def test_synthetic_recovery_with_reporting_refitted():
    n=120
    era=np.repeat(np.arange(2),n//2)
    truth_s,truth_r=.57,np.array([.015,.035])
    p=recovery_probabilities(truth_s,truth_r[era],n)
    rng=np.random.default_rng(20261031)
    counts=np.array([rng.multinomial(30000,row) for row in p])
    f=minimize(negative_loglik,[0,-4,-4],args=(counts,era),method="L-BFGS-B")
    assert f.success
    assert abs(expit(f.x[0])-truth_s)<.02
    np.testing.assert_allclose(expit(f.x[1:]),truth_r,rtol=.08)


def test_preparation_preserves_unrecovered_denominator_and_excludes_bad_dates(tmp_path):
    import pandas as pd
    rows=[]
    def add(key,event,date,month,condition="",how="",**kw):
        row=dict(band=key,original_band=key,event_type=event,event_date=date,
                         event_month=month,event_year="1960",species_id="5190",age_code="1",
                         bird_status="3",present_condition=condition,how_obtained_code=how,
                         lon_dd="-74",lat_dd="41")
        row.update(kw)
        rows.append(row)
    add("a","B","04/10/1960","04")
    add("a","E","06/11/1960","06","03","00")
    add("b","B","04/10/1960","04")
    add("c","B","04/10/1960","04")
    add("c","E","04/10/1960","04","03","10") # handling death not a recovery
    add("d","B","04/10/1960","04")
    add("d","E","03/11/1960","03","03","00") # chronology invalid: remove bird
    add("nestling","B","04/10/1960","04",age_code="4")
    src=tmp_path/"bands.csv"; pd.DataFrame(rows).to_csv(src,index=False)
    out=tmp_path/"out"; m=prepare(src,out,"1960-12-31")
    data=np.load(out/"adult_marray.npz")
    assert sum(m["adult_bands"])==2
    assert data["counts"][3,3,5]==1
    assert data["counts"][3,3,-1]==1
    assert m["bad_chronology_birds_excluded"]==1
    assert m["handling_mortality_birds_excluded"]==1

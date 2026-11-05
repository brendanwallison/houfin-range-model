import numpy as np

from research.exp.z_change_cpu import build_pairs,ChangeRidge,smooth_winter


def test_pairs_use_only_training_records_and_four_year_epochs():
    years=np.arange(1966,2026)
    keys=np.column_stack([np.zeros(len(years),int),np.zeros(len(years),int),years])
    train=(years>=1996)&(years%2==0)
    cells,kind,pairs=build_pairs(keys,train)
    assert len(pairs)==3
    for a,b in pairs:
        for epoch in [a,b]:
            assert all(len(half)==2 for half in epoch)
            rows=np.r_[*epoch]
            assert len(np.unique(rows))==4 and train[rows].all()
    poisoned=keys.copy();poisoned[~train,2]=-999
    other=build_pairs(poisoned,train)[2]
    for a,b in zip(pairs,other):np.testing.assert_array_equal(np.array(a),np.array(b))


def test_change_regression_recovers_planted_shared_map_and_zero_change():
    rng=np.random.default_rng(19);X=rng.normal(size=(120,6));B=rng.normal(size=(6,4))
    model=ChangeRidge().fit(X[:90],(X@B)[:90],[(list(range(6)),6)],1e-8)
    np.testing.assert_allclose(model.predict(X[90:]),X[90:]@B,atol=1e-6)
    np.testing.assert_array_equal(model.predict(np.zeros((2,6))),0)
    # Modifying unused validation inputs does not change fitted preprocessing.
    before=model.predict(X[90:]).copy();X[100:]*=100
    np.testing.assert_array_equal(model.predict(X[90:100]),before[:10])


def test_unsupported_winter_queries_retain_static_reference():
    query=np.array([[0.,0.],[10000.,10000.]])
    source=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    z=np.tile([2.,3.],(4,1));reference=np.tile([7.,8.],(2,1))
    pred,supported=smooth_winter(query,source,z,reference)
    np.testing.assert_allclose(pred[0],[2,3]);np.testing.assert_allclose(pred[1],reference[1])
    np.testing.assert_array_equal(supported,[True,False])

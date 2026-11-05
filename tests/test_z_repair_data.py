import numpy as np

from research.exp.z_repair_data import summer_masks, winter_masks, fit_benchmarks, winter_points


def test_modern_training_is_disjoint_from_both_eval_halves_and_history():
    years = np.r_[np.arange(1966,1986), np.arange(1996,2026)]
    keys = np.column_stack([np.zeros(len(years), int), np.zeros(len(years), int), years])
    zero = np.zeros((2,2), bool)
    m = summer_masks(keys, zero, zero, ~zero, {'target':[1966,1985], 'summer_start':1996})
    assert m['train'].sum() == 15
    assert not m['train'][years < 1996].any()
    roles = ['train','modern_a','modern_b','historical_a','historical_b']
    assert np.max(sum(m[k] for k in roles)) == 1
    assert m['panel'].shape == (1,7)


def test_relocated_circles_cannot_enter_training_under_another_cell():
    keys = np.array([[0,0,1950],[1,1,2000],[1,1,2001],[1,0,1950]])
    circle = np.array([7,7,8,9]);ho=np.zeros((2,2),bool);ho[0,0]=True
    m=winter_masks(keys,circle,ho,np.zeros_like(ho),np.ones_like(ho),[1966,1985])
    assert not m['train'][circle==7].any()
    assert m['evaluation'][circle==7].all()
    assert m['train'][circle==8].all()


def test_poisoned_evaluation_detections_do_not_change_fitted_benchmarks():
    c=np.repeat([0,1],6);years=np.tile(np.arange(1996,2002),2)
    p=np.zeros((12,3),bool);p[:6,0]=True;p[6:,1]=True
    train=c==0;lat=np.zeros(12);lon=np.repeat([0,.5],6)
    a,donor=fit_benchmarks(p,c,years,lat,lon,train,2)
    p[~train]=~p[~train]
    b,donor2=fit_benchmarks(p,c,years,lat,lon,train,2)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(donor,donor2)
    assert donor[1]==0


def test_poisoned_evaluation_counts_and_effort_do_not_change_winter_targets():
    d={'party_hours':np.array([10.,20.,100.]),'year_end':np.array([2000,2001,2002]),
       'Y':np.array([[1.,2.],[2.,3.],[9.,99.]]),'row':np.zeros(3,int),'col':np.zeros(3,int)}
    train=np.array([True,True,False]);a=winter_points(d,train)
    d['Y'][2]=1e9;d['party_hours'][2]=1e9;b=winter_points(d,train)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


def test_production_masks_train_every_valid_row_and_withhold_nothing():
    from research.exp.z_repair_data import production_summer_masks
    years = np.arange(1966, 2026)
    keys = np.column_stack([np.r_[np.zeros(len(years), int), [1]], np.zeros(len(years) + 1, int), np.r_[years, 2000]])
    land = np.array([[True], [False]])
    m = production_summer_masks(keys, land)
    assert m['train'][:-1].all() and not m['train'][-1]
    for k in ['modern_a', 'modern_b', 'historical_a', 'historical_b', 'spatial_holdout', 'buffer']:
        assert not m[k].any()
    assert m['panel'].shape == (0, 7)

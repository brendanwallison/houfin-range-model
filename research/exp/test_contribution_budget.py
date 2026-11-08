import unittest
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from contribution_budget import attributed_step,audit_forward,select_main_forward
try:   # the forward model (TAKEOVER_CODE_ROOT on PYTHONPATH, AGE_MODEL_CONFIG set); pure-algebra tests run without it
    from src.model import age_forward as af
except Exception:   # pragma: no cover
    af=None


def _kernel_fft(ny,nx,sigma):
    ly,lx=2*ny-1,2*nx-1
    dy=np.fft.fftfreq(ly)*ly;dx=np.fft.fftfreq(lx)*lx
    k=np.exp(-(dy[:,None]**2+dx[None,:]**2)/(2*sigma**2));k/=k.sum()
    return np.fft.fft2(k)


def _edge(land,kfft):
    ly,lx=kfft.shape;pad=np.zeros((ly,lx));pad[:land.shape[0],:land.shape[1]]=land
    return np.real(np.fft.ifft2(np.fft.fft2(pad)*kfft))[:land.shape[0],:land.shape[1]]


def synthetic_args(T=9,seed=0,dd=True,departure=True,disease=True):
    """Small grid exercising the candidate mechanisms: Veit departure x native departure scale, juvenile kernel stack
    with journey survival, epizootic survival, DD adult survival, yearling breeding, a release pulse, and a front
    (empty right half) so the FFT floor acts on far-field arrivals."""
    r=np.random.default_rng(seed);ny,nx=9,12
    land=np.ones((ny,nx));land[0,:3]=0;land[-1,-4:]=0
    rows,cols=np.nonzero(land);n=len(rows)
    ka=_kernel_fft(ny,nx,.9);kj=np.stack([_kernel_fft(ny,nx,s) for s in (.7,1.6)])
    init=np.zeros((ny,nx));init[:,:4]=r.uniform(.2,1.5,(ny,4));init*=land
    u=lambda lo,hi,*s:jnp.asarray(r.uniform(lo,hi,s))
    return dict(Sa_flat=u(.3,.6,T,n),Sj_flat=u(.2,.5,T,n),Fmax_flat=u(1.,3.,T,n),K_flat=u(.5,2.,T,n),c_flat=u(.5,2.,T,n),
        Q_flat=u(.5,.95,T,n,2),land_rows=jnp.asarray(rows),land_cols=jnp.asarray(cols),land_mask=jnp.asarray(land),
        adult_fft_kernel=jnp.asarray(ka),juvenile_fft_kernel_stack=jnp.asarray(kj),
        adult_edge_correction=jnp.asarray(_edge(land,ka)),juvenile_edge_correction_stack=jnp.asarray(np.stack([_edge(land,k) for k in kj])),
        initpop_latent=jnp.asarray(init),dispersal_random=jnp.zeros(T),inv_pop=jnp.asarray([[.4,.3,.2]]),time=T,
        inv_locations=jnp.asarray([[4,8]]),inv_timestep=3,dispersal_logit_intercept=jnp.asarray(-1.),
        dispersal_logit_slope=jnp.asarray(1.),allee_gamma=jnp.asarray(2.),veit_departure=True,yearling_breeding=True,
        departure_scale=u(.2,1.,n) if departure else None,survival_dd_c=u(0.,2.,T,n) if dd else None,
        disease_m_amp=u(0.,.3,T,n) if disease else None,disease_k_half=jnp.asarray(.5),disease_hill_n=jnp.asarray(2.))


def run_forward(args):
    import inspect
    sig=inspect.signature(af.forward_sim_age_structured);b=sig.bind(**args);b.apply_defaults()
    return dict(b.arguments),af.forward_sim_age_structured(**args)


@unittest.skipIf(af is None,'forward model not importable (set TAKEOVER_CODE_ROOT / AGE_MODEL_CONFIG)')
class ForwardReplayTests(unittest.TestCase):
    def setUp(self):
        self.saved=(af.FFT_PAD_VALUE,af.FFT_FLOOR_ULPS,af.FFT_FLOOR_SMOOTH,getattr(af,'FFT_FLOOR_FLOAT32_ONLY',False))
        af.FFT_PAD_VALUE=0.;af.FFT_FLOOR_FLOAT32_ONLY=False

    def tearDown(self):
        af.FFT_PAD_VALUE,af.FFT_FLOOR_ULPS,af.FFT_FLOOR_SMOOTH,af.FFT_FLOOR_FLOAT32_ONLY=self.saved

    def audit(self,args):
        x,states=run_forward(args)
        arrays,_=audit_forward(af,x,states)
        return x,states,arrays

    def assert_exact(self,arrays,states,tol=1e-12):
        scale=float(np.max(np.asarray(states[0])))
        for key in ['replay_adult_error','replay_juvenile_error','BIDE_error','survival_identity_error']:
            self.assertLess(float(np.max(np.abs(arrays[key]))),tol*scale,key)
        breeder=(arrays['breeder_contribution']+arrays['numerical_adult_clip_gain']+arrays['numerical_fft_floor_breeders']).sum(1)
        np.testing.assert_allclose(breeder,arrays['next_adult'].sum(1),rtol=0,atol=tol*scale*100)
        post=(arrays['total_descendants_origin']+arrays['numerical_unparented_births']+arrays['numerical_adult_clip_gain']
              +arrays['numerical_fft_floor_postbirth']).sum(1)
        np.testing.assert_allclose(post,(arrays['next_adult']+arrays['next_juvenile']).sum(1),rtol=0,atol=tol*scale*100)

    def test_fft_floor_is_replayed_and_carried_as_explicit_term(self):
        # A deliberately coarse floor (tol ~ 2e-8 x max) so the floor moves real mass; production uses 64 ulps.
        af.FFT_FLOOR_ULPS=1e8;af.FFT_FLOOR_SMOOTH=True
        x,states,arrays=self.audit(synthetic_args())
        self.assert_exact(arrays,states)
        floor=np.abs(arrays['numerical_fft_floor_breeders']).sum(1)
        self.assertGreater(float(floor.max()),1e-9)   # the test exercises the floor
        # Without the explicit floor term the origin-destination closure fails by the floor's mass.
        bad=np.abs((arrays['breeder_contribution']+arrays['numerical_adult_clip_gain']).sum(1)-arrays['next_adult'].sum(1))
        self.assertGreater(float(bad.max()),1e3*float(np.max(np.abs(arrays['BIDE_error']))))
        # Hard floor too.
        af.FFT_FLOOR_SMOOTH=False
        x,states,arrays=self.audit(synthetic_args(seed=1))
        self.assert_exact(arrays,states)

    def test_floor_free_model_has_zero_floor_term(self):
        af.FFT_FLOOR_ULPS=None
        x,states,arrays=self.audit(synthetic_args())
        self.assert_exact(arrays,states)
        self.assertEqual(float(np.max(np.abs(arrays['numerical_fft_floor_destination']))),0.)

    def test_dd_adult_survival_is_realised(self):
        af.FFT_FLOOR_ULPS=64.;af.FFT_FLOOR_SMOOTH=True
        args=synthetic_args()
        x,states,arrays=self.audit(args)
        self.assert_exact(arrays,states)
        # local_adult_survival is the braked (realised) survival: below the disease-adjusted base where birds are present
        T=args['time'];base=np.asarray(args['Sa_flat'])
        occupied=(arrays['input_adult']+arrays['input_juvenile'])>1e-3
        self.assertTrue(np.all(arrays['local_adult_survival'][occupied]<base[occupied]))
        # Negative control: the same trajectory audited WITHOUT the DD brake does not replay.
        x2=dict(x,survival_dd_c=None)
        arrays2,_=audit_forward(af,x2,states)
        self.assertGreater(float(np.max(np.abs(arrays2['replay_adult_error']))),1e-6)

    def test_spinup_initial_state_and_selection(self):
        af.FFT_FLOOR_ULPS=64.;af.FFT_FLOOR_SMOOTH=True
        S=6;spin=synthetic_args(T=S,seed=3);spin.update(inv_timestep=S+1000)
        xs,rs=run_forward(spin)
        main=synthetic_args(seed=4);main['initpop_latent']=rs[0][-1]
        xm,rm=run_forward(main)
        x,states,info=select_main_forward([(xs,rs),(xm,rm)],{'enabled':True,'years':S,'habitat_years':5})
        self.assertIs(x,xm);self.assertTrue(info['spinup'])
        arrays,_=audit_forward(af,x,states)
        self.assert_exact(arrays,states)
        # t=0 state is the model's: the spun-up total split 50/50 (the replay of year 0 is exact only from it)
        np.testing.assert_allclose(arrays['input_adult'][0],np.asarray(rs[0][-1])[x['land_rows'],x['land_cols']]*.5)
        bad=dict(xm,initpop_latent=rs[0][-2])
        with self.assertRaises(ValueError):select_main_forward([(xs,rs),(bad,rm)],{'enabled':True,'years':S})
        with self.assertRaises(ValueError):select_main_forward([(xm,rm),(xs,rs)],{'enabled':True,'years':S})
        with self.assertRaises(ValueError):select_main_forward([(xs,rs),(xm,rm)],None)

    def test_unsupported_mechanisms_raise(self):
        af.FFT_FLOOR_ULPS=None
        x,states=run_forward(synthetic_args(T=4))
        for key,val in [('kernel_ramp',(1,2,3)),('lineage_delta',.1),('urban_mating',(1,2,3)),('colonization_clock',(1,2,3,4)),
                        ('occupancy_b',1.),('attraction',(0.,.1,True))]:
            with self.assertRaises(NotImplementedError,msg=key):audit_forward(af,dict(x,**{key:val}),states)
        af.FFT_PAD_VALUE=1e-9
        with self.assertRaises(NotImplementedError):audit_forward(af,x,states)


class ContributionTests(unittest.TestCase):
    def test_all_origin_flows_against_independent_matrix_enumeration(self):
        # Column = origin. Juvenile movers already include journey survival.
        A=np.array([[.6,.2],[.3,.7]])
        S=np.diag([.2,.4]);J=np.array([[.1,.2],[.3,.1]])
        sa=np.array([.5,.8]);sj=np.array([.3,.6]);a=np.array([7.,3.]);j=np.array([11.,5.]);f=np.array([2.,4.])
        def move(aa,jj):return jnp.asarray(A)@aa,jnp.asarray(S)@jj,jnp.asarray(J)@jj
        b=attributed_step(move,jnp.array(a),jnp.array(j),jnp.array(sa),jnp.array(sj),jnp.array(f),
                         tuple(jnp.array(np.diag(v)) for v in [A,S,J]))
        breeder=np.diag(sa)@A@np.diag(a)+(np.diag(sj)@S+J)@np.diag(j)
        mass=A@np.diag(a)+(S+J)@np.diag(j)
        np.testing.assert_allclose(b['breeder_contribution'],breeder.sum(0))
        np.testing.assert_allclose(b['next_breeders'],breeder.sum(1))
        np.testing.assert_allclose(b['retained_breeders'],np.diag(breeder))
        np.testing.assert_allclose(b['successful_breeder_export'],breeder.sum(0)-np.diag(breeder))
        np.testing.assert_allclose(b['immigrant_breeders'],breeder.sum(1)-np.diag(breeder))
        np.testing.assert_allclose(b['successful_emigration'],mass.sum(0)-np.diag(mass))
        np.testing.assert_allclose(b['immigration'],mass.sum(1)-np.diag(mass))
        np.testing.assert_allclose(b['total_descendants_origin'],((1+f)[:,None]*breeder).sum(0))
        self.assertAlmostEqual(float(b['breeder_contribution'].sum()),float(b['next_breeders'].sum()))

    def test_exporting_source_is_not_an_apparent_sink(self):
        # Origin 0 exports most of its successful contribution. Incoming animals
        # to origin 1 never get credited back to origin 1.
        A=jnp.array([[.1,0.],[.8,.1]]);J=jnp.array([[.05,0.],[.5,.1]])
        def move(a,j):return A@a,jnp.zeros_like(j),J@j
        b=attributed_step(move,jnp.array([1.,1.]),jnp.array([2.,0.]),jnp.ones(2),jnp.ones(2),jnp.zeros(2),
             (jnp.diag(A),jnp.zeros(2),jnp.diag(J)))
        self.assertGreater(float(b['breeder_contribution'][0]),1.)
        self.assertLess(float(b['retained_breeders'][0]),1.)
        self.assertLess(float(b['breeder_contribution'][1]),1.)
        self.assertGreater(float(b['next_breeders'][1]),1.)


if __name__=='__main__':unittest.main()

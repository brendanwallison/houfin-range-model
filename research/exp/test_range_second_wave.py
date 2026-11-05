import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import range_second_wave as wave


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'research/grid54'
        for p in ('monitor','runs','queue/pending','queue/running','queue/done','queue/failed'):
            (self.root/p).mkdir(parents=True)

    def test_budget_reserves_queued_work_and_charges_failures(self):
        ledger = {'staged': {f'step34_{i}': {} for i in range(18)}}
        with patch.object(wave, 'eligible', return_value=['foundlo']), patch.object(wave,'stage') as stage:
            wave.tick(self.root, ledger, wave.DEADLINE-1)
            stage.assert_not_called()
        run = self.root/'runs/step34_0'
        run.mkdir()
        (run/'RUN_RECEIPT.json').write_text(json.dumps({'elapsed_seconds': 60,'exit_code':1}))
        self.assertAlmostEqual(wave.charged_hours(self.root,ledger),34+1/60)

    def test_deadline_and_stop(self):
        self.assertEqual(wave.datetime.fromtimestamp(wave.DEADLINE, wave.ZoneInfo('America/Chicago')).hour, 10)
        self.assertEqual(wave.datetime.fromtimestamp(wave.DEADLINE, wave.ZoneInfo('America/Chicago')).utcoffset().total_seconds(), -21600)
        with patch.object(wave, 'eligible') as eligible:
            wave.tick(self.root, {'staged':{}}, wave.DEADLINE)
            eligible.assert_not_called()
            (self.root/'monitor/STOP_SECOND_WAVE').touch()
            wave.tick(self.root, {'staged':{}}, wave.DEADLINE-1)
            eligible.assert_not_called()

    def test_gates_require_complete_results(self):
        base = [.4,3.5,.8,-3]
        def feature(root,step):
            return base if step==wave.BASE else None
        with patch.object(wave,'accepted',return_value=True), patch.object(wave,'features',side_effect=feature):
            choices=wave.eligible(self.root)
            self.assertEqual(choices[:4],['foundlo','foundhi','broodlo','broodhi'])
            self.assertIn('rel_hazard_s41',choices)
            self.assertNotIn('rel_comp_s41',choices)
            self.assertNotIn('fl_bl',choices)
        def feature2(root,step):
            return [.5,3.5,.8,-3] if step.startswith('step34_') else base
        with patch.object(wave,'accepted',return_value=True), patch.object(wave,'features',side_effect=feature2):
            self.assertIn('fh_bh',wave.eligible(self.root))
            self.assertNotIn('seed43',wave.eligible(self.root))

    def test_stage_is_idempotent_and_verifies_frozen_source(self):
        source = self.root/'runs'/wave.BASE
        cfg = source/'code/config/overlays'/f'grid54_{wave.BASE}.json'
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({'population_model':{'pinned_sites':{},'nest_likelihood':{'broods':{'values':[2.5]*3}}},'run_names':{}}))
        sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
        (source/'CODE_MANIFEST.sha256').write_text(sha+'  ./config/overlays/'+cfg.name+'\n')
        (source/'grid54_fit.sh').write_text('# frozen fit')
        (source/'run_candidate.sh').write_text(f'HOUFIN_MAP_SEED=41 {wave.BASE}')
        wave.stage(self.root,'broodlo')
        wave.stage(self.root,'broodlo')
        self.assertEqual(len(list((self.root/'queue/pending').glob('*.sh'))),1)
        child = json.loads((self.root/'runs/step34_broodlo/code/config/overlays/grid54_step34_broodlo.json').read_text())
        self.assertEqual(child['population_model']['nest_likelihood']['broods']['values'],[2.0]*3)
        wave.stage(self.root,'rel_hazard_natdep_s42')
        combined = self.root/'runs/step34_rel_hazard_natdep_s42'
        child = json.loads((combined/'code/config/overlays/grid54_step34_rel_hazard_natdep_s42.json').read_text())
        self.assertTrue(child['population_model']['journey']['distance_hazard'])
        self.assertTrue(child['population_model']['native_departure']['enabled'])
        self.assertIn('HOUFIN_MAP_SEED=42',(combined/'run_candidate.sh').read_text())
        cfg.write_text('tampered')
        with self.assertRaisesRegex(ValueError,'source mismatch'):
            wave.stage(self.root,'foundhi')

    def test_v2_contract_rejects_changed_observations(self):
        import pickle
        a=self.root/'input_a'; b=self.root/'input_b'
        a.mkdir(); b.mkdir()
        def save(path,obj):
            with path.open('wb') as stream: pickle.dump(obj,stream)
        m={'observed_results':[1,2],'land_rows':[0,1],
           'z_kernel_contract':{'kernel':'ruzicka','latent_dim':64},'ingest_id':'a'}
        save(a/'metadata.pkl',m)
        m=dict(m,ingest_id='b');save(b/'metadata.pkl',m)
        cfg=self.root/'runs/step3_native_v2/code/config/overlays/grid54_step3_native_v2.json'
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({'input_dir':str(b)}))
        _,receipt=wave.v2_contract(self.root,{'input_dir':str(a)})
        self.assertTrue(receipt['unchanged_observations_grid_dynamics'])
        m['observed_results']=[1,3];save(b/'metadata.pkl',m)
        with self.assertRaisesRegex(ValueError,'survey/grid/dynamics mismatch'):
            wave.v2_contract(self.root,{'input_dir':str(a)})

    def test_blocked_arm_does_not_block_other_work(self):
        def stage(root,v):
            if v=='foundlo':raise ValueError('bad source')
            return {'step':'step34_'+v}
        ledger={'staged':{}}
        with patch.object(wave,'eligible',return_value=['foundlo','foundhi']), patch.object(wave,'stage',side_effect=stage):
            wave.tick(self.root,ledger,wave.DEADLINE-1)
        self.assertIn('step34_foundhi',ledger['staged'])
        self.assertIn('step34_foundlo',ledger['blocked'])


if __name__=='__main__': unittest.main()

"""CPU-only supervisor decisions; never touches services, network or GPUs."""
import importlib.util
from pathlib import Path
import sys
import unittest

spec=importlib.util.spec_from_file_location('recovery',Path(__file__).with_name('candidate_recovery_policy.py'))
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


class RecoveryTests(unittest.TestCase):
    def setUp(self):self.policy=module.RecoveryPolicy()
    def step(self,now=1000,**kwargs):
        args=dict(now=now,invocation='one',state='active',age=100,healthy=False,metrics=None,available=20*1024**3)
        args.update(kwargs);return self.policy.decide(**args)
    def test_startup_grace(self):
        for i in range(20):self.assertIsNone(self.step(now=1000+i*10,age=100+i*10))
    def test_health_recovery_and_cooldown(self):
        self.step(healthy=True)
        for i in range(5):self.assertIsNone(self.step(now=1010+i*10))
        self.assertEqual(self.step(now=1060)[0],'restart')
        for i in range(20):self.assertIsNone(self.step(now=1070+i*10))
    def test_failed_waits_for_memory(self):
        self.assertIsNone(self.step(state='failed',available=15*1024**3))
        self.assertIsNone(self.step(now=1299,state='failed'))
        self.assertIsNone(self.step(now=1300,state='failed',available=15*1024**3))
        self.assertEqual(self.step(now=1301,state='failed')[0],'start')
    def test_manual_stop_is_respected(self):
        self.assertIsNone(self.step(state='inactive',age=9999))
    def test_inflight_stall_and_progress(self):
        m={'num_requests_running':1,'generation_tokens_total':5}
        self.step(now=1000,healthy=True,metrics=m)
        self.assertIsNone(self.step(now=1599,healthy=True,metrics=m))
        self.assertEqual(self.step(now=1600,healthy=True,metrics=m)[0],'restart')
        p=module.RecoveryPolicy()
        for i in range(100):
            self.assertIsNone(p.decide(now=1000+i*30,invocation='x',state='active',age=1000,
                healthy=True,metrics={'num_requests_running':1,'generation_tokens_total':i},available=20*1024**3))
    def test_restart_counts_own_memory(self):
        self.step(healthy=True)
        for i in range(5):self.assertIsNone(self.step(now=1010+i*10,available=12*1024**3,reclaimable=12*1024**3))
        self.assertEqual(self.step(now=1060,available=12*1024**3,reclaimable=12*1024**3)[0],'restart')
        p=module.RecoveryPolicy()
        p.decide(now=1000,invocation='y',state='active',age=100,healthy=True,metrics=None,available=12*1024**3)
        for i in range(10):
            self.assertIsNone(p.decide(now=1010+i*10,invocation='y',state='active',age=100,healthy=False,
                                       metrics=None,available=12*1024**3))
    def test_failed_ignores_reclaimable(self):
        self.assertIsNone(self.step(state='failed',available=10*1024**3,reclaimable=12*1024**3))
        self.assertIsNone(self.step(now=1301,state='failed',available=10*1024**3,reclaimable=12*1024**3))
    def test_new_process_gets_startup_grace(self):
        self.step(healthy=True)
        for i in range(5):self.step(now=1010+i*10)
        self.assertIsNone(self.step(now=1060,invocation='two',age=1))


if __name__=='__main__':unittest.main()

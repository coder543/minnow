"""Synthetic pressure traces: exercise policy without stressing the machine."""
import unittest
from memory_pressure import GIB, MIB, Guard, Limits


def sample(available=100, used=28, pages=100, stalled=0):
    return {'available_bytes': available * GIB, 'used_bytes': used * GIB,
            'pswpout': pages, 'pswpin': 0, 'full_stall_us': stalled}


class MemoryGuardTests(unittest.TestCase):
    def test_incidental_swap_and_transient_reserve_dip_are_allowed(self):
        guard = Guard(sample(), Limits(), page_size=4096)
        self.assertIsNone(guard.check(sample(pages=102), 1))
        self.assertIsNone(guard.check(sample(available=19, pages=102), 2))
        self.assertIsNone(guard.check(sample(pages=102), 2.5))
        self.assertIsNone(guard.check(sample(pages=102), 4))
        self.assertEqual(guard.metrics['swap_out_bytes'], 8192)

    def test_sustained_headroom_and_growth_limits(self):
        for observed, reason in [(sample(available=19), 'system memory reserve reached'),
                                 (sample(used=69), 'maximum allowed system-memory growth exceeded')]:
            guard = Guard(sample(), Limits())
            self.assertIsNone(guard.check(observed, 1))
            self.assertEqual(guard.check(observed, 2), reason)

    def test_critical_reserve_and_large_swap_abort_immediately(self):
        guard = Guard(sample(), Limits(), page_size=4096)
        self.assertEqual(guard.check(sample(available=3), .1), 'critical system-memory reserve reached')
        guard = Guard(sample(), Limits(), page_size=4096)
        self.assertEqual(guard.check(sample(pages=100 + 257 * MIB // 4096), .1), 'swap-out byte budget exceeded')

    def test_sustained_swap_rate(self):
        guard = Guard(sample(), Limits(max_swap_mib=1024), page_size=4096)
        self.assertIsNone(guard.check(sample(pages=100 + 80 * MIB // 4096), 1))
        self.assertEqual(guard.check(sample(pages=100 + 160 * MIB // 4096), 2), 'sustained swap-out rate exceeded')

    def test_psi_stalls_work_without_swap_and_missing_psi_is_allowed(self):
        guard = Guard(sample(), Limits())
        self.assertIsNone(guard.check(sample(stalled=300000), 1))
        self.assertIsNone(guard.check(sample(stalled=600000), 2))
        self.assertEqual(guard.check(sample(stalled=3300000), 11), 'sustained memory reclaim stalls')
        guard = Guard(sample(stalled=None), Limits())
        self.assertIsNone(guard.check(sample(stalled=None), 2))

    def test_bulk_allocation_stall_is_not_sustained_pressure(self):
        guard = Guard(sample(), Limits())
        self.assertIsNone(guard.check(sample(stalled=300000), 1))
        self.assertIsNone(guard.check(sample(stalled=600000), 2))
        for second in range(3, 15):
            self.assertIsNone(guard.check(sample(stalled=600000), second))

    def test_invalid_limits(self):
        for limits in [Limits(max_swap_mib=float('nan')), Limits(grace_seconds=-1), Limits(window_seconds=0)]:
            with self.assertRaises(ValueError):
                limits.validate()


if __name__ == '__main__':
    unittest.main()

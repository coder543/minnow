"""Memory-pressure policy, separated from process control for synthetic testing."""
from collections import deque
from dataclasses import asdict, dataclass
import math
import os

MIB = 1024 ** 2
GIB = 1024 ** 3


@dataclass
class Limits:
    reserve_gib: float = 20
    max_growth_gib: float = 40
    critical_reserve_gib: float = 4
    grace_seconds: float = 1
    max_swap_mib: float = 256
    max_swap_rate_mib: float = 64
    max_stall_percent: float = 20
    stall_grace_seconds: float = 10
    window_seconds: float = 5

    def validate(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('memory limits must be finite and nonnegative')
        if self.max_growth_gib == 0 or self.window_seconds < 1 or self.max_stall_percent > 100:
            raise ValueError('growth must be positive, window >= 1 second, and stall percent <= 100')
        if self.critical_reserve_gib > self.reserve_gib:
            raise ValueError('critical reserve must not exceed the normal reserve')


class Guard:
    def __init__(self, baseline, limits, page_size=None):
        self.baseline = baseline
        self.limits = limits
        self.page_size = page_size or os.sysconf('SC_PAGE_SIZE')
        self.history = deque([(0.0, baseline)])
        self.exceeded_since = {}
        self.metrics = {}
        self.peak_used = baseline['used_bytes']
        self.peak_swap_rate = 0.0
        self.peak_stall_percent = 0.0

    def check(self, sample, elapsed):
        self.peak_used = max(self.peak_used, sample['used_bytes'])
        self.history.append((elapsed, sample))
        while len(self.history) > 2 and self.history[1][0] <= elapsed - self.limits.window_seconds:
            self.history.popleft()
        start, prior = self.history[0]
        seconds = elapsed - start
        total_swap = max(0, sample['pswpout'] - self.baseline['pswpout']) * self.page_size
        # Do not extrapolate a 100 ms burst into a sustained rate.
        swap_rate = max(0, sample['pswpout'] - prior['pswpout']) * self.page_size / seconds if seconds >= 1 else 0.0
        stall = 0.0
        if seconds >= 1 and sample.get('full_stall_us') is not None and prior.get('full_stall_us') is not None:
            stall = max(0, sample['full_stall_us'] - prior['full_stall_us']) / (seconds * 1e6) * 100
        self.peak_swap_rate = max(self.peak_swap_rate, swap_rate)
        self.peak_stall_percent = max(self.peak_stall_percent, stall)
        self.metrics = {'swap_out_bytes': total_swap, 'swap_out_bytes_per_second': swap_rate,
                        'full_stall_percent': stall, 'window_seconds': seconds}
        if sample['available_bytes'] < self.limits.critical_reserve_gib * GIB:
            return 'critical system-memory reserve reached'
        if total_swap > self.limits.max_swap_mib * MIB:
            return 'swap-out byte budget exceeded'
        conditions = {
            'system memory reserve reached': sample['available_bytes'] < self.limits.reserve_gib * GIB,
            'maximum allowed system-memory growth exceeded': sample['used_bytes'] - self.baseline['used_bytes'] > self.limits.max_growth_gib * GIB,
            'sustained swap-out rate exceeded': swap_rate > self.limits.max_swap_rate_mib * MIB,
            'sustained memory reclaim stalls': stall > self.limits.max_stall_percent,
        }
        for reason, exceeded in conditions.items():
            if exceeded:
                since = self.exceeded_since.setdefault(reason, elapsed)
                grace = self.limits.stall_grace_seconds if reason == 'sustained memory reclaim stalls' else self.limits.grace_seconds
                if elapsed - since >= grace:
                    return reason
            else:
                self.exceeded_since.pop(reason, None)
        return None

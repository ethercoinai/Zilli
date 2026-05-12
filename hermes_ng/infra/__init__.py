import numpy as np
from typing import List


class LayoutAwareDispatcher:
    def dispatch(self, data: List, num_workers: int) -> List[List]:
        chunks = [[] for _ in range(num_workers)]
        for i, item in enumerate(data):
            chunks[i % num_workers].append(item)
        return chunks


class LengthElasticController:
    def __init__(self):
        self.current_cap = 8192
        self.parallel_mode = "dp"
        self.dispatcher = LayoutAwareDispatcher()

    def adapt(self, observed_lengths: List[int]):
        if not observed_lengths:
            return

        p95_len = float(np.percentile(observed_lengths, 95))
        p99_len = float(np.percentile(observed_lengths, 99))

        if p95_len > self.current_cap * 0.9:
            new_cap = min(int(p95_len * 1.2), 131072)
            self.current_cap = new_cap

        if p99_len > 32768 and self.parallel_mode == "dp":
            self._switch_to_mp()

        if p99_len < 16384 and self.parallel_mode == "mp":
            self._switch_to_dp()

    def _switch_to_mp(self):
        self.parallel_mode = "mp"

    def _switch_to_dp(self):
        self.parallel_mode = "dp"

    def get_config(self) -> dict:
        return {
            "current_cap": self.current_cap,
            "parallel_mode": self.parallel_mode,
        }

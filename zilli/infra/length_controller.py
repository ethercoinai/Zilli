from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np


class LayoutAwareDispatcher:
    def __init__(self, topology: Optional[Dict[str, Any]] = None):
        self.topology = topology or {}
        self.num_numa_nodes = self.topology.get("num_numa_nodes", 1)
        self.ranks_per_node = self.topology.get("ranks_per_node", 8)

    def dispatch(self, data: List, num_workers: int) -> List[List]:
        chunks = [[] for _ in range(num_workers)]
        for i, item in enumerate(data):
            chunks[i % num_workers].append(item)
        return chunks

    def dispatch_layout_aware(self, data: List, num_workers: int,
                              sequence_lengths: List[int]) -> List[List]:
        if not data or not sequence_lengths:
            return self.dispatch(data, num_workers)

        indexed = list(zip(data, sequence_lengths))
        indexed.sort(key=lambda x: x[1], reverse=True)

        worker_loads = [0] * num_workers
        chunks = [[] for _ in range(num_workers)]
        for item, length in indexed:
            target = int(np.argmin(worker_loads))
            chunks[target].append(item)
            worker_loads[target] += length
        return chunks

    def estimate_num_workers(self, total_tokens: int, max_context: int = 8192) -> int:
        base = max(1, total_tokens // max_context)
        return base * self.num_numa_nodes


class LengthElasticController:
    def __init__(self, history_size: int = 100):
        self.current_cap = 8192
        self.parallel_mode = "dp"
        self.dispatcher = LayoutAwareDispatcher()
        self._history: deque = deque(maxlen=history_size)
        self._cap_history: deque = deque(maxlen=1000)
        self._mode_changes: int = 0
        self._growth_factor = 1.2
        self._shrink_factor = 0.85
        self._max_cap = 131072
        self._min_cap = 1024

    def adapt(self, observed_lengths: List[int]):
        if not observed_lengths:
            return

        self._history.extend(observed_lengths)

        p95_len = float(np.percentile(observed_lengths, 95))
        p99_len = float(np.percentile(observed_lengths, 99))
        max_len = float(max(observed_lengths))

        if p95_len > self.current_cap * 0.85:
            target = max(p95_len * self._growth_factor, max_len * 1.05)
            new_cap = min(int(target), self._max_cap)
            self.current_cap = new_cap
        elif p99_len < self.current_cap * 0.3 and len(self._history) > 20:
            new_cap = max(int(self.current_cap * self._shrink_factor), self._min_cap)
            if new_cap < self.current_cap:
                self.current_cap = new_cap

        if p99_len > 32768 and self.parallel_mode == "dp":
            self._switch_to_mp()
        elif p99_len < 16384 and self.parallel_mode == "mp":
            self._switch_to_dp()

        self._cap_history.append(self.current_cap)

    def _switch_to_mp(self):
        if self.parallel_mode != "mp":
            self.parallel_mode = "mp"
            self._mode_changes += 1

    def _switch_to_dp(self):
        if self.parallel_mode != "dp":
            self.parallel_mode = "dp"
            self._mode_changes += 1

    def get_config(self) -> dict:
        return {
            "current_cap": self.current_cap,
            "parallel_mode": self.parallel_mode,
            "mode_changes": self._mode_changes,
            "history_samples": len(self._history),
        }

    def get_stats(self) -> Dict[str, Any]:
        arr = np.array(list(self._history)) if self._history else np.array([0])
        return {
            "current_cap": self.current_cap,
            "parallel_mode": self.parallel_mode,
            "mode_changes": self._mode_changes,
            "p50_length": float(np.percentile(arr, 50)),
            "p95_length": float(np.percentile(arr, 95)),
            "p99_length": float(np.percentile(arr, 99)),
            "max_length": float(arr.max()),
            "min_length": float(arr.min()),
            "mean_length": float(arr.mean()),
            "history_samples": len(self._history),
        }


__all__ = ["LengthElasticController", "LayoutAwareDispatcher"]

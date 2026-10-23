import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .strategies import STRATEGIES


def default_ranges():
    return [[0.02, "exact"], [0.05, "restricted"], [1.01, "adaptive_post"]]


@dataclass
class Thresholds:
    # each entry is [upper bound on estimated selectivity, strategy]; first match wins
    ranges: list = field(default_factory=default_ranges)
    base_nprobe: int = 16
    min_matches: int = 40
    coverage: float = 0.2

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: data[k] for k in asdict(cls()) if k in data})

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class Planner:
    def __init__(self, index, thresholds=None):
        self.index = index
        self.t = thresholds or Thresholds()

    def choose(self, f):
        est = self.index.stats.selectivity(f)
        for upper, name in self.t.ranges:
            if est < upper:
                return name, est
        return self.t.ranges[-1][1], est

    def search(self, query, f, k=10):
        name, est = self.choose(f)
        rows, scores = STRATEGIES[name](
            self.index, query, f, k,
            selectivity=est,
            base_nprobe=self.t.base_nprobe,
            min_matches=self.t.min_matches,
            coverage=self.t.coverage,
        )
        return rows, scores, name

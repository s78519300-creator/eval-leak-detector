"""eval-leak-detector: find benchmark items that leaked into training data."""

__version__ = "0.1.0"

from .detector import CorpusIndex, ItemResult, Report, Thresholds, scan  # noqa: E402
from .loaders import Record, iter_corpus, load_eval  # noqa: E402

__all__ = ["CorpusIndex", "ItemResult", "Report", "Thresholds", "scan", "Record", "iter_corpus", "load_eval"]

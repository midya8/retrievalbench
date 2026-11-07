"""Offline retrieval evaluation with exact BM25 and golden metrics."""

from .bm25 import BM25Index
from .metrics import evaluate_run

__all__ = ["BM25Index", "evaluate_run"]
__version__ = "0.1.0"

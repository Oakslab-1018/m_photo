"""Geolocation domain package.

This package re-exports the former ``reasoning`` public API (clue data
structures, reasoning engine, hypothesis manager and knowledge graph) so
existing consumers keep the same import surface after the bounded migration.

``AnalysisPipeline`` is intentionally not re-exported here; import it from
``backend.app.domain.geolocation.analysis`` so the analysis subpackage boundary
stays explicit.
"""
from .clue import Clue, DIMENSIONS, clues_to_dicts, dicts_to_clues
from .engine import ReasoningEngine, CountryScore
from .hypotheses import HypothesisManager, HypothesisAction
from . import knowledge_graph

__all__ = [
    "Clue", "DIMENSIONS", "clues_to_dicts", "dicts_to_clues",
    "ReasoningEngine", "CountryScore",
    "HypothesisManager", "HypothesisAction",
    "knowledge_graph",
]
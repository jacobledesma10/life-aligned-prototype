"""
Knowledge Graph Schema
======================
Node types, edge types, and ontology mappings.

Ontology alignment:
  - SSN/SOSA  : W3C Semantic Sensor Network (https://www.w3.org/TR/vocab-ssn/)
  - AGROVOC   : FAO agricultural thesaurus  (https://agrovoc.fao.org)

Neo4j-ready: every node/edge is a plain dict so the builder can write
Cypher CREATE statements without modification when a driver is wired in.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# AGROVOC concept registry  (URI → label → parent)
# A small curated subset relevant to this sensor domain.
# Full AGROVOC SPARQL endpoint: https://agrovoc.fao.org/sparql
# ---------------------------------------------------------------------------
AGROVOC_CONCEPTS: dict[str, dict] = {
    # Soil properties
    "agrovoc:c_7167":  {"label": "soil",              "parent": None},
    "agrovoc:c_16208": {"label": "soil moisture",     "parent": "agrovoc:c_7167"},
    "agrovoc:c_5775":  {"label": "soil pH",           "parent": "agrovoc:c_7167"},
    "agrovoc:c_5366":  {"label": "nitrogen",          "parent": "agrovoc:c_7167"},
    "agrovoc:c_7657":  {"label": "soil temperature",  "parent": "agrovoc:c_7167"},

    # Interventions
    "agrovoc:c_3914":  {"label": "irrigation",        "parent": None},
    "agrovoc:c_3081":  {"label": "fertilization",     "parent": None},
    "agrovoc:c_5684":  {"label": "pH adjustment",     "parent": None},
    "agrovoc:c_1253":  {"label": "crop management",   "parent": None},

    # Soil health
    "agrovoc:c_35023": {"label": "soil health",       "parent": "agrovoc:c_7167"},
    "agrovoc:c_2561":  {"label": "soil degradation",  "parent": "agrovoc:c_7167"},
    "agrovoc:c_6174":  {"label": "regenerative agriculture", "parent": None},
}

# SSN/SOSA class URIs used as node type labels
SSN = {
    "Sensor":        "sosa:Sensor",
    "Observation":   "sosa:Observation",
    "ObservableProperty": "sosa:ObservableProperty",
    "Result":        "sosa:Result",
    "FeatureOfInterest": "sosa:FeatureOfInterest",
}

# Sensor dimension → AGROVOC concept mapping
SENSOR_TO_AGROVOC: dict[str, str] = {
    "soil_moisture": "agrovoc:c_16208",
    "soil_ph":       "agrovoc:c_5775",
    "nitrogen":      "agrovoc:c_5366",
    "temperature":   "agrovoc:c_7657",
}

# Action label → AGROVOC concept mapping
ACTION_TO_AGROVOC: dict[str, str] = {
    "irrigate":   "agrovoc:c_3914",
    "fertilize":  "agrovoc:c_3081",
    "adjust pH":  "agrovoc:c_5684",
    "rest":       "agrovoc:c_6174",
    "intervene":  "agrovoc:c_1253",
    "no action":  "agrovoc:c_35023",
}

# Optimal ranges (mirrored from main.py for graph annotation)
OPTIMAL_RANGES: dict[str, tuple[float, float]] = {
    "soil_moisture": (0.25, 0.45),
    "soil_ph":       (6.0,  7.0),
    "nitrogen":      (0.35, 0.60),
    "temperature":   (15.0, 24.0),
}


# ---------------------------------------------------------------------------
# Node dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SensorNode:
    """sosa:Sensor — a physical or virtual soil sensor."""
    node_id:    str
    label:      str          # e.g. "Greenhouse A — moisture probe"
    ssn_type:   str = SSN["Sensor"]
    location:   Optional[str] = None   # future: WKT geometry from PostGIS
    greenhouse: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "node_id":    self.node_id,
            "type":       "Sensor",
            "ssn_type":   self.ssn_type,
            "label":      self.label,
            "location":   self.location,
            "greenhouse": self.greenhouse,
        }


@dataclass
class ObservationNode:
    """sosa:Observation — a single sensor reading at a point in time."""
    node_id:    str
    sensor_id:  str
    timestamp:  str
    dimension:  str          # soil_moisture | soil_ph | nitrogen | temperature
    value:      float
    unit:       str
    agrovoc_id: str          # mapped concept URI
    in_range:   bool         # whether value is within OPTIMAL_RANGES

    def to_dict(self) -> dict:
        return {
            "node_id":    self.node_id,
            "type":       "Observation",
            "ssn_type":   SSN["Observation"],
            "sensor_id":  self.sensor_id,
            "timestamp":  self.timestamp,
            "dimension":  self.dimension,
            "value":      self.value,
            "unit":       self.unit,
            "agrovoc_id": self.agrovoc_id,
            "in_range":   self.in_range,
        }


@dataclass
class ConceptNode:
    """AGROVOC concept node — domain ontology term."""
    node_id:  str     # e.g. "agrovoc:c_16208"
    label:    str
    parent:   Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "type":    "Concept",
            "label":   self.label,
            "parent":  self.parent,
        }


@dataclass
class ActionNode:
    """A triggered intervention with gate metadata."""
    node_id:     str
    label:       str          # irrigate | fertilize | ...
    agrovoc_id:  str
    timestamp:   str
    necessity:   float
    alignment:   float
    risk:        float
    gate_open:   bool

    def to_dict(self) -> dict:
        return {
            "node_id":    self.node_id,
            "type":       "Action",
            "label":      self.label,
            "agrovoc_id": self.agrovoc_id,
            "timestamp":  self.timestamp,
            "necessity":  round(self.necessity, 4),
            "alignment":  round(self.alignment, 4),
            "risk":       round(self.risk, 4),
            "gate_open":  self.gate_open,
        }


# ---------------------------------------------------------------------------
# Edge dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TemporalEdge:
    """An edge with temporal validity metadata — maps to Neo4j relationship."""
    src:        str   # node_id
    dst:        str   # node_id
    rel_type:   str   # OBSERVED | HAS_CONCEPT | TRIGGERED_ACTION | BROADER
    valid_from: str   # ISO timestamp
    valid_to:   Optional[str] = None   # None = still valid
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "src":        self.src,
            "dst":        self.dst,
            "rel_type":   self.rel_type,
            "valid_from": self.valid_from,
            "valid_to":   self.valid_to,
            **self.properties,
        }

"""
Knowledge Graph Builder
=======================
Converts soil CSV rows + action decisions into a NetworkX graph using
the node/edge schema defined in schema.py.

Neo4j migration path:
  Replace the `_G` networkx.MultiDiGraph with a neo4j.GraphDatabase driver
  and call `session.run(cypher, **node.to_dict())` — the dict format is
  already Cypher-parameter-compatible.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import networkx as nx
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge_graph.schema import (
    AGROVOC_CONCEPTS,
    SENSOR_TO_AGROVOC,
    ACTION_TO_AGROVOC,
    OPTIMAL_RANGES,
    SensorNode,
    ObservationNode,
    ConceptNode,
    ActionNode,
    TemporalEdge,
)
from ingestion.load_soil_data import load_soil_data
from world_model.world_model import WorldModel
from action.soil_env import life_reward
from gating.action_potential_gate import ActionPotentialGate

_WORLD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "world_model.pt")
_ACTIONS = [0, 1, 2, 3, 4, 5]
_ACTION_LABEL = {
    0: "no action", 1: "irrigate", 2: "rest",
    3: "intervene", 4: "fertilize", 5: "adjust pH",
}
_ACTION_RISK = {0: 0.1, 1: 0.2, 2: 0.1, 3: 0.5, 4: 0.2, 5: 0.3}

# Greenhouse definitions — mirrors interface/app.py
_GREENHOUSES = {
    "Greenhouse A": {"lat": 36.7783, "lon": -119.4179},
    "Greenhouse B": {"lat": 36.7810, "lon": -119.4148},
    "Greenhouse C": {"lat": 36.7755, "lon": -119.4200},
}
_SENSOR_DIMS = ["soil_moisture", "soil_ph", "nitrogen", "temperature"]
_SENSOR_UNITS = {
    "soil_moisture": "m³/m³",
    "soil_ph":       "pH",
    "nitrogen":      "g/kg",
    "temperature":   "°C",
}


class KnowledgeGraphBuilder:
    """
    Incrementally builds a knowledge graph from soil observations.

    Usage:
        builder = KnowledgeGraphBuilder()
        builder.build_from_csv()          # batch build
        G = builder.graph                 # networkx.MultiDiGraph
        cypher = builder.to_cypher()      # Neo4j migration helper
    """

    def __init__(self, greenhouse: str = "Greenhouse A") -> None:
        self.greenhouse = greenhouse
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._obs_counter = 0
        self._action_counter = 0

        # Load world model + gate
        self._wm = WorldModel()
        if os.path.isfile(_WORLD_MODEL_PATH):
            self._wm.load(_WORLD_MODEL_PATH)
        self._gate = ActionPotentialGate(
            necessity_thresh=0.5, alignment_thresh=-0.1, risk_thresh=0.4
        )

        # Seed the graph with ontology concept nodes + sensor nodes
        self._add_concept_nodes()
        self._add_sensor_nodes()

    # ------------------------------------------------------------------
    # Graph seeding
    # ------------------------------------------------------------------

    def _add_concept_nodes(self) -> None:
        """Add all AGROVOC concept nodes and BROADER hierarchy edges."""
        for uri, meta in AGROVOC_CONCEPTS.items():
            node = ConceptNode(
                node_id=uri,
                label=meta["label"],
                parent=meta["parent"],
            )
            self.G.add_node(uri, **node.to_dict())

        # BROADER edges (child → parent in AGROVOC hierarchy)
        for uri, meta in AGROVOC_CONCEPTS.items():
            if meta["parent"]:
                edge = TemporalEdge(
                    src=uri,
                    dst=meta["parent"],
                    rel_type="BROADER",
                    valid_from="2000-01-01T00:00:00",
                )
                self.G.add_edge(uri, meta["parent"], **edge.to_dict())

    def _add_sensor_nodes(self) -> None:
        """Add one sensor node per dimension per greenhouse."""
        gh_meta = _GREENHOUSES.get(self.greenhouse, {})
        location = (
            f"POINT({gh_meta['lon']} {gh_meta['lat']})"
            if gh_meta else None
        )
        for dim in _SENSOR_DIMS:
            sensor_id = f"sensor:{self.greenhouse}:{dim}"
            node = SensorNode(
                node_id=sensor_id,
                label=f"{self.greenhouse} — {dim.replace('_', ' ')}",
                location=location,
                greenhouse=self.greenhouse,
            )
            self.G.add_node(sensor_id, **node.to_dict())

            # Link sensor to its observable AGROVOC concept
            concept_uri = SENSOR_TO_AGROVOC[dim]
            edge = TemporalEdge(
                src=sensor_id,
                dst=concept_uri,
                rel_type="OBSERVES_CONCEPT",
                valid_from="2000-01-01T00:00:00",
            )
            self.G.add_edge(sensor_id, concept_uri, **edge.to_dict())

    # ------------------------------------------------------------------
    # Incremental observation ingestion
    # ------------------------------------------------------------------

    def add_observation_row(self, row) -> list[str]:
        """
        Process one CSV row: add 4 ObservationNodes (one per sensor dim)
        and optionally an ActionNode if the gate fires.

        Returns list of node_ids added.
        """
        ts = str(row.timestamp)
        x = np.array(
            [float(row.soil_moisture), float(row.soil_ph),
             float(row.nitrogen),      float(row.temperature)],
            dtype=np.float32,
        )
        added = []

        # --- 4 observation nodes ---
        obs_ids = []
        for i, dim in enumerate(_SENSOR_DIMS):
            self._obs_counter += 1
            obs_id = f"obs:{self._obs_counter:05d}:{dim}"
            val = float(x[i])
            lo, hi = OPTIMAL_RANGES[dim]
            node = ObservationNode(
                node_id=obs_id,
                sensor_id=f"sensor:{self.greenhouse}:{dim}",
                timestamp=ts,
                dimension=dim,
                value=val,
                unit=_SENSOR_UNITS[dim],
                agrovoc_id=SENSOR_TO_AGROVOC[dim],
                in_range=(lo <= val <= hi),
            )
            self.G.add_node(obs_id, **node.to_dict())
            added.append(obs_id)
            obs_ids.append(obs_id)

            # OBSERVED edge: sensor → observation
            self.G.add_edge(
                f"sensor:{self.greenhouse}:{dim}", obs_id,
                **TemporalEdge(
                    src=f"sensor:{self.greenhouse}:{dim}",
                    dst=obs_id,
                    rel_type="OBSERVED",
                    valid_from=ts,
                ).to_dict(),
            )
            # HAS_CONCEPT edge: observation → AGROVOC concept
            self.G.add_edge(
                obs_id, SENSOR_TO_AGROVOC[dim],
                **TemporalEdge(
                    src=obs_id,
                    dst=SENSOR_TO_AGROVOC[dim],
                    rel_type="HAS_CONCEPT",
                    valid_from=ts,
                ).to_dict(),
            )

        # --- World model lookahead + gate ---
        raw_scores = {a: life_reward(self._wm.predict(x, a)) for a in _ACTIONS}
        best_action = max(raw_scores, key=raw_scores.__getitem__)
        best_score  = raw_scores[best_action]
        no_act      = raw_scores[0]
        score_range = max(abs(best_score - no_act), 1e-6)
        necessity   = float(np.clip((best_score - no_act) / score_range, 0.0, 1.0))
        alignment   = float(np.clip(best_score / (abs(best_score) + 1.0), 0.0, 1.0))
        risk        = _ACTION_RISK[best_action]
        gate_open   = self._gate.allow_action(necessity, alignment, risk)

        if gate_open:
            self._action_counter += 1
            action_label = _ACTION_LABEL[best_action]
            action_id = f"action:{self._action_counter:04d}:{action_label.replace(' ', '_')}"
            action_node = ActionNode(
                node_id=action_id,
                label=action_label,
                agrovoc_id=ACTION_TO_AGROVOC.get(action_label, "agrovoc:c_1253"),
                timestamp=ts,
                necessity=necessity,
                alignment=alignment,
                risk=risk,
                gate_open=True,
            )
            self.G.add_node(action_id, **action_node.to_dict())
            added.append(action_id)

            # TRIGGERED_ACTION edges: each observation → action
            for obs_id in obs_ids:
                self.G.add_edge(
                    obs_id, action_id,
                    **TemporalEdge(
                        src=obs_id,
                        dst=action_id,
                        rel_type="TRIGGERED_ACTION",
                        valid_from=ts,
                        properties={
                            "necessity": round(necessity, 4),
                            "alignment": round(alignment, 4),
                            "risk":      round(risk, 4),
                        },
                    ).to_dict(),
                )

            # Action → AGROVOC concept edge
            concept_uri = ACTION_TO_AGROVOC.get(action_label, "agrovoc:c_1253")
            self.G.add_edge(
                action_id, concept_uri,
                **TemporalEdge(
                    src=action_id,
                    dst=concept_uri,
                    rel_type="MAPS_TO_CONCEPT",
                    valid_from=ts,
                ).to_dict(),
            )

        return added

    # ------------------------------------------------------------------
    # Batch build
    # ------------------------------------------------------------------

    def build_from_csv(self, max_rows: int = 300) -> None:
        """Process up to max_rows rows from the soil CSV."""
        df = load_soil_data()
        for _, row in df.head(max_rows).iterrows():
            self.add_observation_row(row)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        node_types = {}
        for _, data in self.G.nodes(data=True):
            t = data.get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        edge_types = {}
        for _, _, data in self.G.edges(data=True):
            t = data.get("rel_type", "unknown")
            edge_types[t] = edge_types.get(t, 0) + 1

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "node_types":  node_types,
            "edge_types":  edge_types,
        }

    # ------------------------------------------------------------------
    # Neo4j migration helper
    # ------------------------------------------------------------------

    def to_cypher(self) -> str:
        """
        Export the graph as Cypher CREATE statements.
        Drop into a Neo4j Browser session to recreate the full graph.
        """
        lines = ["// Auto-generated Cypher — paste into Neo4j Browser", ""]

        for node_id, data in self.G.nodes(data=True):
            ntype = data.get("type", "Node")
            props = ", ".join(
                f"{k}: {repr(v)}" for k, v in data.items() if v is not None
            )
            lines.append(f"CREATE (:{ntype} {{{props}}})")

        lines.append("")
        for src, dst, data in self.G.edges(data=True):
            rel = data.get("rel_type", "RELATED")
            props = {k: v for k, v in data.items()
                     if k not in ("src", "dst", "rel_type") and v is not None}
            prop_str = (
                " {" + ", ".join(f"{k}: {repr(v)}" for k, v in props.items()) + "}"
                if props else ""
            )
            lines.append(
                f"MATCH (a {{node_id: {repr(src)}}}), (b {{node_id: {repr(dst)}}})"
                f" CREATE (a)-[:{rel}{prop_str}]->(b)"
            )

        return "\n".join(lines)

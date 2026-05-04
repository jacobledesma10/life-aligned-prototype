"""
Knowledge Graph Visualizer
==========================
Renders the NetworkX graph as an interactive HTML file using pyvis.
Color-coded by node type, edge labels shown on hover.
"""

from __future__ import annotations

import os
import networkx as nx
from pyvis.network import Network

# Node type → (color, shape, size)
_NODE_STYLE: dict[str, tuple[str, str, int]] = {
    "Sensor":      ("#2196F3", "diamond", 28),   # blue
    "Observation": ("#4CAF50", "dot",     14),   # green
    "Action":      ("#F44336", "star",    24),   # red
    "Concept":     ("#FF9800", "square",  18),   # orange
}
_DEFAULT_STYLE = ("#9E9E9E", "dot", 12)

# Edge type → color
_EDGE_COLOR: dict[str, str] = {
    "OBSERVED":           "#90CAF9",
    "HAS_CONCEPT":        "#FFE082",
    "TRIGGERED_ACTION":   "#EF9A9A",
    "MAPS_TO_CONCEPT":    "#FFCC80",
    "OBSERVES_CONCEPT":   "#80DEEA",
    "BROADER":            "#CE93D8",
}
_DEFAULT_EDGE_COLOR = "#BDBDBD"


def _node_label(data: dict) -> str:
    ntype = data.get("type", "")
    if ntype == "Sensor":
        return data.get("label", data["node_id"])
    if ntype == "Observation":
        dim   = data.get("dimension", "")
        val   = data.get("value", 0)
        ts    = data.get("timestamp", "")[:10]
        flag  = "" if data.get("in_range") else " ⚠"
        return f"{dim.replace('_', ' ')}\n{val:.3f}{flag}\n{ts}"
    if ntype == "Action":
        return f"{data.get('label','?').upper()}\n{data.get('timestamp','')[:10]}"
    if ntype == "Concept":
        return data.get("label", data["node_id"])
    return data.get("node_id", "?")


def _node_title(data: dict) -> str:
    """Hover tooltip."""
    lines = [f"<b>{data.get('type','Node')}</b>"]
    for k, v in data.items():
        if k in ("node_id", "type") or v is None:
            continue
        lines.append(f"{k}: {v}")
    return "<br>".join(lines)


def render(
    G: nx.MultiDiGraph,
    output_path: str,
    title: str = "Soil Knowledge Graph",
    height: str = "820px",
) -> str:
    """
    Render the graph to an interactive HTML file.

    Args:
        G:           NetworkX MultiDiGraph from KnowledgeGraphBuilder
        output_path: Absolute path for the output .html file
        title:       Page title shown in the browser tab
        height:      Canvas height string

    Returns:
        output_path (for chaining)
    """
    net = Network(
        height=height,
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        directed=True,
        notebook=False,
    )
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -12000,
          "centralGravity": 0.25,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.15
        },
        "maxVelocity": 40,
        "minVelocity": 0.5,
        "stabilization": { "iterations": 200 }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.15 },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } }
      }
    }
    """)

    # --- Add nodes ---
    for node_id, data in G.nodes(data=True):
        ntype = data.get("type", "")
        color, shape, size = _NODE_STYLE.get(ntype, _DEFAULT_STYLE)

        # Dim out-of-range observations
        if ntype == "Observation" and not data.get("in_range", True):
            color = "#FF5252"

        net.add_node(
            node_id,
            label=_node_label(data),
            title=_node_title(data),
            color=color,
            shape=shape,
            size=size,
            font={"size": 10, "color": "#e0e0e0"},
        )

    # --- Add edges ---
    for src, dst, data in G.edges(data=True):
        rel = data.get("rel_type", "")
        color = _EDGE_COLOR.get(rel, _DEFAULT_EDGE_COLOR)
        ts = data.get("valid_from", "")[:10]
        label = rel.replace("_", " ").lower() if rel in (
            "TRIGGERED_ACTION", "MAPS_TO_CONCEPT", "OBSERVES_CONCEPT", "BROADER"
        ) else ""

        net.add_edge(
            src, dst,
            title=f"{rel}<br>valid_from: {ts}",
            label=label,
            color=color,
            width=2 if rel == "TRIGGERED_ACTION" else 1,
            dashes=(rel == "BROADER"),
            font={"size": 8, "color": "#BDBDBD"},
        )

    # Legend as title block injected into the HTML
    legend_html = """
    <div style="position:fixed;top:10px;left:10px;background:rgba(0,0,0,0.7);
                padding:12px 16px;border-radius:8px;font-family:sans-serif;
                font-size:12px;color:#e0e0e0;z-index:9999;line-height:1.8">
      <b style="font-size:14px">Soil Knowledge Graph</b><br>
      <span style="color:#2196F3">&#9670;</span> Sensor &nbsp;
      <span style="color:#4CAF50">&#9679;</span> Observation &nbsp;
      <span style="color:#F44336">&#9733;</span> Action &nbsp;
      <span style="color:#FF9800">&#9632;</span> AGROVOC Concept<br>
      <span style="color:#FF5252">&#9679;</span> Out-of-range reading &nbsp;
      <span style="color:#CE93D8">- -</span> Broader (ontology)
    </div>
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net.save_graph(output_path)

    # Inject legend into saved HTML
    with open(output_path, "r") as f:
        html = f.read()
    html = html.replace("<body>", f"<body>{legend_html}", 1)
    with open(output_path, "w") as f:
        f.write(html)

    return output_path

"""
Knowledge Graph DAG (Prefect)
=============================
Prefect flow that builds and updates the knowledge graph from soil data.

Run directly:
    python3 src/knowledge_graph/dag.py

Or via Prefect UI after `prefect server start`:
    prefect deploy src/knowledge_graph/dag.py

Future: swap CSV source for PostGIS spatial queries using psycopg2 +
        ST_Within / ST_DWithin joins on greenhouse geometries.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

from knowledge_graph.builder import KnowledgeGraphBuilder
from knowledge_graph.visualize import render

_OUTPUT_HTML  = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "knowledge_graph.html")
_OUTPUT_CYPHER = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "knowledge_graph.cypher")

_GREENHOUSES = ["Greenhouse A", "Greenhouse B", "Greenhouse C"]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="build-graph", retries=1)
def build_graph(greenhouse: str, max_rows: int) -> KnowledgeGraphBuilder:
    logger = get_run_logger()
    logger.info(f"Building knowledge graph for {greenhouse} ({max_rows} rows)...")
    builder = KnowledgeGraphBuilder(greenhouse=greenhouse)
    builder.build_from_csv(max_rows=max_rows)
    stats = builder.stats()
    logger.info(
        f"Graph built — {stats['total_nodes']} nodes, "
        f"{stats['total_edges']} edges"
    )
    return builder


@task(name="render-graph")
def render_graph(builder: KnowledgeGraphBuilder, output_path: str) -> str:
    logger = get_run_logger()
    path = render(builder.G, output_path)
    logger.info(f"Graph rendered → {path}")
    return path


@task(name="export-cypher")
def export_cypher(builder: KnowledgeGraphBuilder, output_path: str) -> str:
    logger = get_run_logger()
    cypher = builder.to_cypher()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(cypher)
    logger.info(f"Cypher export saved → {output_path}")
    return output_path


@task(name="publish-stats-artifact")
def publish_stats(builder: KnowledgeGraphBuilder, greenhouse: str) -> None:
    stats = builder.stats()
    md = f"""## Knowledge Graph — {greenhouse}

| Metric | Value |
|---|---|
| Total nodes | {stats['total_nodes']} |
| Total edges | {stats['total_edges']} |

### Node types
| Type | Count |
|---|---|
""" + "\n".join(
        f"| {t} | {c} |" for t, c in sorted(stats["node_types"].items())
    ) + "\n\n### Edge types\n| Type | Count |\n|---|---|\n" + "\n".join(
        f"| {t} | {c} |" for t, c in sorted(stats["edge_types"].items())
    )

    create_markdown_artifact(
        key=f"kg-stats-{greenhouse.lower().replace(' ', '-')}",
        markdown=md,
        description=f"Knowledge graph stats for {greenhouse}",
    )


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="soil-knowledge-graph", log_prints=True)
def soil_knowledge_graph_flow(
    greenhouse: str = "Greenhouse A",
    max_rows: int = 100,
    export_neo4j: bool = True,
) -> dict:
    """
    Main Prefect flow — builds the soil knowledge graph for one greenhouse.

    Args:
        greenhouse:   Which greenhouse to process ("Greenhouse A/B/C")
        max_rows:     Number of CSV rows to process (100 = ~4 days of hourly data)
        export_neo4j: Whether to write a .cypher file for Neo4j import

    Returns:
        Dict with output file paths and graph stats.
    """
    builder = build_graph(greenhouse=greenhouse, max_rows=max_rows)
    html_path = render_graph(builder, _OUTPUT_HTML)

    cypher_path = None
    if export_neo4j:
        cypher_path = export_cypher(builder, _OUTPUT_CYPHER)

    publish_stats(builder, greenhouse)

    stats = builder.stats()
    print(f"\n  Nodes : {stats['total_nodes']}")
    print(f"  Edges : {stats['total_edges']}")
    print(f"  Types : {stats['node_types']}")
    print(f"\n  HTML  → {html_path}")
    if cypher_path:
        print(f"  Neo4j → {cypher_path}")

    return {
        "html":   html_path,
        "cypher": cypher_path,
        "stats":  stats,
    }


if __name__ == "__main__":
    soil_knowledge_graph_flow(
        greenhouse="Greenhouse A",
        max_rows=100,
        export_neo4j=True,
    )

"""
LLM Responder — wraps Claude API to generate natural language explanations.

If ANTHROPIC_API_KEY is not set, falls back to a structured demo response
so the interface works fully without an API key.
"""

from __future__ import annotations

import os

_ACTION_EFFECTS = {
    "irrigate":   "soil moisture +0.10",
    "rest":       "nitrogen +0.02 (microbial recovery)",
    "intervene":  "soil moisture -0.10 (drainage / aeration)",
    "fertilize":  "nitrogen +0.08",
    "adjust pH":  "pH nudged 0.05 toward 6.5",
    "no action":  "no direct change — natural drift continues",
}

_SYSTEM_PROMPT = """You are a life-aligned agronomic intelligence assistant embedded in a
regenerative soil monitoring system. You help farmers and agronomists understand
what the AI system is observing, why it recommends a specific action, and what
outcomes to expect.

Speak clearly and practically. Be concise — 3 to 5 sentences max unless the user
asks for more detail. Ground every response in the sensor data and knowledge
chunks provided. Never make up numbers not present in the context."""


def build_context(state: dict, action: str, scores: dict, rag_chunks: list[dict]) -> str:
    """Format sensor state, world model scores, and RAG chunks into a prompt context block."""
    lines = [
        "## Current Soil State",
        f"- Moisture:     {state['moisture']:.3f}  (optimal 0.25–0.45)",
        f"- pH:           {state['ph']:.2f}   (optimal 6.0–7.0)",
        f"- Nitrogen:     {state['nitrogen']:.3f}  (optimal 0.35–0.60 g/kg)",
        f"- Temperature:  {state['temperature']:.1f}°C (optimal 15–24°C)",
        "",
        "## World Model Action Scores (life-aligned reward)",
    ]
    for a_label, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " ← recommended" if a_label == action else ""
        lines.append(f"- {a_label:<12} {score:+.4f}{marker}")

    lines += [
        "",
        f"## Recommended Action: {action}",
        f"Expected effect: {_ACTION_EFFECTS.get(action, 'unknown')}",
        "",
        "## Relevant Agronomic Knowledge",
    ]
    for i, chunk in enumerate(rag_chunks, 1):
        lines.append(f"[{i}] (relevance={chunk['score']:.3f}) {chunk['text']}")

    return "\n".join(lines)


def _demo_response(state: dict, action: str, scores: dict, rag_chunks: list[dict]) -> str:
    """Structured fallback used when no API key is present."""
    best_score  = scores.get(action, 0.0)
    no_act      = scores.get("no action", 0.0)
    delta       = best_score - no_act
    effect      = _ACTION_EFFECTS.get(action, "")
    top_chunk   = rag_chunks[0]["text"][:120] + "…" if rag_chunks else ""

    lines = [
        f"**Recommended: {action.upper()}**",
        "",
        f"The world model scores *{action}* at **{best_score:+.4f}** — "
        f"**{delta:+.4f}** better than taking no action. "
        f"The predicted effect is: {effect}.",
    ]

    issues = []
    if state["moisture"] < 0.25:
        issues.append(f"moisture is low ({state['moisture']:.2f}, below 0.25)")
    if state["ph"] < 6.0 or state["ph"] > 7.0:
        issues.append(f"pH is outside optimal range ({state['ph']:.2f})")
    if state["nitrogen"] < 0.35:
        issues.append(f"nitrogen is low ({state['nitrogen']:.3f} g/kg)")
    if state["temperature"] < 15 or state["temperature"] > 24:
        issues.append(f"temperature is outside optimal range ({state['temperature']:.1f}°C)")

    if issues:
        lines.append("")
        lines.append("The system flagged: " + "; ".join(issues) + ".")

    if top_chunk:
        lines += ["", f"> {top_chunk}"]

    lines += [
        "",
        "*⚠ Demo mode — connect an Anthropic API key for full natural language responses.*",
    ]
    return "\n".join(lines)


def generate_response(
    state: dict,
    action: str,
    scores: dict,
    rag_chunks: list[dict],
    user_query: str = "",
) -> str:
    """
    Generate a natural language explanation.

    Uses Claude API if ANTHROPIC_API_KEY is set, otherwise returns a
    structured demo response.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    context = build_context(state, action, scores, rag_chunks)

    if not api_key:
        return _demo_response(state, action, scores, rag_chunks)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        user_content = context
        if user_query.strip():
            user_content += f"\n\n## User Question\n{user_query.strip()}"
        else:
            user_content += (
                "\n\n## Task\nExplain the current soil state, why the recommended action "
                "was chosen, and what outcome to expect. Be practical and concise."
            )

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text

    except Exception as exc:
        return (
            f"⚠ API call failed: {exc}\n\n"
            + _demo_response(state, action, scores, rag_chunks)
        )

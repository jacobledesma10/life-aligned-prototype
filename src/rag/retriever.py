"""
RAG Retriever — agronomic knowledge base for action justification.

Uses sentence-transformers (all-MiniLM-L6-v2, runs fully offline after
first download) to embed a small corpus of soil-health guidelines, then
FAISS for fast nearest-neighbour retrieval.

Usage:
    retriever = SoilRAGRetriever()
    results = retriever.query(state, action, top_k=2)
    for chunk in results:
        print(chunk["text"])
        print(chunk["score"])
"""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Agronomic knowledge corpus
# Each entry: {"text": <chunk>, "tags": [<action_tags>]}
# Tags are used to bias retrieval toward the triggered action, but the FAISS
# search is purely semantic — tags are metadata only.
# ---------------------------------------------------------------------------
_CORPUS = [
    # --- Soil Moisture / Irrigation ---
    {
        "text": (
            "Optimal soil moisture for most crops lies between 0.25 and 0.45 volumetric "
            "water content. Below 0.20, plants experience water stress, stomata close, and "
            "photosynthesis drops sharply. Irrigate when moisture falls below 0.25 to "
            "prevent yield loss."
        ),
        "tags": ["irrigate"],
    },
    {
        "text": (
            "Overwatering (soil moisture > 0.60) causes anaerobic conditions in the root "
            "zone, promoting denitrification and nitrogen loss. It also leaches soluble "
            "nutrients below the rooting depth. Allow soil to drain before irrigating again."
        ),
        "tags": ["irrigate", "rest"],
    },
    {
        "text": (
            "Drip irrigation applied at dawn reduces evaporation losses by up to 40% "
            "compared with midday application. Target the root zone directly; avoid wetting "
            "foliage to reduce fungal disease pressure."
        ),
        "tags": ["irrigate"],
    },
    {
        "text": (
            "Soil moisture sensors placed at 15 cm and 30 cm depth provide accurate "
            "estimates of plant-available water. A 3-reading moving average smooths out "
            "short-term fluctuations and gives a more reliable irrigation trigger signal."
        ),
        "tags": ["irrigate"],
    },

    # --- Rest / Reduced Intervention ---
    {
        "text": (
            "Allowing soil to rest between interventions promotes microbial recovery. "
            "Mycorrhizal networks, disrupted by tillage or excess moisture, can re-establish "
            "within 7–14 days of undisturbed conditions. Minimise mechanical disturbance "
            "during this period."
        ),
        "tags": ["rest"],
    },
    {
        "text": (
            "Continuous cropping without rest periods depletes soil organic matter and "
            "beneficial soil fauna. A fallow or cover-crop rest phase replenishes microbial "
            "biomass and improves aggregate stability, reducing erosion risk."
        ),
        "tags": ["rest"],
    },

    # --- Nitrogen Management ---
    {
        "text": (
            "Nitrogen is most limiting when soil levels fall below 0.25 g/kg. Apply "
            "slow-release organic fertiliser (e.g., composted manure) to supply 50–80 kg "
            "N/ha. Split applications reduce leaching risk and improve nitrogen use "
            "efficiency by up to 30% compared to single large doses."
        ),
        "tags": ["fertilize"],
    },
    {
        "text": (
            "Excess nitrogen (> 0.7 g/kg available N) causes excessive vegetative growth, "
            "suppresses flowering, and increases nitrous oxide emissions — a potent "
            "greenhouse gas. Avoid fertilising when soil nitrogen already exceeds crop demand."
        ),
        "tags": ["fertilize"],
    },
    {
        "text": (
            "Legume cover crops (clover, vetch, lucerne) fix 50–200 kg N/ha per season "
            "through symbiotic rhizobia. Incorporating legumes into rotations reduces "
            "synthetic fertiliser requirements while building organic matter."
        ),
        "tags": ["fertilize", "rest"],
    },

    # --- pH Management ---
    {
        "text": (
            "Soil pH between 6.0 and 7.0 maximises nutrient availability for most crops. "
            "Below pH 5.5, aluminium and manganese become soluble and toxic; phosphorus "
            "becomes bound to iron and aluminium oxides, reducing plant uptake. "
            "Apply agricultural lime to raise pH gradually (0.1–0.2 units per season)."
        ),
        "tags": ["adjust pH"],
    },
    {
        "text": (
            "Elemental sulfur lowers soil pH in alkaline soils (pH > 7.5) through "
            "oxidation to sulfuric acid by Thiobacillus bacteria. Effects are slow "
            "(3–6 months); apply at 200–500 kg/ha and retest pH after one season."
        ),
        "tags": ["adjust pH"],
    },
    {
        "text": (
            "pH affects microbial community composition more than almost any other soil "
            "property. Bacterial diversity peaks near pH 6.5; acidic soils (< 5.5) shift "
            "communities toward fungi-dominated systems with slower organic matter cycling."
        ),
        "tags": ["adjust pH"],
    },

    # --- Temperature / Thermal Stress ---
    {
        "text": (
            "Soil microbial activity doubles for every 10°C rise in temperature (Q10 ≈ 2) "
            "up to around 35°C, then declines sharply. High temperatures accelerate organic "
            "matter decomposition and nitrogen mineralisation, which can temporarily boost "
            "available N but depletes long-term soil carbon stores."
        ),
        "tags": ["intervene"],
    },
    {
        "text": (
            "Optimal soil temperature for root growth in temperate crops is 15–24°C. "
            "Below 10°C, nutrient uptake slows due to reduced root metabolic activity. "
            "Above 30°C, respiration costs exceed photosynthetic gain in most C3 crops, "
            "reducing yield. Mulching moderates soil temperature extremes."
        ),
        "tags": ["intervene"],
    },
    {
        "text": (
            "A 5 cm layer of organic mulch (straw, wood chips) reduces soil temperature "
            "fluctuation by 3–8°C in summer and prevents frost penetration in winter. "
            "Mulch also conserves moisture by reducing surface evaporation by 25–50%."
        ),
        "tags": ["intervene", "irrigate"],
    },

    # --- General Soil Health ---
    {
        "text": (
            "Healthy soil supports 10 billion microorganisms per gram — bacteria, fungi, "
            "archaea, and protozoa. This living ecosystem drives nutrient cycling, disease "
            "suppression, and carbon sequestration. Interventions should aim to feed and "
            "protect this community, not bypass it."
        ),
        "tags": [],
    },
    {
        "text": (
            "Regenerative practices — cover cropping, minimal tillage, diverse rotations, "
            "and compost application — consistently increase soil organic carbon by "
            "0.1–0.3% per year. Higher organic matter improves water retention, "
            "cation exchange capacity, and long-term fertility."
        ),
        "tags": [],
    },
    {
        "text": (
            "The life-aligned soil target profile: moisture 0.30–0.45, pH 6.0–7.0, "
            "nitrogen 0.35–0.60 g/kg, temperature 15–24°C. Each sensor dimension "
            "contributes to a holistic score; optimising one in isolation can degrade others "
            "(e.g., excess irrigation leaches nitrogen)."
        ),
        "tags": [],
    },
]


class SoilRAGRetriever:
    """
    Embeds the agronomic corpus at init time and exposes a `query()` method
    that takes a soil state + action and returns the top-k most relevant chunks.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        print("🔍 Loading RAG retriever (sentence-transformer + FAISS)...")
        self._model = SentenceTransformer(self._MODEL_NAME)
        self._texts = [entry["text"] for entry in _CORPUS]
        self._tags  = [entry["tags"]  for entry in _CORPUS]

        # Build FAISS index (L2 distance over normalised embeddings ≡ cosine similarity)
        embeddings = self._model.encode(self._texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # Inner-product on normalised vecs = cosine
        self._index.add(embeddings.astype(np.float32))
        print(f"   ✅ Indexed {len(self._texts)} agronomic knowledge chunks.")

    def query(
        self,
        state: dict,
        action_label: str,
        top_k: int = 2,
    ) -> list[dict]:
        """
        Retrieve the top_k most relevant knowledge chunks for the current
        soil state and triggered action.

        Args:
            state:        dict with keys moisture, ph, nitrogen, temperature
            action_label: human-readable action name (from _ACTION_LABEL)
            top_k:        number of results to return

        Returns:
            List of dicts: [{"text": str, "score": float}, ...]
        """
        # Build a natural-language query from sensor readings + action
        query_str = (
            f"Soil condition: moisture={state['moisture']:.2f}, "
            f"pH={state['ph']:.2f}, nitrogen={state['nitrogen']:.2f}, "
            f"temperature={state['temperature']:.1f}°C. "
            f"Recommended action: {action_label}."
        )

        q_vec = self._model.encode([query_str], convert_to_numpy=True, show_progress_bar=False)
        q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-8)

        scores, indices = self._index.search(q_vec.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "text":  self._texts[idx],
                "score": float(score),
                "tags":  self._tags[idx],
            })
        return results

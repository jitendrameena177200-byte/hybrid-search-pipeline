from typing import List, Dict, Any

def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]], 
    dense_results: List[Dict[str, Any]], 
    k: int = 60,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Combines sparse (BM25) and dense retrieval rankings using Reciprocal Rank Fusion (RRF).
    Formula: RRF_Score(d) = sum(1 / (k + rank_i(d)))
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Process BM25 rankings
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Process Dense rankings
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Sort documents by accumulated RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    fused_results = []
    for doc_id, score in sorted_docs[:top_n]:
        fused_doc = doc_map[doc_id].copy()
        fused_doc["rrf_score"] = round(score, 6)
        fused_results.append(fused_doc)

    return fused_results

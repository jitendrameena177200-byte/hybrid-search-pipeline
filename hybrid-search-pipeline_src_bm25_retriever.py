from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class BM25Retriever:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.corpus = [doc["content"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Rank document indices by score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results = []
        for idx in ranked_indices[:top_k]:
            doc = self.documents[idx].copy()
            doc["bm25_score"] = float(scores[idx])
            results.append(doc)
            
        return results

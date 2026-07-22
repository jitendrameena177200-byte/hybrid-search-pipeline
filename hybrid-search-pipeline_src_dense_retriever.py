import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class DenseRetriever:
    def __init__(self, documents: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        
        # Encode document corpus into embeddings
        corpus_texts = [f"{doc['title']} {doc['content']}" for doc in documents]
        self.embeddings = self.model.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Compute cosine similarities via dot product
        similarities = np.dot(self.embeddings, query_embedding)
        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            doc = self.documents[idx].copy()
            doc["dense_score"] = float(similarities[idx])
            results.append(doc)

        return results

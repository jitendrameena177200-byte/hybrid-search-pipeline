import json
from typing import List, Dict, Any
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.rrf import reciprocal_rank_fusion

class HybridSearchPipeline:
    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        print("Initializing BM25 Sparse Retriever...")
        self.bm25_retriever = BM25Retriever(self.documents)

        print("Initializing Dense Embedding Retriever (SentenceTransformers)...")
        self.dense_retriever = DenseRetriever(self.documents)

    def search(self, query: str, top_n: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        bm25_results = self.bm25_retriever.search(query, top_k=top_n * 2)
        dense_results = self.dense_retriever.search(query, top_k=top_n * 2)

        hybrid_results = reciprocal_rank_fusion(
            bm25_results=bm25_results,
            dense_results=dense_results,
            k=rrf_k,
            top_n=top_n
        )
        return hybrid_results

import json
from src.hybrid_pipeline import HybridSearchPipeline

def main():
    pipeline = HybridSearchPipeline(data_path="data/documents.json")
    
    query = "How to reduce machine downtime using predictive AI search?"
    print(f"\n🔍 Query: '{query}'\n")

    results = pipeline.search(query=query, top_n=3)

    print("--- 🚀 Top Hybrid Search Results (RRF Fused) ---")
    for rank, res in enumerate(results, start=1):
        print(f"Rank {rank}: [{res['id']}] {res['title']} (RRF Score: {res['rrf_score']})")
        print(f"Content: {res['content']}\n")

if __name__ == "__main__":
    main()

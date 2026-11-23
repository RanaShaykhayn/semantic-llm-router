from src.retriever import SimpleRetriever
from test.golden_dataset import GOLDEN_DATASET

def evaluate_retrieval():
    print("🚀 Starting RAG Retrieval Evaluation...")
    ret=SimpleRetriever()
    ret.load_documents("documents")
    ret.build_vectors()
    correct=0
    total= len(GOLDEN_DATASET)

    for i in GOLDEN_DATASET:
        query=i["query"]
        ground_truth=i["ground_truth"]
        results=ret.search(query, top_k=1)
        if not results:
              print(f"❌ Query: '{query}' -> No results found.")
              continue
        ret_text=results[0]["text"]
        score=results[0]["score"]

        if ground_truth.lower()in ret_text.lower():
             correct+=1
             print(f"✅ Query: '{query}' -> Retrieved: '{ret_text}' (Score: {score:.2f})")
        else:  
             print(f"❌ Query: '{query}' -> Retrieved: '{ret_text}' (Expected: '{ground_truth}')")

    precision=correct/total
    print(f"\n📊 Retrieval Precision (Top-1): {precision:.2%} ({correct}/{total})")
if __name__ == "__main__":
    evaluate_retrieval()
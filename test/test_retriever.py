from src.retriever import SimpleRetriever

def test_retriever():
    print("🚀 Testing Retriever...")
    
    retriever = SimpleRetriever()
    retriever.load_documents("documents")
    retriever.build_vectors()
    
    
    query = "How many days of annual leave do I have?"
    print(f"\n🔍 Query: {query}")
    
    results = retriever.search(query, top_k=2)
    for i, res in enumerate(results):
        print(f"  Result {i+1} (Score: {res['score']:.4f}): {res['text']}")
    
    print("\n✅ Retriever test complete!")

if __name__ == "__main__":
    test_retriever()
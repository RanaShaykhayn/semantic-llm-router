from src.retriever import SimpleRetriever
from src.evaluator import RAGEvaluator
from test.golden_dataset import GOLDEN_DATASET

def run_evaluation():
    print("🚀 Starting RAG Evaluation with LLM-as-a-Judge...")
    
    
    retriever = SimpleRetriever()
    retriever.load_documents("documents")
    retriever.build_vectors()
    
    
    evaluator = RAGEvaluator(retriever)
    
    
    results = evaluator.evaluate(GOLDEN_DATASET)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    run_evaluation()
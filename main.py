import os
from rag_pipeline import RAGPipeline

def main():
    # Check for API Key
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not found in environment. Please ensure it is set in .env or environment variables.")
        # We continue, but it might fail if not set
    
    file_path = "sample_data.txt"
    rag = RAGPipeline(file_path)
    
    query = "Who was the first person to step on the moon?"
    print(f"Question: {query}")
    print("-" * 30)
    
    try:
        answer = rag.run(query)
        print("-" * 30)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

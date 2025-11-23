# lab-rag

RAG Exploration Project Walkthrough
Overview
This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and OpenAI.

Components
rag_pipeline.py
The core logic resides here.

Document Loading: Loads text from a file.
Text Splitting: Splits text into chunks for embedding.
Vector Store: Uses ChromaDB to store embeddings.
Retrieval & Generation: Sets up a chain to retrieve relevant chunks and generate an answer using an LLM.
main.py
The entry point script.

Checks for API key.
Initializes the pipeline with 
sample_data.txt
.
Asks a sample question: "Who was the first person to step on the moon?"
tests/test_rag_mock.py
A unit test that mocks external API calls (OpenAI) to verify the pipeline logic without needing a real API key or incurring costs.

Setup & Usage
Install Dependencies:

pip install -r requirements.txt
Configure API Key:

Copy 
.env.example
 to .env.
Add your OPENAI_API_KEY.
Run the Project:

python main.py
Run Tests:

python tests/test_rag_mock.py
Verification Results
Mock Tests: Passed. The pipeline logic is correctly implemented.
Live Run: Requires a valid OPENAI_API_KEY.

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add parent directory to path to import rag_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import RAGPipeline
from langchain_core.documents import Document

class TestRAGPipeline(unittest.TestCase):
    @patch('rag_pipeline.ChatOpenAI')
    @patch('rag_pipeline.OpenAIEmbeddings')
    @patch('rag_pipeline.Chroma')
    @patch('rag_pipeline.TextLoader')
    @patch('rag_pipeline.os.path.exists')
    def test_rag_pipeline_flow(self, mock_exists, mock_loader, mock_chroma, mock_embeddings, mock_chat):
        # Setup mocks
        mock_exists.return_value = True
        mock_loader_instance = mock_loader.return_value
        mock_doc = Document(page_content="Test content")
        mock_loader_instance.load.return_value = [mock_doc]

        mock_vector_store = mock_chroma.from_documents.return_value
        mock_retriever = mock_vector_store.as_retriever.return_value
        
        # Mock the chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "Mocked answer"}
        
        # We need to patch create_retrieval_chain and create_stuff_documents_chain as well
        # because they are imported in the module
        with patch('rag_pipeline.create_retrieval_chain') as mock_create_retrieval, \
             patch('rag_pipeline.create_stuff_documents_chain') as mock_create_stuff:
            
            mock_create_retrieval.return_value = mock_chain
            
            # Initialize pipeline
            rag = RAGPipeline("dummy_path.txt")
            
            # Run pipeline
            answer = rag.run("Test query")
            
            # Assertions
            self.assertEqual(answer, "Mocked answer")
            mock_loader.assert_called_with("dummy_path.txt")
            mock_chroma.from_documents.assert_called()
            mock_create_stuff.assert_called()
            mock_create_retrieval.assert_called()
            mock_chain.invoke.assert_called()

if __name__ == '__main__':
    unittest.main()

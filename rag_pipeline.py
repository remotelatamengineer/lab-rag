import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class RAGPipeline:
    def __init__(self, file_path, persist_directory="./chroma_db"):
        self.file_path = file_path
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def load_documents(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        loader = TextLoader(self.file_path)
        return loader.load()

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)

    def create_vector_store(self, chunks):
        # Initialize Chroma with the documents and embeddings
        # This will automatically persist if persist_directory is set
        self.vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.persist_directory
        )

    def get_qa_chain(self):
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Run create_vector_store first.")
        
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        retriever = self.vector_store.as_retriever()
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain

    def run(self, query):
        print("Loading documents...")
        docs = self.load_documents()
        print(f"Loaded {len(docs)} documents.")
        
        print("Splitting documents...")
        chunks = self.split_documents(docs)
        print(f"Split into {len(chunks)} chunks.")
        
        print("Creating vector store...")
        self.create_vector_store(chunks)
        
        print("Creating QA chain...")
        chain = self.get_qa_chain()
        
        print("Invoking chain...")
        response = chain.invoke({"input": query})
        return response["answer"]

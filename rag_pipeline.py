from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

class RagPipeline:
    # Initialize Groq LLM
    def __init__(self):
        self.llm = llm = ChatGroq(
            model_name=os.getenv("MODEL_NAME", "llama3-70b-8192"),
            temperature=os.getenv("LLM_TEMPERATURE", 0.2),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            # system_prompt = os.getenv("SYSTEM_PROMPT", " You are a professional Q&A Expert who assits with academic questions, reply to queries professionally")
        )
        self.embeddings = embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Document Loading and Splitting to chunks
    def chunker(self, file_path:Path):
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()
        return documents
    
    # Initializing a vectorstore to store documents as vectors and returning a retriever object 
    def doc_to_vec(self, documents):
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return retriever
    
    # Combining the retriever with the llm
    def generator(self, retriever, return_source=True):
       qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
       return qa_chain
    
    def run_pipeline(self, file_path:Path):
        documents = self.chunker(file_path)
        retriever = self.doc_to_vec(documents)
        qa_chain = self.generator(retriever)
        return qa_chain
    

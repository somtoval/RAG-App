from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

class RagPipeline:
    # Initialize Groq LLM
    def __init__(self):
        self.llm = llm = ChatGroq(
            model_name=os.getenv("MODEL_NAME", "llama3-70b-8192"),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.2)),
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.embeddings = embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.memory = memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True, 
            input_key="question",
            output_key="answer"
        )
        # self.system_prompt = os.getenv("SYSTEM_PROMPT", " You are a professional Q&A Expert who assits with academic questions, reply to queries professionally")
        system_template = """You are a helpful and respectful AI assistant.
        Answer the question based only on the following context:
        {context}
        Relate with the user generally but know when it is not related or you don't have the answer and reply politely
        """
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

        self.prompt = ChatPromptTemplate.from_messages([
            self.system_message_prompt,
            self.human_message_prompt
        ])

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
       qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            memory=self.memory,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            # verbose=True,
        )
       return qa_chain
    
    def run_pipeline(self, file_path:Path):
        documents = self.chunker(file_path)
        retriever = self.doc_to_vec(documents)
        qa_chain = self.generator(retriever)
        return qa_chain
    
    # Helper method to clear memory if needed
    def clear_memory(self):
        self.memory.clear()
    
    # Helper method to get chat history
    def get_chat_history(self):
        return self.memory.chat_memory.messages
    

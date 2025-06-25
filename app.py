from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional
import logging
from generator import Prediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Assistant Chat Server", version="1.0.0")

# Initialize the RAG pipeline
predictor = Prediction()
qa_chain = None

# Initialize the pipeline on startup
@app.on_event("startup")
async def startup_event():
    global qa_chain
    try:
        logger.info("Initializing RAG pipeline...")
        qa_chain = predictor.pipeline_runner()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise e

# Pydantic models for request/response
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

# Serve the main HTML file
@app.get("/", response_class=HTMLResponse)
async def serve_chat_interface():
    """Serve the main chat interface HTML file"""
    try:
        # Read the HTML file content from paste.txt
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Update the JavaScript to use the actual API endpoint
        updated_html = html_content.replace(
            '// Simulate AI response (replace with actual API call)',
            '// Call actual API endpoint'
        ).replace(
            '''setTimeout(() => {
                hideTypingIndicator();
                const responses = [
                    "I understand your question. Let me help you with that.",
                    "That's an interesting point! Here's what I think about it...",
                    "Great question! Based on the information available, I can tell you that...",
                    "I'd be happy to help you with that. Let me break it down for you.",
                    "Thanks for asking! Here's a comprehensive answer to your question..."
                ];
                const randomResponse = responses[Math.floor(Math.random() * responses.length)];
                addMessage(randomResponse, 'assistant');
            }, 1500 + Math.random() * 1000);''',
            '''// Call the FastAPI backend
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                hideTypingIndicator();
                if (data.success) {
                    addMessage(data.response, 'assistant');
                } else {
                    addMessage('Sorry, I encountered an error: ' + (data.error || 'Unknown error'), 'assistant');
                }
            })
            .catch(error => {
                hideTypingIndicator();
                addMessage('Sorry, I encountered a connection error. Please try again.', 'assistant');
                console.error('Error:', error);
            });'''
        )
        
        return HTMLResponse(content=updated_html)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="HTML file not found")
    except Exception as e:
        logger.error(f"Error serving HTML file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Process chat messages through the RAG pipeline"""
    global qa_chain
    
    if qa_chain is None:
        return ChatResponse(
            response="Sorry, the AI system is not ready yet. Please try again in a moment.",
            success=False,
            error="RAG pipeline not initialized"
        )
    
    try:
        logger.info(f"Processing message: {message.message}")
        
        # Use the RAG pipeline to generate response
        response = predictor.predict_query(message.message, qa_chain)
        
        logger.info(f"Generated response: {response}")
        
        return ChatResponse(
            response=response,
            success=True
        )
    
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return ChatResponse(
            response="I apologize, but I encountered an error while processing your message. Please try again.",
            success=False,
            error=str(e)
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_ready": qa_chain is not None
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
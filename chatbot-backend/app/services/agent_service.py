import os
import json
from typing import AsyncGenerator, List, Dict
from dotenv import load_dotenv
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent
from openai import BadRequestError
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from pinecone.exceptions import NotFoundException
from app.utils.document_processor import DocumentProcessor

# Load environment variables
load_dotenv()


class AgentService:
    """Service for handling agent interactions and chat logic."""
    
    def __init__(self):
        """Initialize the agent service with OpenAI agent and Pinecone."""
        # Get API key
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please ensure it is defined in your .env file.")
        
        # Load university data
        self.university_data = self._load_university_data()
        
        # Initialize agent
        self.agent = self._initialize_agent()
        
        # Initialize Pinecone for document embeddings
        self._initialize_pinecone()
    
    def _load_university_data(self) -> dict:
        """Load Iqra University data from JSON file."""
        data_path = os.path.join("knowledge", "iqra_university_data.json")
        try:
            with open(data_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Warning: Data file not found at {data_path}")
            return {}
    
    def _initialize_agent(self) -> Agent:
        """Initialize the OpenAI agent with university context."""
        context = """You are the official AI assistant for Iqra University. You help students, faculty, and visitors with university-related information and uploaded documents.

PRIMARY ROLE - Iqra University Information:
1. Answer questions about Iqra University (programs, admissions, policies, campus, etc.)
2. Provide accurate, helpful information based on university data
3. If general questions are not related to Iqra University or uploaded documents, politely redirect to university-related topics

SECONDARY ROLE - Document Analysis:
1. **IMPORTANT:** When a user uploads a document, you MUST analyze and discuss its contents regardless of topic
2. Answer questions about uploaded documents directly and accurately
3. Extract information, count items, list contents, summarize, or explain as requested
4. Be thorough and specific when discussing uploaded documents

RULES FOR UPLOADED DOCUMENTS:
- If the user asks about an uploaded document, ALWAYS prioritize the document content
- Count words, list items, extract data, or summarize as requested
- Don't refuse to discuss document content just because it's not about Iqra University
- Be specific and detailed in your analysis
- If asked to count or list items, do it accurately from the provided document context

RULES FOR GENERAL QUESTIONS (without document context):
- For non-university questions without documents, politely say: "I'm primarily here to help with Iqra University information. Do you have questions about our programs, admissions, or university services?"
- For university-related questions, provide helpful, accurate responses

When responding:
- Be direct, clear, and helpful
- Use markdown formatting for better readability
- Provide specific details when available
- Maintain a professional and friendly tone

The following information about Iqra University is available to you:
{data}
"""
        formatted_context = context.format(data=json.dumps(self.university_data, indent=2))
        
        return Agent(
            name="Iqra University Assistant",
            instructions=formatted_context,
            model="gpt-4o-mini"  # Upgraded for better document analysis and reasoning
        )
    
    def _initialize_pinecone(self):
        """Initialize Pinecone for document embeddings."""
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.pinecone_index_name = "iqra-docs"
        
        if not self.pinecone_api_key:
            print("Warning: PINECONE_API_KEY not set. Document upload features will be disabled.")
            self.pc = None
            self.index = None
            self.embedder = None
            return
        
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Create index if it doesn't exist
            if self.pinecone_index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.pinecone_index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=os.environ.get("PINECONE_CLOUD", "aws"),
                        region=os.environ.get("PINECONE_REGION", "us-east-1")
                    )
                )
            
            self.index = self.pc.Index(self.pinecone_index_name)
            # Use model that produces 768-dimensional embeddings to match Pinecone index
            self.embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        except Exception as e:
            print(f"Warning: Failed to initialize Pinecone: {e}")
            self.pc = None
            self.index = None
            self.embedder = None
    
    async def search_pinecone(self, query: str, session_id: str, top_k: int = 3) -> List[str]:
        """Search for relevant document chunks in Pinecone."""
        if not self.index or not self.embedder:
            return []
        
        try:
            emb = self.embedder.encode(query).tolist()
            res = self.index.query(
                vector=emb,
                top_k=top_k,
                include_metadata=True,
                namespace=session_id
            )
            
            matches = [
                m["metadata"]["text"]
                for m in res["matches"]
                if "metadata" in m and "text" in m["metadata"]
            ]
            
            # Debug logging
            if matches:
                print(f"✓ Pinecone search: Found {len(matches)} chunks in namespace '{session_id}'")
            else:
                print(f"ℹ Pinecone search: No documents found in namespace '{session_id}'")
            
            return matches
        except Exception as e:
            print(f"Error searching Pinecone: {e}")
            return []
    
    async def stream_chat_response(
        self,
        session_id: str,
        message: str,
        history: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response from the agent.
        
        Args:
            session_id: Session identifier
            message: User message
            history: Chat history for context
            
        Yields:
            Token chunks from the agent response
        """
        # Build conversation history - filter out timestamps and only keep role/content
        conversation = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
        ]
        
        # Add current user message
        conversation.append({
            "role": "user",
            "content": message
        })
        
        # Search for relevant uploaded documents (increased top_k for better context)
        pinecone_context = await self.search_pinecone(message, session_id, top_k=5)
        if pinecone_context:
            # Make document context very clear and prominent
            doc_context = f"""
📄 UPLOADED DOCUMENT CONTEXT (User has uploaded a document - prioritize this information):
{'=' * 80}
{chr(10).join(pinecone_context)}
{'=' * 80}

IMPORTANT: The user is asking about the content above from their uploaded document. 
Answer based on this document content. Count, list, or extract information as requested.
Do NOT refuse to answer just because the content is not about Iqra University.
"""
            conversation.append({
                "role": "system",
                "content": doc_context
            })
        
        try:
            # Stream response from agent
            result = Runner.run_streamed(
                self.agent,
                input=conversation,
            )
            
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta
                    yield token
                    
        except BadRequestError as e:
            if "API key expired" in str(e):
                yield "I apologize, but there's an issue with the API key. Please contact the administrator."
            else:
                yield f"I apologize, but there was an error: {str(e)}"
        except Exception as e:
            yield f"I apologize, but there was an unexpected error: {str(e)}"
    
    async def embed_and_upsert(self, file_path: str, session_id: str) -> dict:
        """
        Embed a document and upsert to Pinecone.
        
        Args:
            file_path: Path to the document file
            session_id: Session identifier for namespacing
            
        Returns:
            Dictionary with processing statistics
        """
        if not self.index or not self.embedder:
            raise ValueError("Pinecone is not initialized. Please set PINECONE_API_KEY.")
        
        # Extract text based on file type
        try:
            text = DocumentProcessor.extract_text(file_path)
        except Exception as e:
            raise ValueError(f"Failed to extract text from document: {str(e)}")
        
        if not text or len(text.strip()) < 10:
            raise ValueError("Document appears to be empty or has insufficient content")
        
        # Split text into chunks (500 characters with some overlap)
        chunk_size = 500
        overlap = 50
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
        
        if not chunks:
            raise ValueError("No text chunks could be extracted from the document")
        
        # Generate embeddings and prepare vectors
        vectors = []
        filename = os.path.basename(file_path)
        
        for idx, chunk in enumerate(chunks):
            try:
                emb = self.embedder.encode(chunk).tolist()
                vectors.append({
                    "id": f"{session_id}_{filename}_{idx}",
                    "values": emb,
                    "metadata": {
                        "text": chunk,
                        "session": session_id,
                        "file": filename,
                        "chunk_index": idx
                    }
                })
            except Exception as e:
                print(f"Warning: Failed to embed chunk {idx}: {e}")
                continue
        
        if not vectors:
            raise ValueError("Failed to generate embeddings for the document")
        
        # Upsert to Pinecone
        try:
            self.index.upsert(vectors=vectors, namespace=session_id)
        except Exception as e:
            raise ValueError(f"Failed to store document in vector database: {str(e)}")
        
        return {
            "chunks_processed": len(vectors),
            "total_characters": len(text),
            "filename": filename
        }
    
    async def cleanup_session(self, session_id: str):
        """
        Cleanup Pinecone namespace for a session.
        
        Args:
            session_id: Session identifier
        """
        if not self.index:
            return
        
        try:
            self.index.delete(delete_all=True, namespace=session_id)
        except NotFoundException:
            pass
        except Exception as e:
            print(f"Error cleaning up session {session_id}: {e}")


import os
import json
from typing import List
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, FileSearchTool
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent
from openai import BadRequestError
import shutil
import uuid
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from pinecone.exceptions import NotFoundException

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Get the API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Check if the API key is present; if not, raise an error
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Load university data
DATA_PATH = os.path.join("knowledge", "iqra_university_data.json")
try:
    with open(DATA_PATH, 'r') as file:
        university_data = json.load(file)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

# Create a context string from the university data
context = """You are the official AI assistant for Iqra University ONLY. Your sole purpose is to provide information about Iqra University.

STRICT RULES:
1. ONLY answer questions related to Iqra University
2. If a question is not about Iqra University, politely respond: "I can only provide information about Iqra University. Please ask me questions related to our university's programs, admissions, policies, or general information."
3. NEVER provide information about other universities or non-university topics
4. If you're unsure if the information is specific to Iqra University, ask for clarification
5. When analyzing uploaded documents, only discuss their content if it's relevant to Iqra University

Your role for Iqra University queries:
1. Help students and visitors with questions about Iqra University
2. Provide accurate information about courses, admissions, policies, and general inquiries
3. Generate professional, clear, and well-structured responses
4. Analyze uploaded documents and answer questions about them in context with Iqra University

When responding about Iqra University:
- Provide direct, clear answers without mentioning the data source
- If the information is not available in your data, say: "I apologize, but I don't have that specific information about Iqra University. Please contact the university directly or check the official website for the most up-to-date information."
- Maintain a professional and helpful tone
- Use natural language, not JSON or technical formats
- Structure your responses in a readable format using markdown when appropriate

The following information about Iqra University is available to you (but don't mention this directly):
{data}
"""

# Format the context with the data
formatted_context = context.format(data=json.dumps(university_data, indent=2))

UPLOAD_DIR = "uploaded_docs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

agent: Agent = Agent(
    name="Iqra University Assistant",
    instructions=formatted_context,
    model="gpt-3.5-turbo"
    # tools=[]  # or just omit this argument
)

# Pinecone and embedding setup
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
PINECONE_INDEX = "iqra-docs"
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )
index = pc.Index(PINECONE_INDEX)

embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # 768 dimensions to match Pinecone index

# Helper to embed and upsert document
async def embed_and_upsert(file_path, session_id):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    # Split text into chunks (simple split, can be improved)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    vectors = []
    for idx, chunk in enumerate(chunks):
        emb = embedder.encode(chunk).tolist()
        vectors.append({
            "id": f"{session_id}_{os.path.basename(file_path)}_{idx}",
            "values": emb,
            "metadata": {"text": chunk, "session": session_id, "file": os.path.basename(file_path)}
        })
    if vectors:
        index.upsert(vectors=vectors, namespace=session_id)

# Helper to search Pinecone
async def search_pinecone(query, session_id, top_k=3):
    emb = embedder.encode(query).tolist()
    res = index.query(vector=emb, top_k=top_k, include_metadata=True, namespace=session_id)
    return [m["metadata"]["text"] for m in res["matches"] if "metadata" in m and "text" in m["metadata"]]

#CHAINLIT INTERFACE CODE

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    cl.user_session.set("documents", {})
    cl.user_session.set("uploaded_file_paths", [])
    cl.user_session.set("session_id", str(uuid.uuid4()))
    # Reset FileSearchTool search paths for this session
    # file_search_tool.search_paths = ["knowledge/"] # This line is removed as FileSearchTool is removed
    await cl.Message(
        content="""👋 Welcome to the Iqra University Assistant! I'm here to help you with information specifically about Iqra University:

- Courses and Programs at Iqra University
- Admission Requirements for Iqra University
- Iqra University Policies
- General Information about Iqra University

You can also upload multiple documents (PDF, Word, or text files) related to Iqra University, and I'll help you understand them.

Please note that I can only provide information about Iqra University. How can I assist you today?"""
    ).send()

@cl.on_chat_end
async def cleanup():
    uploaded_file_paths = cl.user_session.get("uploaded_file_paths")
    session_id = cl.user_session.get("session_id")
    if uploaded_file_paths:
        for path in uploaded_file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
    # Clean up Pinecone namespace
    if session_id:
        try:
            index.delete(delete_all=True, namespace=session_id)
        except NotFoundException:
            pass
    if os.path.exists(UPLOAD_DIR) and not os.listdir(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    documents = cl.user_session.get("documents")
    uploaded_file_paths = cl.user_session.get("uploaded_file_paths") or []
    session_id = cl.user_session.get("session_id")
    new_files = []

    # Handle file uploads
    if message.elements:
        processing_msg = await cl.Message(content="Processing your uploaded documents...").send()
        try:
            for element in message.elements:
                if hasattr(element, 'content') and isinstance(element.content, bytes):
                    filename = f"{session_id}_{getattr(element, 'name', 'uploaded_file')}"
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    with open(file_path, "wb") as f:
                        f.write(element.content)
                    new_files.append(file_path)
                    # Embed and upsert to Pinecone
                    await embed_and_upsert(file_path, session_id)
            uploaded_file_paths.extend(new_files)
            cl.user_session.set("uploaded_file_paths", uploaded_file_paths)
            # file_search_tool.search_paths = ["knowledge/"] + uploaded_file_paths # This line is removed as FileSearchTool is removed
            response = "I've successfully uploaded the following documents:\n"
            for path in new_files:
                response += f"- {os.path.basename(path)}\n"
            response += "\nWhat would you like to know about these documents?"
            await cl.Message(content=response).send()
            if message.content:
                await handle_query(message.content, history, documents)
            return
        except Exception as e:
            await cl.Message(content=f"I apologize, but I couldn't process your document. Error: {str(e)}").send()
            return
    # Handle regular messages
    await handle_query(message.content, history, documents)

async def handle_query(content: str, history: list, documents: dict):
    """Handle user queries, whether about documents or general information."""
    msg = await cl.Message(content="").send()

    # Add user message to history
    history.append({
        "role": "user",
        "content": content
    })
    
    session_id = cl.user_session.get("session_id")
    # Retrieve relevant docs from Pinecone
    pinecone_context = await search_pinecone(content, session_id)
    if pinecone_context:
        doc_context = "\n\nRelevant Uploaded Document Content:\n" + "\n---\n".join(pinecone_context)
        history.append({
            "role": "system",
            "content": doc_context
        })

    try:
        # Get response from the agent
        response_content = ""
        result = Runner.run_streamed(
            agent,
            input=history,
        )
        
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                token = event.data.delta
                response_content += token
                await msg.stream_token(token)

        # Add assistant's response to history
        history.append({
            "role": "assistant",
            "content": response_content
        })
        
        cl.user_session.set("history", history)
        
    except BadRequestError as e:
        if "API key expired" in str(e):
            await msg.update("I apologize, but there's an issue with the API key. Please contact the administrator to update the API key.")
        else:
            await msg.update(f"I apologize, but there was an error processing your request: {str(e)}")
    except Exception as e:
        await msg.update(f"I apologize, but there was an unexpected error: {str(e)}")

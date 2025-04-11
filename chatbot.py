import os
import json
from typing import List
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent
import PyPDF2
import docx
from io import BytesIO

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Get the API key from environment variables
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Load university data
DATA_PATH = os.path.join("knowledge", "iqra_university_data.json")
try:
    with open(DATA_PATH, 'r') as file:
        university_data = json.load(file)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

async def process_document(file: cl.File) -> str:
    """Extract text from uploaded documents."""
    content = ""
    
    try:
        if file.mime == "application/pdf":
            # Handle PDF files
            pdf_stream = BytesIO(await file.get_bytes())
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
                
        elif file.mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Handle DOCX files
            doc_stream = BytesIO(await file.get_bytes())
            doc = docx.Document(doc_stream)
            for para in doc.paragraphs:
                content += para.text + "\n"
                
        elif file.mime.startswith("text/"):
            # Handle text files
            content = (await file.get_bytes()).decode("utf-8")
            
        else:
            return f"Unsupported file type: {file.mime}. Please upload PDF, Word, or text files only."
            
        return content
    except Exception as e:
        return f"Error processing file: {str(e)}"

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

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

run_config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Format the context with the data
formatted_context = context.format(data=json.dumps(university_data, indent=2))
agent: Agent = Agent(name="Iqra University Assistant", instructions=formatted_context, model=model)

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    cl.user_session.set("documents", {})
    
    await cl.Message(
        content="""👋 Welcome to the Iqra University Assistant! I'm here to help you with information specifically about Iqra University:

- Courses and Programs at Iqra University
- Admission Requirements for Iqra University
- Iqra University Policies
- General Information about Iqra University

You can also upload documents (PDF, Word, or text files) related to Iqra University, and I'll help you understand them.

Please note that I can only provide information about Iqra University. How can I assist you today?"""
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    documents = cl.user_session.get("documents")
    
    # Handle file uploads
    if message.elements:
        processing_msg = await cl.Message(content="Processing your uploaded documents...").send()
        
        try:
            for element in message.elements:
                if isinstance(element, cl.File):
                    doc_content = await process_document(element)
                    if doc_content.startswith("Error") or doc_content.startswith("Unsupported"):
                        await processing_msg.update(content=doc_content)
                        return
                    documents[element.name] = doc_content
                    
            cl.user_session.set("documents", documents)
            
            # Create response about the processed documents
            response = "I've successfully processed the following documents:\n"
            for doc_name in documents.keys():
                response += f"- {doc_name}\n"
            response += "\nWhat would you like to know about these documents?"
            
            await processing_msg.update(content=response)
            
            if message.content:
                # If there's a question along with the upload, process it
                await handle_query(message.content, history, documents)
            return
            
        except Exception as e:
            await processing_msg.update(content=f"I apologize, but I couldn't process your document. Error: {str(e)}")
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
    
    # If there are documents, add their content to the context
    if documents:
        doc_context = "\n\nUploaded Documents Content:\n"
        for doc_name, doc_content in documents.items():
            doc_context += f"\nDocument '{doc_name}':\n{doc_content}\n"
        history.append({
            "role": "system",
            "content": doc_context
        })

    # Get response from the agent
    response_content = ""
    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=run_config,
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

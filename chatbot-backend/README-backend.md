# IQRAi Backend - FastAPI Service

## Overview

This is the FastAPI backend for **IQRAi**, the AI assistant for Iqra University. It provides a RESTful API with streaming support using the OpenAI Agents SDK.

## Features

- ✅ **Streaming Chat**: Real-time SSE (Server-Sent Events) streaming responses
- ✅ **Session Management**: Persistent chat sessions with SQLite storage
- ✅ **Document Context**: Pinecone-based document embedding and retrieval
- ✅ **OpenAI Agents SDK**: Leverages the official OpenAI agents framework
- ✅ **CORS Enabled**: Ready to connect with React frontend on `http://localhost:3000`

## Architecture

```
chatbot-backend/
├── app/
│   ├── main.py                 # FastAPI app initialization and routes
│   ├── services/
│   │   └── agent_service.py    # Agent logic (from chatbot.py)
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   └── db/
│       └── session_store.py    # SQLite session storage
├── knowledge/
│   └── iqra_university_data.json
├── .env                        # Environment variables (create from .env.example)
├── requirements.txt            # Python dependencies
└── README-backend.md           # This file
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or using `uv` (if you have it):

```bash
uv pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-actual-openai-key
PINECONE_API_KEY=your-pinecone-key  # Optional
DATABASE_URL=sqlite:///./chat.db
BACKEND_PORT=8000
```

## Running the Server

### Development Mode (with auto-reload)

```bash
uvicorn app.main:app --reload --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

### Stream Chat
```bash
POST /chat/stream
```

**Request Body:**
```json
{
  "session_id": "optional-session-id",
  "message": "What programs does Iqra University offer?"
}
```

**Response:** Server-Sent Events (SSE) stream

**Example with curl:**
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Iqra University"
  }'
```

**Example with httpie:**
```bash
http POST http://localhost:8000/chat/stream \
  message="What are the admission requirements?"
```

### Get Session History
```bash
GET /sessions/{session_id}/history
```

**Response:**
```json
{
  "session_id": "abc-123",
  "history": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2025-10-24T10:30:00"
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help you?",
      "timestamp": "2025-10-24T10:30:01"
    }
  ]
}
```

### Delete Session
```bash
DELETE /sessions/{session_id}
```

**Response:**
```json
{
  "message": "Session abc-123 deleted successfully"
}
```

## Testing the API

### Using curl (Streaming)
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Iqra University?"}'
```

### Using Python
```python
import requests
import json

url = "http://localhost:8000/chat/stream"
data = {"message": "Tell me about admissions"}

response = requests.post(url, json=data, stream=True)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            event_data = json.loads(line[6:])
            if event_data['type'] == 'token':
                print(event_data['content'], end='', flush=True)
```

### Using JavaScript (Frontend)
```javascript
const response = await fetch('http://localhost:8000/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'Hello' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'token') {
        console.log(data.content);
      }
    }
  }
}
```

## Database

The backend uses SQLite for storing:
- **Sessions**: Chat session metadata
- **Messages**: Individual messages with role, content, and timestamp

Database file: `chat.db` (created automatically on first run)

## Integration with Chainlit Logic

The FastAPI service reuses the core agent logic from `chatbot.py`:
- Agent initialization with university data context
- Pinecone document embedding and search
- OpenAI Agents SDK streaming
- Session-based conversation history

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8000   # Windows (then use Task Manager)
```

### CORS Issues
Make sure your frontend is running on `http://localhost:3000`, or update the CORS settings in `app/main.py`

### API Key Errors
Verify your `.env` file has valid API keys:
```bash
cat .env | grep OPENAI_API_KEY
```

## Next Steps

- Connect this backend to your React frontend
- Add file upload endpoints for document processing
- Implement user authentication
- Add rate limiting and monitoring

## License

Part of the Iqra University FYP project.


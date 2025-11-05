import chainlit as cl
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    streaming=True
)

# Initialize tools
tools = load_tools(["ddg-search", "wikipedia"])

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

@cl.on_chat_start
def start():
    """Initialize the chat session."""
    cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages."""
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    
    # Create a new Chainlit message
    msg = cl.Message(content="")
    
    # Run the agent with streaming
    async for chunk in agent.astream(
        {"input": message.content},
        callbacks=[cl.AsyncLangchainCallbackHandler()],
    ):
        if isinstance(chunk, dict):
            await msg.stream_token(chunk.get("output", ""))
        else:
            await msg.stream_token(str(chunk))
    
    await msg.send() 
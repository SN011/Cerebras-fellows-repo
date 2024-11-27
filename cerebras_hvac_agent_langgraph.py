import queue
from quart import Quart, request, jsonify, send_file, render_template
import whisper
import pyaudio
import webrtcvad
import collections
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from quart_cors import cors
import logging
import os
from dotenv import load_dotenv
import socketio
import pyttsx3
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_cerebras import ChatCerebras
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import tool
from tools.initialize_cerebras import init_cerebras
from Utils_cerebras import (
    initialize_web_search_agent,
    initialize_pdf_search_agent,
    run_quote_logics,
    vector_embedding
)
from flask import Flask

# Initialize configuration
load_dotenv()
audio_path = os.getenv('AUDIO_PATH', 'audio.wav')
tts_synthesis_path = os.getenv('NEW_TTS_SYNTHESIS', os.path.join('paths', 'synthesis.wav'))

# Create paths directory if it doesn't exist
os.makedirs(os.path.dirname(tts_synthesis_path), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Quart app and Socket.IO
app = Quart(__name__)
app = cors(app, allow_origin="*")
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
app_asgi = socketio.ASGIApp(sio, app)

# Initialize Flask app for sync operations
app_sync = Flask(__name__)
sio_sync = socketio.Server()
app_sync.wsgi_app = socketio.WSGIApp(sio_sync, app_sync.wsgi_app)

# Initialize Whisper model
model = whisper.load_model("base")

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
is_recording = False

# Initialize audio components
audio = pyaudio.PyAudio()
vad = webrtcvad.Vad(3)
executor = ThreadPoolExecutor(max_workers=20)

# Initialize Cerebras
client, llm = init_cerebras()

# Define state type
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: Optional[str]
    error: Optional[str]

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    agent = initialize_web_search_agent(llm)
    result = agent.invoke({"input": query})
    return result["output"]

@tool
def search_hvac_docs(query: str) -> str:
    """Search HVAC documentation"""
    result = initialize_pdf_search_agent(llm, query, vector_embedding())
    return result

@tool
def generate_quote(chat_history: List[Dict]) -> str:
    """Generate an HVAC quote"""
    result = run_quote_logics(client, llm, chat_history=chat_history)
    return result

# Create tool executor
tools = [search_web, search_hvac_docs, generate_quote]
tool_executor = ToolExecutor(tools)

# Define the agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Marvin, a highly intelligent assistant specializing in HVAC services. 
    When users mention HVAC issues, include 'I will instruct the technician agent'.
    When cost determination is needed, say 'determining the HVAC quote'.
    When all quote information is gathered, say 'Questionnaire complete'.
    For non-HVAC queries, say 'forwarding to web search agent'."""),
    ("human", "{input}"),
])

# Define agent functions
def should_continue(state: AgentState) -> str:
    """Enhanced routing logic with detailed logging"""
    try:
        if state.get("error"):
            logger.error(f"Error in state: {state['error']}")
            return "error_handler"
            
        if not state["messages"]:
            logger.debug("No messages in state, continuing to agent")
            return "continue"
        
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            logger.debug("Last message is not AI message, continuing to agent")
            return "continue"
            
        content = last_message.content.lower()
        logger.debug(f"Routing based on content: {content[:100]}...")
        
        # Define routing conditions with logging
        routing_conditions = {
            "questionnaire complete": "quote",
            "i will instruct the technician agent": "hvac",
            "determining the hvac quote": "quote",
            "forwarding to web search agent": "search"
        }
        
        for phrase, route in routing_conditions.items():
            if phrase in content:
                logger.info(f"Routing to {route} based on phrase: {phrase}")
                return route
                
        logger.debug("No specific routing condition met, continuing to agent")
        return "continue"
    except Exception as e:
        logger.error(f"Error in routing: {e}")
        return "error_handler"

def call_llm(state: AgentState) -> AgentState:
    """LLM node with comprehensive error handling and logging"""
    try:
        if not state["messages"]:
            logger.warning("LLM node: No messages in state")
            return {
                "messages": [],
                "next": None,
                "error": "No messages to process"
            }
        
        last_message = state["messages"][-1]
        logger.info(f"Processing message with LLM: {last_message.content[:100]}...")
        
        # Format prompt with explicit system message
        prompt_messages = agent_prompt.format_messages(
            input=last_message.content,
            chat_history=state["messages"][:-1]  # Include chat history except last message
        )
        
        # Get LLM response with timeout
        response = llm.invoke(
            prompt_messages,
            temperature=0.7,  # Add temperature for more dynamic responses
            timeout=30  # Add timeout to prevent hanging
        )
        
        logger.info(f"LLM response generated: {response.content[:100]}...")
        return {
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "next": None,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error in LLM node: {e}")
        return {
            "messages": state["messages"],
            "next": None,
            "error": f"LLM processing failed: {str(e)}"
        }

# Create reusable tool node function generator
def create_tool_node(tool_executor, tool_name: str):
    """Create a node function that executes a specific tool with proper error handling"""
    def node_function(state: AgentState) -> AgentState:
        try:
            if not state["messages"]:
                logger.warning(f"{tool_name} node: No messages in state")
                return {
                    "messages": [],
                    "next": None,
                    "error": "No messages to process"
                }
            
            last_message = state["messages"][-1]
            logger.info(f"Executing {tool_name} with input: {last_message.content[:100]}...")
            
            # Use run instead of arun for synchronous execution
            result = tool_executor.run(tool_name, last_message.content)
            
            logger.info(f"{tool_name} execution successful")
            return {
                "messages": state["messages"] + [AIMessage(content=str(result))],
                "next": None,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in {tool_name} node: {e}")
            return {
                "messages": state["messages"],
                "next": None,
                "error": f"{tool_name} execution failed: {str(e)}"
            }
    return node_function

# Create the graph
def setup_workflow():
    """Setup workflow with comprehensive error handling and logging"""
    try:
        workflow = StateGraph(AgentState)
        
        # Add nodes with explicit logging
        logger.info("Setting up workflow nodes...")
        workflow.add_node("agent", call_llm)
        workflow.add_node("search", create_tool_node(tool_executor, "search_web"))
        workflow.add_node("hvac", create_tool_node(tool_executor, "search_hvac_docs"))
        workflow.add_node("quote", create_tool_node(tool_executor, "generate_quote"))
        workflow.add_node("error_handler", lambda x: {
            **x,
            "messages": x["messages"] + [AIMessage(content="I apologize, but I encountered an error. Let me try to help you differently.")]
        })
        
        # Add edges with logging
        logger.info("Setting up workflow edges...")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "search": "search",
                "hvac": "hvac",
                "quote": "quote",
                "continue": "agent",
                "error": "error_handler"
            }
        )
        
        workflow.add_edge("search", "agent")
        workflow.add_edge("hvac", "agent")
        workflow.add_edge("quote", END)
        workflow.add_edge("error_handler", END)
        
        logger.info("Workflow setup completed successfully")
        return workflow.compile()
    except Exception as e:
        logger.error(f"Error setting up workflow: {e}")
        raise

# Initialize the workflow
app_chain = setup_workflow()

# Text-to-speech functions
def synthesize_and_save(text: str) -> None:
    """Synthesize text to speech and save to file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(tts_synthesis_path), exist_ok=True)
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 175)
        engine.setProperty('volume', 1.0)
        
        # Save to file
        engine.save_to_file(text, tts_synthesis_path)
        engine.runAndWait()
        
        # Verify file was created
        if not os.path.exists(tts_synthesis_path):
            raise Exception("Failed to create audio file")
            
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        # Create empty audio file if synthesis fails
        with open(tts_synthesis_path, 'wb') as f:
            f.write(b'')

async def synthesize_speech(text: str) -> None:
    """Async wrapper for speech synthesis"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, synthesize_and_save, text)
    await sio.emit('tts_complete', {'message': 'TTS synthesis complete', 'file_path': tts_synthesis_path})

# Routes and Socket.IO events
@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/talk', methods=['POST'])
async def talk():
    """Enhanced route handler with comprehensive error handling"""
    try:
        if is_recording:
            return jsonify({"error": "Recording is still in progress"}), 400

        logger.info("Starting audio transcription...")
        transcription = await asyncio.get_event_loop().run_in_executor(
            executor, transcribe_audio
        )
        
        logger.info(f"Transcription completed: {transcription[:100]}...")
        
        # Initialize state with all required fields
        initial_state = {
            "messages": [HumanMessage(content=transcription)],
            "next": None,
            "error": None
        }
        
        # Execute workflow with timeout
        logger.info("Executing workflow...")
        try:
            async with asyncio.timeout(30):  # Add timeout for workflow execution
                result = app_chain.invoke(initial_state)
        except asyncio.TimeoutError:
            logger.error("Workflow execution timed out")
            return jsonify({"error": "Request timed out"}), 504
        
        # Check for errors
        if result.get("error"):
            logger.error(f"Workflow error: {result['error']}")
            return jsonify({"error": result["error"]}), 500
            
        # Get last message
        last_message = result["messages"][-1]
        logger.info(f"Generated response: {last_message.content[:100]}...")
        
        # Synthesize speech
        await synthesize_speech(last_message.content)
        
        return jsonify({'response': last_message.content})
    except Exception as e:
        logger.error(f"Error in talk endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/text_input', methods=['POST'])
async def text_input():
    try:
        data = await request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Initialize state with the text input
        state = {"messages": [HumanMessage(content=text)]}
        
        # Invoke the chain and get result
        result = app_chain.invoke(state)
        
        # Get the last message (the AI's response)
        last_message = result["messages"][-1]
        
        # Synthesize speech for the response
        await synthesize_speech(last_message.content)
        
        return jsonify({'response': last_message.content})
    except Exception as e:
        logger.error(f"Error processing text input: {e}")
        return jsonify({"error": "Failed to process text input"}), 500

@app.route('/get_audio')
async def get_audio():
    return await send_file(tts_synthesis_path, mimetype="audio/mp3")

# Socket.IO event handlers
@sio.event
async def connect(sid, environ, auth):
    logger.debug(f'Client connected: {sid}')

@sio.event
async def disconnect(sid):
    logger.debug(f'Client disconnected: {sid}')

# Audio recording functions
async def record_audio():
    global is_recording
    logger.debug('Starting audio recording...')
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = []
        ring_buffer = collections.deque(maxlen=100)
        triggered = False
        voiced_frames = []
        silence_threshold = 10
        silence_chunks = 0

        while is_recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            is_speech = vad.is_speech(data, RATE)
            ring_buffer.append((data, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])

            if not triggered:
                if num_voiced > 0.6 * ring_buffer.maxlen:
                    triggered = True
                    voiced_frames.extend([f for f, s in ring_buffer])
                    ring_buffer.clear()
            else:
                voiced_frames.append(data)
                if num_voiced < 0.2 * ring_buffer.maxlen:
                    silence_chunks += 1
                    if silence_chunks > silence_threshold:
                        triggered = False
                        break
                else:
                    silence_chunks = 0

        stream.stop_stream()
        stream.close()

        async with aiofiles.open(audio_path, 'wb') as wf:
            await wf.write(b''.join(voiced_frames))
        logger.debug('Audio recording completed and file saved.')
    except Exception as e:
        logger.error(f"An error occurred while recording audio: {e}")

def transcribe_audio():
    try:
        result = model.transcribe(audio_path)
        transcription = result['text']
        logger.debug(f'Audio transcription completed: {transcription}')
        return transcription
    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}")
        return ""

# Main execution
if __name__ == '__main__':
    import uvicorn
    config = uvicorn.Config(app_asgi, host="127.0.0.1", port=5000, log_level="info")
    server = uvicorn.Server(config)
    
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(server.serve())
    else:
        loop.run_until_complete(server.serve()) 
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
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import tool
from tools.initialize_cerebras import init_cerebras
from Utils_cerebras import (
    initialize_web_search_agent,
    initialize_pdf_search_agent,
    initialize_quote_bot,
    vector_embedding
)
from flask import Flask
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

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

# Define tools properly with clear return types
@tool
async def web_search(query: str) -> str:
    """Search the web for HVAC related information."""
    try:
        web_chain = await initialize_web_search_agent(llm)
        result = await web_chain.ainvoke({"input": query})
        return str(result["output"]) if "output" in result else str(result)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Web search error: {str(e)}"

@tool
async def pdf_search(query: str) -> str:
    """Search through HVAC documentation and manuals."""
    try:
        pdf_chain = await initialize_pdf_search_agent(llm, "", vector_embedding(), [])
        result = await pdf_chain.ainvoke({
            "input": query,
            "context": "",  # Add empty context if needed
            "messages": [{"role": "user", "content": query}]
        })
        return str(result["output"]) if "output" in result else str(result)
    except Exception as e:
        logger.error(f"PDF search failed: {e}")
        return f"PDF search error: {str(e)}"

# Create tools array
tools = [web_search, pdf_search]

# Define the state
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Create the main graph
graph = StateGraph(State)

# Define chatbot node with proper tool handling
async def chatbot(state: State) -> dict:
    """Main chatbot node that processes messages using tools"""
    try:
        messages = state["messages"]
        last_message = messages[-1].content.lower()
        
        # Check if message contains HVAC or technical terms
        hvac_keywords = ["hvac", "heating", "cooling", "ventilation", "manual", 
                        "system", "temperature", "maintenance", "fluctuation"]
        
        if any(keyword in last_message for keyword in hvac_keywords):
            try:
                # Use ainvoke instead of direct call
                pdf_result = await pdf_search.ainvoke(last_message)
                logger.info(f"PDF search result: {pdf_result}")
                
                if isinstance(pdf_result, str) and "error" in pdf_result.lower():
                    # Fallback to web search if PDF search fails
                    web_result = await web_search.ainvoke(last_message)
                    response_text = f"From web search: {web_result}"
                else:
                    response_text = f"From HVAC documentation: {pdf_result}"
                
                await sio.emit('bot_response', {'response': response_text})
                await sio.emit('message', {'text': response_text, 'type': 'bot'})
                
                return {"messages": [AIMessage(content=response_text)]}
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                raise
        
        # For general conversation, use the LLM directly
        llm_with_tools = llm.bind_tools(tools=tools)
        response = await llm_with_tools.ainvoke(messages)
        response_text = response.content
        
        await sio.emit('bot_response', {'response': response_text})
        await sio.emit('message', {'text': response_text, 'type': 'bot'})
        
        return {"messages": [response]}
        
    except Exception as e:
        error_msg = f"Error in chatbot node: {e}"
        logger.error(error_msg)
        await sio.emit('error', {'error': error_msg})
        return {"messages": [AIMessage(content="I encountered an error processing your request.")]}

# Create tool node with proper error handling
tool_node = ToolNode(tools=tools)

# Add nodes to graph
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)

# Add conditional edges with more specific routing
def route_message(state: State) -> str:
    """Route messages to appropriate node based on content"""
    if not state["messages"]:
        return "chatbot"
    
    last_message = state["messages"][-1].content.lower()
    if any(tool.name.lower() in last_message for tool in tools):
        return "tools"
    return "chatbot"

graph.add_conditional_edges(
    START,
    route_message,
    {
        "tools": "tools",
        "chatbot": "chatbot"
    }
)
graph.add_edge("tools", "chatbot")
graph.add_edge("chatbot", END)

# Compile graph
app_chain = graph.compile()
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
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
        
        # Initialize state with transcription
        initial_state = {
            "messages": [HumanMessage(content=transcription)]
        }
        
        # Execute workflow with timeout
        try:
            async with asyncio.timeout(30):
                result = await app_chain.ainvoke(initial_state)
        except asyncio.TimeoutError:
            logger.error("Workflow execution timed out")
            return jsonify({"error": "Request timed out"}), 504
        
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
        logger.info(f"Received text input: {text}")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Initialize state with the text input
        state = {
            "messages": [HumanMessage(content=text)]
        }
        
        try:
            async with asyncio.timeout(60):  # Increased timeout
                result = await app_chain.ainvoke(state)
                
                if result and "messages" in result and result["messages"]:
                    last_message = result["messages"][-1]
                    response_text = last_message.content
                    
                    logger.info(f"Generated response: {response_text[:100]}...")
                    
                    # Emit response through both SocketIO channels
                    await sio.emit('bot_response', {'response': response_text})
                    await sio.emit('message', {'text': response_text, 'type': 'bot'})
                    
                    # Synthesize speech
                    await synthesize_speech(response_text)
                    
                    return jsonify({
                        'response': response_text,
                        'status': 'success'
                    })
                    
        except asyncio.TimeoutError:
            error_msg = "Request timed out"
            await sio.emit('error', {'error': error_msg})
            logger.error("Workflow execution timed out")
            return jsonify({"error": error_msg}), 504
            
    except Exception as e:
        error_msg = str(e)
        await sio.emit('error', {'error': error_msg})
        logger.error(f"Error processing text input: {e}")
        return jsonify({"error": error_msg}), 500

@app.route('/get_audio')
async def get_audio():
    return await send_file(tts_synthesis_path, mimetype="audio/mp3")

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    await sio.emit('connected', {'status': 'connected'}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

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
    async def startup():
        await initialize_agents()
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(server.serve())
    else:
        loop.run_until_complete(server.serve()) 
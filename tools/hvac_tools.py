from langchain_core.tools import tool
from typing import Optional
from Utils_cerebras import (
    initialize_web_search_agent,
    initialize_pdf_search_agent,
    initialize_quote_bot,
    vector_embedding
)

@tool
async def web_search(query: str) -> str:
    """Search the web for general HVAC information."""
    state = {"messages": [HumanMessage(content=query)]}
    result = await web_search_graph.ainvoke(state)
    return result["messages"][-1].content

@tool
async def pdf_search(query: str) -> str:
    """Search HVAC documentation and manuals."""
    state = {"messages": [HumanMessage(content=query)]}
    result = await pdf_search_graph.ainvoke(state)
    return result["messages"][-1].content

@tool
async def generate_quote(request: str) -> str:
    """Generate an HVAC service quote based on requirements."""
    # Create a queue for the quote bot
    response_queue = queue.Queue()
    
    def input_func():
        return request
        
    def output_func(text: str):
        response_queue.put(text)
        
    # Initialize and run quote bot
    initialize_quote_bot(client, llm, input_func, output_func, response_queue)
    
    # Get final response
    final_response = response_queue.get()
    return final_response 
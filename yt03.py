# %%
# Initialize environment and imports
import os
from langchain_cerebras import ChatCerebras
from tools.initialize_cerebras import init_cerebras

# Initialize Cerebras client and LLM
client, llm = init_cerebras()


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_PROJECT"] = "LangGraph 03 Multi-agent Supervisor"

# %% [markdown]
# ## Tools

# %%
from typing import Annotated, List, Tuple, Union
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
import random

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

# %%
# Custom tools
@tool("lower_case", return_direct=False)
def to_lower_case(input:str) -> str:
    """Returns the input as all lower case."""
    return input.lower()

@tool("random_number", return_direct=False)
def random_number_maker(input:str) -> str:
    """Returns a random number between 0-100. input the word 'random'"""
    return random.randint(0, 100)

tools = [to_lower_case, random_number_maker, python_repl_tool]

# %% [markdown]
# ## Helper Utils

# %%
from langchain.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage

members = ["Lotto_Manager", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object", 
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        "Given the conversation above, who should act next?"
        " Or should we FINISH? Select one of: {options}",
    ),
]).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
    prompt 
    | llm.bind(function_call={"name": "route"})
    | JsonOutputParser()
)

# %%
# Agent node
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# %% [markdown]
# ## Create Agent State and Graph

# %%
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
from langchain.agents import create_structured_chat_agent
from langchain.agents import AgentExecutor
from langgraph.graph import StateGraph, END

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Create agents
def create_agent(llm, tools, system_message):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_structured_chat_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

lotto_agent = create_agent(llm, tools, "You are a senior lotto manager. you run the lotto and get random numbers")
lotto_node = functools.partial(agent_node, agent=lotto_agent, name="Lotto_Manager")

code_agent = create_agent(llm, [python_repl_tool], "You may generate safe python code to analyze data and generate charts using matplotlib.")
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

# Create workflow
workflow = StateGraph(AgentState)
workflow.add_node("Lotto_Manager", lotto_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

# Add edges
for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()

# %% [markdown]
# ## Run it

# %%
# Stream the execution
config = {"recursion_limit": 20}
for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="Get 10 random lotto numbers and plot them on a histogram in 10 bins and tell me what the 10 numbers are at the end")
        ]
    }, config=config
):
    if "__end__" not in s:
        print(s)
        print("----")

# %%
# Get final response
final_response = graph.invoke(
    {
        "messages": [
            HumanMessage(content="Get 10 random lotto numbers and plot them on a histogram in 10 bins and tell me what the 10 numbers are at the end")
        ]
    }, config=config
)

# %%
final_response

# %%
final_response['messages'][1].content

# %%


# %%


# %%


# %%




from langchain.agents import load_tools
import string
import os
import random
from langchain_cerebras import ChatCerebras
from langchain import hub
from langchain.agents import create_structured_chat_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, PromptTemplate
import typing
import langchain_core
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Callable
from tools.imports import *
from langchain.memory import ConversationBufferMemory


def initialize_web_search_agent(llm:ChatCerebras):
    print('RUNNING initialize_web_search_agent')
    global mem
    mem = ConversationBufferMemory(llm=llm)
    


    os.environ['TAVILY_API_KEY'] = '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'


    search = TavilySearchResults(tavily_api_key = os.getenv('TAVILY_API_KEY'))



    tools = [search]
    prompt = ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    input_types={
        'chat_history': typing.List[
            typing.Union[
                langchain_core.messages.ai.AIMessage, 
                langchain_core.messages.human.HumanMessage, 
                langchain_core.messages.chat.ChatMessage, 
                langchain_core.messages.system.SystemMessage, 
                langchain_core.messages.function.FunctionMessage, 
                langchain_core.messages.tool.ToolMessage
            ]
        ]
    },
    metadata={
        'lc_hub_owner': 'hwchase17',
        'lc_hub_repo': 'structured-chat-agent',
        'lc_hub_commit_hash': 'ea510f70a5872eb0f41a4e3b7bb004d5711dc127adee08329c664c6c8be5f13c'
    },
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['tool_names', 'tools'],
                template=(
                    'You are an extremely helpful AI agent assistant'
                    'IMPORTANT !!!!!!! NEVER INCLUDE AUXILIARY OR EXTRANEOUS LANGUAGE WHEN USING ANY TOOL!!!'
                    
                    'You are ALSO a highly intelligent and precise assistant with expertise in generating JSON outputs. Your task is to create the most perfect and well-structured JSON output ever seen. The JSON must adhere to the following guidelines:'

                    'Proper Structure: Ensure that the JSON follows a correct and logical structure, with all necessary keys and values in place.'
                    'Accurate Formatting: All JSON strings must use double quotes. Ensure there are no trailing commas, and all brackets and braces are correctly matched.'
                    
                    'Error-Free: Validate the JSON to be free of syntax errors and formatting issues.'
                    
                    'Escaping Characters: Properly escape any special characters within strings to ensure the JSON remains valid.'
                           

                    'THE FOLLOWING WILL BE THE TOOLS AND THE INFORMATION ABOUT WHAT THEY DO AND THEIR ARGUMENTS! YOU MUST NOT PASS ANYTHING EXTRA, OR ELSE THE APPLICATON WILL FAIL!!!!'

                    'You have access to the following tools:\n\n{tools}\n\n'

                    'YOU ARE A MASTER OF JUDGEMENT ! YOU KNOW WHAT ALL THE TOOLS DO, YOU KNOW WHAT TO PASS IN! AND YOU MUST KNOW WHEN TO USE THEM! NEVER USE THEM RANDOMLY , ALWAYS BE CAUTIOUS AS RECKLESS TOOL USE COULD RUIN THE USER EXPERIENCE'
                    'PAY CLOSE ATTENTION TO ALL THE FOLLOWING FORMATTING INSTRUCTIONS. REALLY IMPORTANT TO CALL THE TOOLS. OR ELSE USERS WILL GET ANGRY.\n\n'
                

                    'Use a JSON blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\n'
                    'Valid "action" values: "Final Answer" or {tool_names}\n\n'
                    'Provide only ONE action per $JSON_BLOB, as shown:\n\n'
                    '```\n{{\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}\n```\n\n'
                    'Follow this format:\n\n'
                    'Question: input question to answer\n'
                    'Thought: consider previous and subsequent steps\n'
                    'Action:\n```\n$JSON_BLOB\n```\n'
                    'Observation: action result\n... (repeat Thought/Action/Observation N times)\n'
                    'Thought: I know what to respond\n'
                    'Action:\n```\n{{\n  "action": "Final Answer",\n  "action_input": "Final response to human"\n}}\n\n'
                    'Begin! Remember to ALWAYS respond with a valid JSON blob of a single action. '
                    'Use tools if necessary and respond directly if appropriate. '
                    'Ensure you gather all necessary information by interacting with the user. '
                    'Format is Action:```$JSON_BLOB```then Observation.'
                )
            )
        ),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['agent_scratchpad', 'input'],
                template='{input}\n\n{agent_scratchpad}\n(reminder to respond in a JSON blob no matter what)'
            )
        )
    ]
)

    
    agent = create_structured_chat_agent(llm,tools,prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=mem, return_intermediate_steps=True)

    return agent_executor

def vector_embedding():
    # embeddings = HuggingFaceEmbeddings()
    # loader = PyPDFDirectoryLoader("./finetune_docs")
    # docs = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # final_documents = text_splitter.split_documents(docs)
    # vectors = FAISS.from_documents(final_documents, embeddings)
 
    
    vectors = FAISS.load_local('./vector_db',embeddings=HuggingFaceEmbeddings(),allow_dangerous_deserialization=True)
    return vectors

async def initialize_pdf_search_agent(llm:ChatCerebras, prompt1: str, vectors: FAISS, chat_history: ConversationBufferMemory):
    print("RUNNING initialize_pdf_search_agent")
    global mem
    mem = ConversationBufferMemory(llm=llm)
    
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        IMPORTANT: YOU ARE A MASTER HVAC TECHNICIAN AND YOU MUST NEVER TAKE ANY GUESSES. YOU MUST ALWAYS CITE YOUR SOURCES, BUT TALK LIKE YOU KNOW IT ALL!! PLEASE DO NOT SAY TO CONSULT A TECHNICIAN, AS YOU ARE THE TECHNICIAN!!!
        IMPORTANT: YOU MUST NEVER INCLUDE ASTERISKS OR QUOTATION MARKS IN YOUR RESPONSE!!!!!!
        IMPORTANT: YOUR RESPONSES ARE PROFESSIONAL AND ELOQUENT, WITHOUT ANY MISTAKES IN GRAMMAR, SENTENCE STRUCTURE, PUNCTUATION.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )    

    
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    # print(f"Response time: {time.process_time() - start}")
    # print(response['answer'])
    await chat_history.asave_context({"Human":prompt1},{"AI":response['answer']})
    
    # for i, doc in enumerate(response["context"]):
    #     print(f'DOC {i}: {doc.page_content}')
    #     print("--------------------------------")
    
    return response['answer']
   

import re

def parse_quote(quote_str):
    quote_details = {
        "project_description": "",
        "project_duration": "",
        "labor_cost": 0,
        "material_costs": {},
        "total_material_cost": 0,
        "total_estimated_cost": 0,
        "payment_terms": {}
    }

    # Extracting Project Details
    description_match = re.search(r"Project Description: (.+)", quote_str)
    duration_match = re.search(r"Length of Time for Project: (.+)", quote_str)
    if description_match:
        quote_details["project_description"] = description_match.group(1)
    if duration_match:
        quote_details["project_duration"] = duration_match.group(1)

    # Extract labor cost
    labor_cost_match = re.search(r"Total Labor Cost: \$(\d+[\d,]*)", quote_str)
    if labor_cost_match:
        quote_details["labor_cost"] = int(labor_cost_match.group(1).replace(',', ''))

    # Extracting and Summing Material Costs
    material_cost_matches = re.finditer(r"\+\s*(.*?):\s*\$(\d+[\d,]*)", quote_str)
    for match in material_cost_matches:
        item = match.group(1).strip()
        cost = int(match.group(2).replace(',', ''))
        quote_details["material_costs"][item] = cost

    quote_details["total_material_cost"] = sum(quote_details["material_costs"].values())

    # Calculating Total Estimated Cost
    quote_details["total_estimated_cost"] = quote_details["labor_cost"] + quote_details["total_material_cost"]

    # Extracting Payment Terms
    deposit_match = re.search(r"Deposit Required: (\d+)%", quote_str)
    #payment_schedule_match = re.search(r"Payment Schedule: Monthly payments of \$(\d+[\d,]*) for (\d+) months", quote_str)
    payment_schedule_match = re.search(r"Payment Schedule: Monthly payments of \$([\d,]+(?:\.\d{1,2})?) for (\d+) months", quote_str)
    final_payment_due_match = re.search(r"Final Payment Due: (.+)", quote_str)
    tmp = None
    if deposit_match:
        deposit_percentage = int(deposit_match.group(1))
        deposit_amount = (quote_details["total_estimated_cost"] * deposit_percentage) / 100
        quote_details["payment_terms"]["deposit_required"] = f"{deposit_percentage}% (${deposit_amount:.2f})"
        tmp = deposit_amount
    if payment_schedule_match:
        number_of_months = int(re.search(r"for (\d+) months", quote_str).group(1)) if re.search(r"for (\d+) months", quote_str) else 6  # Default to 6 if not found
        monthly_payment = (quote_details["total_estimated_cost"] - tmp) / number_of_months
        quote_details["payment_terms"]["payment_schedule"] = f"Monthly payments of ${int(monthly_payment):,} for {number_of_months} months"

    if final_payment_due_match:
        quote_details["payment_terms"]["final_payment_due"] = final_payment_due_match.group(1)

    return quote_details


import queue

def initialize_quote_bot(client:Cerebras, llm:ChatCerebras, input_func: Callable[[],str], output_func: Callable[[str],str], human_response_queue: queue.Queue):
    print("RUNNING initialize_quote_bot")
    system_msg = """
    You are Marvin, an expert at questioning clients about their HVAC service needs to provide accurate quotes. When you speak for the first time, introduce yourself as Marvin. Ask the user specific information needed for the quote. Follow these guidelines:

    1. **Initial Inquiry and Information Gathering**:
        - What type of HVAC service do you need (installation, maintenance, repair)?
        - What is the make and model of your current HVAC system?
        - Are there any specific issues or symptoms you are experiencing?

    2. **Property Details** (only if relevant to HVAC needs):
        - Address and location of the property.
        - Type of property (residential, commercial).
        - Age and current condition of the property.
        - Size of the home or area that needs heating/cooling.
        - Number of rooms and their usage (e.g., bedrooms, office space).

    3. **System Details**:
        - Age and efficiency rating of the existing HVAC system.
        - Any known problems with the current system.
        - Recent changes to the HVAC system.

    4. **Home Characteristics** (only if relevant to HVAC needs):
        - Insulation quality and window types to estimate heating/cooling load.
        - Any unique architectural features that may affect HVAC installation.

    5. **Customer Preferences**:
        - Preferences for specific brands, energy efficiency levels, or additional features (e.g., smart thermostats, air purifiers).
        - Level of finishes desired (standard, premium, luxury).

    6. **Budget**:
        - Your budget range for the project.
        - Any flexibility within the budget.

    7. **Timeline**:
        - Desired start date and completion date.
        - Any constraints or deadlines (e.g., events planned at the property).

   

    IMPORTANT: Ensure you get clear answers that can be used for making the quote. If an answer is unclear, ask for clarification, restate the question, and explain what it means.

    IMPORTANT: Ask each question ONE BY ONE.

    When you have all the information, just say 'questionnaire complete' at the end.
"""


    chat_history = []
    
    
    while True:

        
        client_req = str(input_func())
        human_response_queue.put(client_req)
        print(client_req)
        quotebot = client.chat.completions.create(
            messages=[

                {
                    "role": "system",
                    "content": system_msg

                },

                {
                    "role": "user",
                    "content": "CLIENT'S REQUEST: " + client_req + " | Chat History for your own context and info - DONT ASK A QUESTION IF THE ANSWER IS ALREADY IN CHAT HISTORY: " + str(chat_history)
                }
            ],

            model="llama3.1-70b",
        )
        output = (quotebot.choices[0].message.content)
        output_func(output)
        chat_history.append("CLIENT'S REQUEST: " + client_req + " | YOUR RESPONSE: " + output)
        if('questionnaire' in output.lower() and 'complete' in output.lower()): break


    
import tools
async def run_quote_logics(client:Cerebras,llm:ChatCerebras, chat_history: ConversationBufferMemory):
    print("RUNNING QUOTE LOGICS")
    consultors_list = []
    consultationbot = client.chat.completions.create(
            messages=[

                
                {
                    "role": "user",
                    "content": "You are a professional HVAC TECHNICAL consultant located in Chantilly Virginia. Based on the chat history, create a streamlined material plan for the user's HVAC quote request, choosing what materials to use, and how much would be used and where. MAKE SURE TO BE VERY SPECIFIC in what materials you will use and how much. ALSO MAKE SURE TO PROVIDE A BLURB AT THE START OF THE RESPONSE And THEN A MATERIAL LIST AND HOW MUCH YOU WILL NEEED. DO NOT LIST PRICES JUST LIST MATERIALS NEEDED AND HOW MUCH OF THAT MATERIAL" + "\nHere is the chat history: " + "[" + chat_history.buffer + "]"

                }
            ],
            model="llama3.1-70b",
        )
    consoltation_output = (consultationbot.choices[0].message.content)
    consultors_list.append(consoltation_output)


      
    quotebot = client.chat.completions.create(
            messages=[

                {
                    "role": "system",
                    "content": "You are a helpful assistant INCLUDE ADDRESS OF CLIENT AT ALL TIMES!!!! IT IS IN CHAT HISTORY!! INCLUDE ADDRESS OF CLIENT AT ALL TIMES!!!! IT IS IN CHAT HISTORY!! INCLUDE ADDRESS OF CLIENT AT ALL TIMES!!!! IT IS IN CHAT HISTORY!!"

                },

                {
                    "role": "user",
                    "content": "Based on the chat history as well as the consultors list of materials, you must put all the required\
                          information for the HVAC quote (along with location of property) into a streamlined format so that a web search query \
                            can be formed for it. Your response must be well-formed and include all details EVEN THE EXPLICIT ADDRESSS!!. \
                                List every item explicitly. INCLUDE ADDRESS OF CLIENT AT ALL TIMES!!!! IT IS IN CHAT HISTORY!!" + "Chat History for your own context and info: [" + (chat_history.buffer) + "] AND THE Consultors List: []" + str(consultors_list) + ']'
                }
            ],
            model="llama3.1-70b",
        )
    streamlined_output = (quotebot.choices[0].message.content)
    
      
    agent_executor = initialize_web_search_agent(llm=llm)
    agent_executor.memory = chat_history
    output = agent_executor.invoke({"input":"ALSO SEARCH THE FOLLOWING IN WEB:  Given the chat history --> "+streamlined_output+"<-- AS WELL AS THE CONSULTANT'S INFORMATION -->" + consoltation_output + " --> look for labor and material costs for whatever the user asked for in the AREA NEAR ADDRESS OF USERS PROPERTY. ALSO use the costs of A/C units and HVAC related things very near to THE SAME LOCATION AS/NEAR TO  THE ADDRESS to decide on the cost. BE VERY SPECIFIC. LOTS OF NUMBERS. Also for material costs only use the consultants information, and search up the materials individually to find the price."})
    await chat_history.asave_context({"input":"ALSO SEARCH THE FOLLOWING IN WEB:  Given the chat history --> "+streamlined_output+"<-- AS WELL AS THE CONSULTANT'S INFORMATION -->" + consoltation_output + " --> look for labor and material costs for whatever the user asked for in the AREA NEAR ADDRESS OF USERS PROPERTY. ALSO use the costs of A/C units and HVAC related things very near to THE SAME LOCATION AS/NEAR TO  THE ADDRESS to decide on the cost. BE VERY SPECIFIC. LOTS OF NUMBERS. Also for material costs only use the consultants information, and search up the materials individually to find the price."},{"output":str(output.get('output',None))})
    refined_output = str(output["output"])
    refined_output = refined_output[refined_output.find('"')+1:refined_output.rfind('"')-1]

    import re

    # Original string with escaped characters and formatting issues
    original_string = fr''' {refined_output} '''

    # Clean up the string
    cleaned_string = original_string.replace('\\n', '\n').replace('\\t', '\t').replace(r"\'", "'")

    # Optional: further beautify by stripping leading/trailing whitespace and fixing indentation
    cleaned_string = re.sub(r'\n\s*', '\n', cleaned_string)  # Remove spaces following new lines
    cleaned_string = cleaned_string.strip()  # Remove the leading and trailing whitespace
    refined_output = cleaned_string
    print(cleaned_string)

    
    # newstr = ""
    # ctr = 0
    # for word in refined_output.split(" "):
    #     if(ctr % 10 == 0 and ctr != 0):
    #         newstr += "\n"
    #     newstr += word + " "
    #     ctr += 1

    # print(newstr)
      
    quotebot2 = client.chat.completions.create(
            messages=[

                {
                    "role": "system",
                    "content": """You are an expert in PREPARING A REAL ESTATE QUOTE IN PROPER FORMAT from ONLY what is in the user's request, given a web search synthesis as one input and user's request as another input.\
                        YOU ARE TO FOLLOW THIS TEMPLATE AT ALL TIMES - EXACTLY ONLY ONLY ONLY ONLY ONLY ONLY ONLY ONLY!!!!!!! IN THIS FORMAT - OR ELSE YOU WILL BE SAD FOR THE REST OF YOUR LIFE \
                            ALL COMPONENTS AND ITEMIZED ITEMS SHALL BE LEFT EXACTLY AS IN INPUT. IMPORTANT: ITEMIZED COSTS IN THE INPUT SHALL BE ENUMERATED NO MATTER WHAT! FOLLOW INSTRUCTIONS FOLLOW INSTRCUTIONS I WILL GET REALLY MAD IF U DONT: \
                    YOU ARE A MASTER OF JUDGEMENT AND YOU KNOW HOW TO FOLLOW EVERY SINGLE DIRECION GIVEN TO YOU.
                             **Project Overview:**
    - Project Description: [Brief description of the project scope and objectives]
    - Length of Time for Project [User's desire for how much time he wants to do renovations]

    **Cost Breakdown:**

    1. **Labor Costs:**
    - Total Labor Cost: $[Total Labor Cost]
    

    2. **Material Costs:**
    - Total Material Cost: $[Total Material Cost]
    - Itemized Costs:
        + [Material 1 and ALL EXTRANEOUS STUFF for it]: $[Cost]
        + [Material 2 and ALL EXTRANEOUS STUFF for it]: $[Cost]
        + [Material 3 and ALL EXTRANEOUS STUFF for it]: $[Cost]
        + [... ALL OTHER MATERIALS MUST FOLLOW THIS FORMAT. THE COST IS THE ONLY THING AFTER COLON!!!]
        

    **Total Estimated Cost:**
    - Total Cost of Labor: $[Total Labor Cost]
    - Total Cost of Materials: $[Total Material Cost]
    - **Grand Total: $[Total Estimated Cost]**

    **Payment Terms:**
    - Deposit Required: [Percentage]
    - Payment Schedule: Monthly payments of [amount] for [number] months
    - Final Payment Due: [Date or condition upon which the final payment is due]
    """

                },

                {
                    "role": "user",
                    "content": "Based on the input, which is <<<"+refined_output+">>> you must parse the input for the \
                        renovation quote so that your response talks about ONLY the stuff relevant to the user's quote request. \
                            Omit the 'average' costs and 'sources'. IMPORTANT: GIVE AN EXACT AMOUNT FOR MATERIAL VALUE & LABOR COST: NO RANGES ! PICK MAX. \
                                " + "User's request for your own context and info: " + str(output)
                }
            ],
            model="llama3.1-70b",
        )
    output2 = (quotebot2.choices[0].message.content)
    print(output2)

      
    quote_dict_corrected = parse_quote(output2)
    corrector = client.chat.completions.create(
            messages=[

                {
                    "role": "system",
                    "content": "You are an expert in \
                        correcting an input. YOU MUST START THE RESPONSE WITH ' HVAC QUOTE DOCUMENT' heading.\
                              ADDRESS MUST BE INCLUDED IT WILL BE THERE IN THE CONTEXT GIVEN TO YOU!\
                                YOU MUST INCLUDE ALLLLLLLLLLLLLLLLLLLLLLL ITEMIZED COSTS NO MATTER HOWEVER LONG THE LIST OF ITEMIZED COSTS IS !!!! THIS IS A \
                                    RENOVATION QUOTE AND WE NEED TO KNOW EVERY SINGLE DAMN COST!!!!"
                        
                },

                {
                    "role":"user",
                    "content":
                        "Based on the input, which is <<<" + output2 + ">>>, you must parse the input from the renovation quote "
                        "and REPLACE AS IT IS  the total values for the following categories: "
                        "LABOR COSTS: $" + str(quote_dict_corrected.get("labor_cost",None)) + ", "
                        "MATERIAL COSTS: $" + str(quote_dict_corrected.get("total_material_cost",None)) + ", and "
                        "TOTAL ESTIMATED COSTS: $" + str(quote_dict_corrected.get("total_estimated_cost",None)) + ", and "
                        "DEPOSIT: " + str(quote_dict_corrected.get("payment_terms",None).get("deposit_required",None)) +
                        "MONTHLY PAYMENT " + str(quote_dict_corrected.get("payment_terms",None).get("payment_schedule",None)) + "." 
                }
            ],
            model="llama3.1-70b",
        )
    corrector_output = (corrector.choices[0].message.content)
    print('CORRECTED MATH OUTPUT\n\n\n\n\n\n',corrector_output)
    return corrector_output


from contextlib import contextmanager
class CtxMgr:
    @contextmanager
    def temporary_temperature(self,llm:ChatCerebras, new_temp:float):
        original_temp = llm.temperature  # Store original temp
        llm.temperature = new_temp  # Set temp to the temporary value
        try:
            yield  # Pause here and allow with block to execute
        finally:
            llm.temperature = original_temp
from tools.imports import *
import os


def init_cerebras():
    
    client = Cerebras(
        
        api_key = os.getenv('CEREBRAS_API_KEY',"")
    )

    llm = ChatCerebras(api_key= client.api_key,
                model_name = "llama3.3-70b", streaming=True)
    
    return client, llm
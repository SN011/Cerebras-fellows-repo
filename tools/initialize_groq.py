from tools.imports import *

api_keys = ['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','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',
                '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','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']

def init_groq():
    
    client = Groq(
        
        api_key = random.choice(api_keys)
    )

    llm = ChatGroq(groq_api_key = client.api_key,
                model_name = "llama3-70b-8192", streaming=True)
    
    return client, llm
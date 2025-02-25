from groq import Groq
from langchain_groq import ChatGroq
import random

api_keys = ['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','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',
                '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','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']

def init_groq(model_name = "llama-3.3-70b-versatile"):
    """
    Supported Models
GroqCloud currently supports the following models:


Production Models
Note: Production models are intended for use in your production environments. They meet or exceed our high standards for speed and quality.

MODEL ID	DEVELOPER	CONTEXT WINDOW (TOKENS)	MAX COMPLETION TOKENS	MAX FILE SIZE	MODEL CARD LINK
distil-whisper-large-v3-en	HuggingFace	-	-	25 MB	
Card
gemma2-9b-it	Google	8,192	-	-	
Card
llama-3.3-70b-versatile	Meta	128k	32,768	-	
Card
llama-3.1-8b-instant	Meta	128k	8,192	-	
Card
llama-guard-3-8b	Meta	8,192	-	-	
Card
llama3-70b-8192	Meta	8,192	-	-	
Card
llama3-8b-8192	Meta	8,192	-	-	
Card
mixtral-8x7b-32768	Mistral	32,768	-	-	
Card
whisper-large-v3	OpenAI	-	-	25 MB	
Card
whisper-large-v3-turbo	OpenAI	-	-	25 MB	
Card

Preview Models
Note: Preview models are intended for evaluation purposes only and should not be used in production environments as they may be discontinued at short notice.

MODEL ID	DEVELOPER	CONTEXT WINDOW (TOKENS)	MAX COMPLETION TOKENS	MAX FILE SIZE	MODEL CARD LINK
deepseek-r1-distill-llama-70b-specdec	DeepSeek	128k	16,384	-	
Card
deepseek-r1-distill-llama-70b	DeepSeek	128k	-	-	
Card
llama-3.3-70b-specdec	Meta	8,192	-	-	
Card
llama-3.2-1b-preview	Meta	128k	8,192	-	
Card
llama-3.2-3b-preview	Meta	128k	8,192	-	
Card
llama-3.2-11b-vision-preview	Meta	128k	8,192	-	
Card
llama-3.2-90b-vision-preview	Meta	128k	8,192	-	
Card

Deprecated models are models that are no longer supported or will no longer be supported in the future. A suggested alternative model for you to use is listed for each deprecated model. See our deprecated models here 


Hosted models are directly accessible through the GroqCloud Models API endpoint using the model IDs mentioned above. You can use the https://api.groq.com/openai/v1/models endpoint to return a JSON list of all active models:

Python
JavaScript
curl

import requests
import os

api_key = os.environ.get("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())
    """
    client = Groq(
        
        api_key = random.choice(api_keys)
    )

    llm = ChatGroq(groq_api_key = client.api_key,
                model_name = model_name, streaming=True)
    

    return client, llm
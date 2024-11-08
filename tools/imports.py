import os
import time
import re
import json
import base64
import random
import smtplib
import mimetypes

#Email Related Imports
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

#Google OAUTH Imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import googleapiclient.errors
import google.auth.exceptions
from googleapiclient.http import MediaFileUpload

#Langchain Imports
from langchain.tools import BaseTool
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import FAISS
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents import initialize_agent, create_structured_chat_agent, AgentType

#Pydantic
from pydantic import BaseModel, Field, Extra

#Writing to local word document [docx]
from docx import Document

#Groq Related Imports
from groq import Groq
from langchain_groq import ChatGroq
from langchain_cerebras import ChatCerebras
from cerebras.cloud.sdk import Cerebras
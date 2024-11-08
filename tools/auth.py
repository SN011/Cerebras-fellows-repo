import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv

load_dotenv()

SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/calendar',
]

CREDENTIALS_PATH = os.getenv('OLD_CREDENTIALS_PATH')
TOKEN_PATH = 'paths/token.json'

def authenticate():
    print('Authenticating...')
    creds = None

    # Load existing credentials from the token file
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    # If there are no valid credentials, initiate the OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Failed to refresh token: {e}")
                os.remove(TOKEN_PATH)  # Delete the token file if refresh fails
                creds = None  # Set creds to None to initiate a new OAuth flow
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(TOKEN_PATH, 'w') as token_file:
                token_file.write(creds.to_json())

    return creds

# Ensure no default credentials are being used
if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
    del os.environ['GOOGLE_APPLICATION_CREDENTIALS']









# import os
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import Flow
# from google.auth.transport.requests import Request
# from dotenv import load_dotenv

# load_dotenv()

# SCOPES = [
#     'https://www.googleapis.com/auth/drive',
#     'https://www.googleapis.com/auth/documents',
#     'https://www.googleapis.com/auth/spreadsheets',
#     'https://www.googleapis.com/auth/gmail.send',
#     'https://www.googleapis.com/auth/calendar',
# ]

# CREDENTIALS_PATH = os.getenv('CREDENTIALS_PATH')
# TOKEN_PATH = 'paths/token.json'
# os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
# def get_auth_flow():
#     return Flow.from_client_secrets_file(
#         CREDENTIALS_PATH, scopes=SCOPES,
#         redirect_uri='https://hypnos.site/oauth2callback')

# def authenticate():
#     creds = None

#     # Load existing credentials from the token file
#     if os.path.exists(TOKEN_PATH):
#         creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

#     # If there are no valid credentials, initiate the OAuth flow
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             try:
#                 creds.refresh(Request())
#             except Exception as e:
#                 print(f"Failed to refresh token: {e}")
#                 os.remove(TOKEN_PATH)  # Delete the token file if refresh fails
#                 creds = None  # Set creds to None to initiate a new OAuth flow
#         if not creds:
#             flow = get_auth_flow()
#             auth_url, _ = flow.authorization_url(prompt='consent')
#             return auth_url

#     return creds

from tools.imports import *
from googleapiclient.http import MediaFileUpload
from tools.auth import authenticate
from tools.file_mgmt_tools import DriveDictUpdateTool

class GoogleDriveUploadTool(BaseTool):
    name:str = "GoogleDriveUploadTool"
    
    description :str =("Uploads a PDF to Google Drive and sets permissions for a specific user. "
                   "Please set the 'rename' parameter to None if the user does not want to rename the file before uploading "
                   "to Google Drive. STOP AFTER ONE-TIME SUCCESSFUL EXECUTION")

    credentials_path: str = Field(..., description="Path to the credentials JSON file")
    
    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.service = build('drive', 'v3', credentials=self.creds)

    def upload_file(self, file_path: str, user_email: str, rename: str = None):
        """Upload a file to Google Drive and set permissions."""
        file_metadata = {'name': rename if rename else os.path.basename(file_path)}

        if not os.path.exists(file_path):
            return f"File not found in the LOCAL SYSTEM: {file_path}"
        
        mime_type, _ = mimetypes.guess_type(file_path)
        print('THE MIMETYPE MIGHT BE: ', mime_type)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if unknown

        media = MediaFileUpload(file_path, mimetype=mime_type)
        try:
            file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            file_id = file.get('id')
            with open('file_id_history.txt', 'w') as id_hist_file:
                id_hist_file.write(file_id)
            print(f"File ID: {file_id}")
            res = DriveDictUpdateTool(self.credentials_path).update_with_new_item(file_id)
            # permission = {
            #     'type': 'user',
            #     'role': 'writer',
            #     'emailAddress': user_email,
            # }

            # try:
            #     self.service.permissions().create(
            #         fileId=file_id,
            #         body=permission,
            #         fields='id',
            #         sendNotificationEmail=False
            #     ).execute()
            #     print(f"Granted {permission['role']} access to {user_email}")
            # except Exception as e:
            #     print(f"Error setting permission: {e}")
            #     return f"Error setting permission: {e}"

            return f"File uploaded successfully. File ID: {file_id}. PLEASE STOP WORKING! THE TOOL HAS BEEN EXECUTED PROPERLY!"
        except Exception as e:
            return f"Error during file upload or conversion: {e}"

    def _run(self, file_path: str, user_email: str, rename: str = None):
        """Run the tool to upload the file to Google Drive and set permissions."""
        result = self.upload_file(file_path, user_email, rename)
        return result

    def _arun(self, file_path: str, user_email: str, rename: str = None):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")


class GoogleSheetsUpdateTool(BaseTool):
    name:str = "GoogleSheetsUpdateTool"
    
    description :str =("Appends rows to a specified range in a Google Sheets spreadsheet.")

    credentials_path: str = Field(..., description="Path to the credentials JSON file")

    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.service = build('sheets', 'v4', credentials=self.creds)

    def read_file_content(self, file_path):
        with open(file_path, 'r') as file:
            content = file.read().strip()
        return content

    def append_row_to_google_sheets(self, spreadsheet_id, range_name, values):
        """Appends rows to the specified range in the Google Sheets spreadsheet."""
        sheet = self.service.spreadsheets()
        body = {'values': values}
        result = sheet.values().append(
            spreadsheetId=spreadsheet_id, range=range_name,
            valueInputOption="RAW", body=body,
            insertDataOption="INSERT_ROWS"
        ).execute()
        print(f"{result.get('updates').get('updatedCells')} cells appended.")

    def _run(self, spreadsheet_name: str = None, spreadsheet_id: str = None, values: list = None, range_name: str = 'Sheet1', **kwargs):
        """Run the tool to append rows with the provided values to the specified range."""
        if not spreadsheet_id:
            return f"Use ImprovedSearchTool to find ID for the file (spreadsheet) by name {spreadsheet_name}, and pass it as an argument to this tool under 'spreadsheet_id' parameter. Be sure to pass in the spreadsheet_name as well."
        else:
            self.append_row_to_google_sheets(spreadsheet_id, range_name, values)
        return "Rows appended successfully. YOU ARE DONE APPENDING ROWS TO THE SHEET!! THE TOOL HAS EXECUTED SUCCESSFULLY!!!!!!!"

    def _arun(self, spreadsheet_id: str, values: list, range_name: str = 'Sheet1'):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")
    
    
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
from tools.file_mgmt_tools import ImprovedSearchTool
class GoogleSheetsCreateTool(BaseTool):
    name:str = "GoogleSheetsCreateTool"
    description :str =("Creates a new Google Sheets spreadsheet with specified column headers.")

    credentials_path: str = Field(..., description="Path to the credentials JSON file")

    class Config:
        extra = Extra.allow
        
    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.service = build('sheets', 'v4', credentials=self.creds)
        self.IST = ImprovedSearchTool(credentials_path=self.credentials_path)

    def create_google_sheet(self, title, headers):
        """Creates a new Google Sheets spreadsheet with the specified title and headers."""
        
        
        sheet = self.service.spreadsheets()
        spreadsheet = {
            'properties': {
                'title': title
            },
            'sheets': [
                {
                    'data': [
                        {
                            'startRow': 0,
                            'startColumn': 0,
                            'rowData': [
                                {
                                    'values': [{'userEnteredValue': {'stringValue': header}} for header in headers]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        result = sheet.create(body=spreadsheet).execute()
        spreadsheet_id = result.get('spreadsheetId')
        print(f"Spreadsheet ID: {spreadsheet_id}")
        return spreadsheet_id

  
    
    def _run(self, title: str, headers: list, **kwargs):
        """Run the tool to create a new Google Sheet with the specified title and headers."""
        
        # result = self.IST._run(name=title)
        # if 'single' in result.lower() or 'multiple' in result.lower():
        #     return result
        spreadsheet_id = self.create_google_sheet(title, headers)
        res = DriveDictUpdateTool(self.credentials_path).update_with_new_item(spreadsheet_id)
        return f"Spreadsheet created successfully with ID: {spreadsheet_id}. IF PREVIOUSLY INSTRUCTED TO POPULATE DATA, USE GOOGLE SHEETS UPDATE TOOL!! THIS TOOL (GOOGLE SHEETS CREATE TOOL) DOES NOT TAKE ANY VALUES. IT ONLY INPUTS THE HEADERS INTO THE SHEET!!!!"

    def _arun(self, title: str, headers: list):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")


SCOPES = ['https://www.googleapis.com/auth/gmail.send']

class GmailSendPdfTool(BaseTool):
    name:str = "GmailSendPdfTool"
    description :str ="Sends an email with an optional PDF attachment using Gmail API."

    credentials_path: str = Field(..., description="Path to the credentials JSON file")

    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.service = build('gmail', 'v1', credentials=self.creds)

    def send_email(self, sender_email, recipient_email, subject, body, pdf_path=None):
        """Send an email with an optional PDF attachment."""
        message = MIMEMultipart()
        message['to'] = recipient_email
        message['from'] = sender_email
        message['subject'] = subject
        body_part = MIMEText(body, 'plain')
        message.attach(body_part)
        
        if pdf_path:
            part = MIMEBase('application', 'octet-stream')
            try:
                with open(pdf_path, 'rb') as file:
                    part.set_payload(file.read())
            except: 
                return "INVALID FILE PATH. WHATEVER YOU HAVE INPUTTED IS A WRONG PATH. PATH MEANS A FILE PATH ON THE LOCAL MACHINE/COMPUTER."
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(pdf_path)}"')
            message.attach(part)
       
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        raw_message = {'raw': raw_message}

        try:
            message = (self.service.users().messages().send(userId="me", body=raw_message).execute())
            print(f'Message Id: {message["id"]}')
            return message
        except HttpError as error:
            print(f'An error occurred: {error}')
            return f'An error occurred: {error}'

    def _run(self, sender_email: str, recipient_email: str, subject: str, body: str, pdf_path: str = None, **kwargs):
        """Run the tool to send the email with the optional PDF attachment."""
        result = self.send_email(sender_email, recipient_email, subject, body, pdf_path)
        return f"Email sent successfully with message ID {result['id']}. YOU ARE DONNNNEEEEEEEEEEEEE!!!!!!!! TOOL EXECUTED SUCCESSFULLY!!!!!!!!!!!!!" if result else "Failed to send email."

    def _arun(self):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")


import datetime

class AppointmentBookingCalendarTool(BaseTool):
    name:str = "AppointmentBookingCalendarTool"
    description :str ="Books appointments by confirming dates in Google Calendar via OAuth, then confirms with the user, and then populates the correct calendar cell with all details provided by the user."

    credentials_path: str = Field(..., description="Path to the credentials JSON file")

    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.service = build('calendar', 'v3', credentials=self.creds)

    def book_appointment(self, summary: str, location: str, description: str, start_time: str, end_time: str):
        if not start_time or not end_time:
            return "Start time and end time must be provided."

        # Convert times to RFC3339 format
        start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").isoformat()
        end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").isoformat()

        event = {
            'summary': summary,
            'location': location,
            'description': description,
            'start': {
                'dateTime': start_time,
                'timeZone': 'America/New_York',
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'America/New_York',
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},
                    {'method': 'popup', 'minutes': 10},
                ],
            },
        }

        # Insert the event into the calendar
        event_result = self.service.events().insert(calendarId='primary', body=event).execute()

        return f"Event created: {event_result.get('htmlLink')}. PLEASE TELL THE HUMAN."

    def list_appointments(self, time_min: str = None, time_max: str = None, max_results: int = 10):
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        if time_min is None:
            time_min = now
        events_result = self.service.events().list(
            calendarId='primary', timeMin=time_min, timeMax=time_max,
            maxResults=max_results, singleEvents=True,
            orderBy='startTime').execute()
        events = events_result.get('items', [])

        if not events:
            return 'No upcoming events found.'

        event_list = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            event_list.append(f"{start}: {event['summary']}. PLEASE TELL THE HUMAN.")

        return "\n".join(event_list)

    def update_appointment(self, event_id: str, updated_summary: str = None, updated_location: str = None,
                           updated_description: str = None, updated_start_time: str = None, updated_end_time: str = None):
        event = self.service.events().get(calendarId='primary', eventId=event_id).execute()

        if updated_summary:
            event['summary'] = updated_summary
        if updated_location:
            event['location'] = updated_location
        if updated_description:
            event['description'] = updated_description
        if updated_start_time:
            event['start']['dateTime'] = datetime.datetime.strptime(updated_start_time, "%Y-%m-%d %H:%M:%S").isoformat()
        if updated_end_time:
            event['end']['dateTime'] = datetime.datetime.strptime(updated_end_time, "%Y-%m-%d %H:%M:%S").isoformat()

        updated_event = self.service.events().update(calendarId='primary', eventId=event['id'], body=event).execute()

        return f"Event updated: {updated_event.get('htmlLink')}. PLEASE TELL THE HUMAN."

    def delete_appointment(self, event_id: str):
        self.service.events().delete(calendarId='primary', eventId=event_id).execute()
        return "Event deleted.  PLEASE TELL THE HUMAN."

    def check_availability(self, start_time: str, end_time: str):
        time_min = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").isoformat() + 'Z'
        time_max = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").isoformat() + 'Z'

        events_result = self.service.events().list(
            calendarId='primary', timeMin=time_min, timeMax=time_max,
            singleEvents=True, orderBy='startTime').execute()
        events = events_result.get('items', [])

        if events:
            return False, events
        return True, []

    def _run(self, action: str, summary: str = '', location: str = '', description: str = '',
             start_time: str = '', end_time: str = '', time_min: str = None, time_max: str = None,
             max_results: int = 10, event_id: str = '', updated_summary: str = None,
             updated_location: str = None, updated_description: str = None,
             updated_start_time: str = None, updated_end_time: str = None):
        if action == 'book':
            if not start_time or not end_time:
                return "Start time and end time must be provided."

            available, conflicting_events = self.check_availability(start_time, end_time)
            if not available:
                conflicts = "\n".join([f"{event['start'].get('dateTime', event['start'].get('date'))}: {event['summary']}" for event in conflicting_events])
                return f"The time slot is not available. Conflicting events:\n{conflicts}"

            return (f"Please confirm the following appointment details WITH THE HUMAN:\n"
                    f"Summary: {summary}\n"
                    f"Location: {location}\n"
                    f"Description: {description}\n"
                    f"Start: {start_time}\n"
                    f"End: {end_time}\n"
                    f"Reply with 'yes' to confirm or 'no' to cancel.")

        elif action == 'confirm_book':
            return self.book_appointment(summary, location, description, start_time, end_time)

        elif action == 'list':
            return self.list_appointments(time_min, time_max, max_results)

        elif action == 'update':
            return self.update_appointment(event_id, updated_summary, updated_location, updated_description, updated_start_time, updated_end_time)

        elif action == 'delete':
            return self.delete_appointment(event_id)

        else:
            return "Invalid action specified.  PLEASE TELL THE HUMAN."

    def _arun(self):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")



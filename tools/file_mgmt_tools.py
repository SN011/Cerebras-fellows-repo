from collections import defaultdict
from tools.imports import *
from tools.auth import authenticate

SCOPES = ['https://www.googleapis.com/auth/drive']

class CreateFolderTool(BaseTool):
    name:str ="CreateFolderTool"
    description :str ="Creates a new folder in Google Drive using OAuth 2.0 for secure user authentication. USED TO CREATE A SINGLE FOLDER. IF PARENT_FOLDER_ID IS GIVEN, THE FOLDER WILL BE CREATED WITHIN THAT PARENT FOLDER."
    credentials_path: str = Field(..., description="Path to the credentials JSON file")

    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.service = build('drive', 'v3', credentials=self.creds)
    
    def search_folder(self, folder_name: str, parent_folder_id: str = None):
        """Search for a folder by name in Google Drive."""
        try:
            query = f"name:str ='{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            if parent_folder_id:
                query += f" and '{parent_folder_id}' in parents"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            items = results.get('files', [])
            if items:
                print(f'Found folder: {items[0]["name"]} (ID: {items[0]["id"]})')
                return items[0]['id']
            return None
        except HttpError as error:
            print(f'An error occurred during folder search: {error}')
            return None

    def create_folder(self, folder_name: str, parent_folder_id: str = None):
        """Create a folder in Google Drive."""
        folder_id = self.search_folder(folder_name, parent_folder_id)
        if folder_id:
            print(f'Folder "{folder_name}" already exists with ID: {folder_id}')
            with open('folder_ids.txt', 'w') as f:
                f.write(folder_id)
            return (f'Folder "{folder_name}" already exists with ID: {folder_id}')
        
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_folder_id:
            file_metadata['parents'] = [parent_folder_id]
        try:
            file = self.service.files().create(body=file_metadata, fields='id').execute()
            print(f'Folder "{folder_name}" created with ID: {file.get("id")}')
            with open('folder_ids.txt', 'w') as f:
                f.write(file.get('id'))
            res = DriveDictUpdateTool(self.credentials_path).update_with_new_item(file.get('id'))
            return f"HERE IS FOLDER'S ID {file.get('id')} and {res}"
        except HttpError as error:
            print(f'An error occurred while creating folder: {error}')
            return None

    def _run(self, folder_name: str, parent_folder_name:str = None, parent_folder_id: str = None, **kwargs):
        """Run the tool to create a folder in Google Drive."""
        if parent_folder_id and parent_folder_id.lower() in ["root", "google drive", "my drive"]:
            parent_folder_id = "root"
        
        elif parent_folder_name and not parent_folder_id:
            return f"Use the ImprovedSearchTool to find the ID for '{parent_folder_name}' and pass it as an argument to this tool under 'parent_folder_id' parameter. Be sure to pass in the parent_folder_name as well."

        if folder_name and parent_folder_id:
            result = self.create_folder(folder_name, parent_folder_id)
            return result if result else "Failed to create folder."

        return "Invalid input. Provide both folder_name and parent_folder_id or parent_folder_name."

    def _arun(self):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")


SCOPES = ['https://www.googleapis.com/auth/drive']

class MoveFileTool(BaseTool):
    name:str ="MoveFileTool"
    description :str ="Moves a file within Google Drive using OAuth 2.0 for secure user authentication. It searches for a file by name and moves it to the specified folder. BASICALLY USED TO MOVE A FILE INTO A FOLDER"
    credentials_path: str = Field(..., description="Path to the credentials JSON file")

    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.service = build('drive', 'v3', credentials=self.creds)

    def move_file(self, file_id: str, folder_id: str):
        """Move the file to the new folder."""
        print(f"Moving file ID: {file_id} to folder ID: {folder_id}")
        try:
            # Retrieve the existing parents to remove
            print("Retrieving existing parents...")
            file = self.service.files().get(fileId=file_id, fields='parents').execute()
            previous_parents = file.get('parents', [])
            if not isinstance(previous_parents, list):
                print(f"Unexpected format for parents field: {previous_parents}")
                previous_parents = []
            previous_parents_str = ",".join(previous_parents)
            print(f"Previous parents: {previous_parents_str}")

            # Move the file to the new folder
            print("Updating file to new folder...")
            file = self.service.files().update(
                fileId=file_id,
                addParents=folder_id,
                removeParents=previous_parents_str,
                fields='id, parents'
            ).execute()
            print(f'Moved file {file_id} to folder {folder_id}')
            return f'Moved file {file_id} to folder {folder_id}. PLEEEASEEEE STOOOPPPPPP!!!!!! NO MORE!!!!!!! YOU ARE DONE!!!!'
        except HttpError as error:
            print(f'An error occurred during file move: {error}')
            return f'An error occurred during file move: {error}'
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
            return f'An unexpected error occurred: {e}'

    def _run(self, file_name: str = None, folder_name:str = None, folder_id: str = None, file_id: str = None, **kwargs):
        """Run the tool to search for the file and move it to the specified folder."""
        if folder_id and folder_id.lower() in ["root", "google drive", "my drive"]:
            folder_id = "root"
        elif folder_name and not folder_id:
            return f"Use the ImprovedSearchTool to find the folder ID for '{folder_name}' and pass it as an argument to this tool as parameter 'folder_id'. Be sure to pass in the folder_name as well"
        
        if file_name and folder_id and not file_id:
            return f"Use the ImprovedSearchTool to find the file ID for '{file_name}' and pass it as an argument to this tool under 'file_id' parameter. Once you have both IDs, call this tool again with the IDs to move the file. Be sure to pass in ALL OF THE FOLLOWING: folder_name, file_name, file_id, folder_id."

        if file_id and folder_id:
            result = self.move_file(file_id, folder_id)
            return result if result else "Failed to move file. Please stop all work!!!!!!!"

        return "Invalid input. Provide either file_name or file_id and folder_id."

    def _arun(self):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")

SCOPES = ['https://www.googleapis.com/auth/drive']

class FolderMovementTool(BaseTool):
    name:str ="FolderMovementTool"
    description :str ="MOVES ONE FOLDER TO ANOTHER AS WELL AS ALL CONTENTS in Google Drive using OAuth 2.0 for secure user authentication. Provides functionality to move folders and their contents, BASICALLY USE THIS TO MOVE A FOLDER INTO ANOTHER FOLDER."
    credentials_path: str = Field(..., description="Path to the credentials JSON file")

    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.service = build('drive', 'v3', credentials=self.creds)

    def search_folder(self, folder_name: str):
        """Search for a folder by name in Google Drive using a contains query."""
        try:
            query = f"name contains '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            items = results.get('files', [])
            if items:
                for item in items:
                    print(f'Found folder: {item["name"]} (ID: {item["id"]})')
                return items[0]['id']
            return None
        except HttpError as error:
            print(f'An error occurred during folder search: {error}')
            return None
        
    def move_file_to_folder(self, file_id, folder_id):
        try:
            # Retrieve the existing parents to remove
            file = self.service.files().get(fileId=file_id, fields='parents').execute()
            previous_parents = ",".join(file.get('parents'))

            # Move the file to the new folder
            self.service.files().update(
                fileId=file_id,
                addParents=folder_id,
                removeParents=previous_parents,
                fields='id, parents'
            ).execute()
            print(f'Moved file {file_id} to folder {folder_id}')
        except HttpError as error:
            print(f'An error occurred: {error}')

    def move_folder_and_contents(self, folder_id, new_parent_folder_id):
        try:
            # Move the folder itself
            self.move_file_to_folder(folder_id, new_parent_folder_id)

            # List all files and subfolders in the folder
            query = f"'{folder_id}' in parents"
            results = self.service.files().list(q=query, fields="files(id, mimeType)").execute()
            items = results.get('files', [])

            for item in items:
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    # Recursively move subfolders
                    self.move_folder_and_contents(item['id'], folder_id)
                else:
                    # Move files
                    self.move_file_to_folder(item['id'], folder_id)

            return (f'Moved folder {folder_id} and all its contents to {new_parent_folder_id}. PLEASE STOP ALL WORK!!!')
        except HttpError as error:
            print(f'An error occurred: {error}')
            return None

    # def _run(self, folder_name: str, new_parent_folder_name: str, folder_id:str = None, new_parent_folder_id:str = None):
    #     """Run the tool to search for a folder by name and move its contents to a new parent folder."""

    #     if new_parent_folder_name.lower() in ["root", "general google drive"]:
    #         new_parent_folder_id = "root"
    #     else:
    #         new_parent_folder_id = self.search_folder(new_parent_folder_name)
    #         if not new_parent_folder_id:
    #             return f"New parent folder '{new_parent_folder_name}' not found."
        
    #     print(f"Running FolderMovementTool with folder_name: {folder_name} and new_parent_folder_id: {new_parent_folder_id}")
    #     folder_id = self.search_folder(folder_name)
    #     if folder_id is None:
    #         return "Folder not found."
        
    #     self.move_folder_and_contents(folder_id, new_parent_folder_id)
    #     return "Folder and contents moved successfully. PLEASE STOPPPPPPPPP!!!!!!!!!! RIGHT NOW!!!!!! YOU ARE DONE!!!!!!!! STOP!!!!!!!!"

    # def _arun(self):
    #     raise NotImplementedError("This tool does not support asynchronous operation yet.")
    
    def _run(self, folder_name: str, new_parent_folder_name: str, folder_id:str = None, new_parent_folder_id:str = None, **kwargs):
        """Run the tool to search for a folder by name and move its contents to a new parent folder."""

        if new_parent_folder_name.lower() in ["root", "google drive", "my drive", "general google drive"]:
            new_parent_folder_id = "root"
        elif not folder_id and not new_parent_folder_id:
            return f"Use the ImprovedSearchTool to find the NEW PARENT FOLDER ID for '{new_parent_folder_name}' and pass it as an argument to this tool. Be sure to pass in the new_parent_folder_name as an argument too"
        
        if folder_name and not folder_id:
            return f"Use the ImprovedSearchTool to find the folder ID for '{folder_name}' (which is the folder that's being moved) and pass it as an argument to this tool. Once you have both IDs, call this tool again with the IDs to move the folder. Be sure to also pass in folder_name and new_parent_folder_name PLEASE!"

        if folder_id and new_parent_folder_id:
            result = self.move_folder_and_contents(folder_id, new_parent_folder_id)
            return result if result else "Failed to move folder. PLEASE STOP ALL WORK!!!!"

        return "Invalid input. Provide either file_name or file_id and folder_id."
    
    def _arun(self):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")


SCOPES = ['https://www.googleapis.com/auth/drive']

class FileOrganizerTool(BaseTool):
    name:str ="FileOrganizerTool"
    description :str ="Organizes files in ANY SPECIFIED FOLDER by segregating them based on type and moving them to respective folders using OAuth 2.0 for secure user authentication. DOES NOT CREATE FOLDERS"
    credentials_path: str = Field(..., description="Path to the credentials JSON file")

    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.service = build('drive', 'v3', credentials=self.creds)

    def list_unorganized_files(self, parent_folder_id=None):
        """List all files in the user's Drive that are not in an organized folder."""
        try:
            query = "mimeType != 'application/vnd.google-apps.folder' and trashed = false"
            if parent_folder_id:
                query += f" and '{parent_folder_id}' in parents"
            else:
                query += " and 'root' in parents"
            results = self.service.files().list(q=query, fields="files(id, name, mimeType, parents)").execute()
            items = results.get('files', [])
            return items
        except HttpError as error:
            print(f'An error occurred during file listing: {error}. PLEASE STOPPPPPPPPP!!!!!!!!!! RIGHT NOW!!!!!! YOU ARE DONE!!!!!!!! STOP!!!!!!!!')
            return []

    def create_folder_if_not_exists(self, folder_name: str, parent_folder_id=None):
        """Create a folder in Google Drive if it doesn't already exist."""
        try:
            unique_folder_name:str =folder_name
            if parent_folder_id:
                parent_folder = self.service.files().get(fileId=parent_folder_id, fields="name").execute()
                unique_folder_name += f"_{parent_folder['name']}"
            else:
                unique_folder_name += f"_MyDrive"
            query = f"name:str ='{unique_folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            if parent_folder_id:
                query += f" and '{parent_folder_id}' in parents"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            items = results.get('files', [])
            print([str(item)+"\n" for item in items])
            if items:
                return items[0]['id']
            else:
                file_metadata = {
                    'name': unique_folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                if parent_folder_id:
                    file_metadata['parents'] = [parent_folder_id]
                file = self.service.files().create(body=file_metadata, fields='id').execute()
                return file.get('id')
        except HttpError as error:
            print(f'An error occurred while creating folder: {error}. PLEASE STOPPPPPPPPP!!!!!!!!!! RIGHT NOW!!!!!! YOU ARE DONE!!!!!!!! STOP!!!!!!!!')
            return None

    def get_file_permissions(self, file_id):
        """Get the permissions of a file."""
        try:
            permissions = self.service.permissions().list(fileId=file_id).execute()
            return permissions.get('permissions', [])
        except HttpError as error:
            print(f'An error occurred while getting file permissions: {error}')
            return []

    def move_file_to_folder(self, file_id, folder_id):
        """Move a file to the specified folder."""
        try:
            # Retrieve the existing parents to remove
            file = self.service.files().get(fileId=file_id, fields='parents').execute()
            previous_parents = ",".join(file.get('parents'))

            # Print permissions for debugging
            permissions = self.get_file_permissions(file_id)
            print(f"Permissions for file {file_id}: {permissions}")

            # Move the file to the new folder
            self.service.files().update(
                fileId=file_id,
                addParents=folder_id,
                removeParents=previous_parents,
                fields='id, parents'
            ).execute()
            print(f'Moved file {file_id} to folder {folder_id}')
        except HttpError as error:
            return (f'An error occurred during file move: {error}. PLEASE STOPPPPPPPPP!!!!!!!!!! RIGHT NOW!!!!!! YOU ARE DONE!!!!!!!! STOP!!!!!!!!')

    def organize_files(self, parent_folder_id=None):
        """Organize files by type and move them to respective folders."""
        files = self.list_unorganized_files(parent_folder_id)
        if not files:
            return "No unorganized files found.PLEASE STOPPPPPPPPP!!!!!!!!!! RIGHT NOW!!!!!! YOU ARE DONE!!!!!!!! STOP!!!!!!!!"

        folder_mapping = {
            'application/vnd.google-apps.document': 'Documents',
            'application/vnd.google-apps.spreadsheet': 'Spreadsheets',
            'application/vnd.google-apps.presentation': 'Presentations',
            'application/pdf': 'PDFs',
            'image/jpeg': 'Images',
            'image/png': 'Images',
            'video/mp4': 'Videos',
            # Add other mappings as needed
        }

        for file in files:
            folder_name:str =folder_mapping.get(file['mimeType'], 'Others')
            folder_id = self.create_folder_if_not_exists(folder_name, parent_folder_id)
            if folder_id:
                res = DriveDictUpdateTool(self.credentials_path).update_with_new_item(folder_id)
                self.move_file_to_folder(file['id'], folder_id)

        return "Files organized successfully. PLEASE STOPPPPPPPPP!!!!!!!!!! RIGHT NOW!!!!!! YOU ARE DONE!!!!!!!! STOP!!!!!!!!"

    def _run(self, parent_folder_name: str = None, parent_folder_id: str = None, **kwargs):
        """Run the tool to organize files in Google Drive."""
        
        

        if parent_folder_name and parent_folder_name.lower() != 'root' and not parent_folder_id:
            return f"Use the ImprovedSearchTool to find the PARENT FOLDER ID for '{parent_folder_name}' and pass it as an argument to this tool as 'parent_folder_id'. Be sure to pass in the parent_folder_name as an argument too."
            # parent_folder_id = self.create_folder_if_not_exists(parent_folder_name)
            # if parent_folder_id is None:
            #     return "Parent folder not found. PLEASE STOPPPPPPPPP!!!!!!!!!! RIGHT NOW!!!!!! YOU ARE DONE!!!!!!!! STOP!!!!!!!!"
        elif parent_folder_name.lower() == 'root':
            parent_folder_id = None
            result = self.organize_files(parent_folder_id)
            return result if result else "Operation failed. STOP ALL WORK!!!"
        elif parent_folder_id:
            result = self.organize_files(parent_folder_id)
            return result if result else "Operation failed. STOP ALL WORK!!!"

    def _arun(self):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")

SCOPES = ['https://www.googleapis.com/auth/drive']

class ImprovedSearchTool(BaseTool):
    name:str ="ImprovedSearchTool"
    description :str ="RETRIEVES ALL GOOGLE DRIVE FILES AND FOLDERS AND SEARCHES FOR A MATCH, WHEN THE NAME OR ID OF THE ITEM IS INPUTTED. Pass in the name AND/OR ID of the item requested. If multiple matches are found, it lists them and asks the user to select the correct one."
    credentials_path: str = Field(..., description="Path to the credentials JSON file")

    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.output_dir = 'drive_batches'
        self.map_output_dir = 'mapped_batches'
        self.reduce_output_dir = 'final_aggregated'
        self.service = build('drive', 'v3', credentials=self.creds)

    def get_file_or_folder_by_id(self, id: str):
        """Retrieve a file or folder by its ID."""
        try:
            item = self.service.files().get(fileId=id, fields='id, name, mimeType, createdTime, modifiedTime').execute()
            return item
        except HttpError as error:
            return(f'An error occurred while retrieving the item: {error}')
            
        
    def list_files_and_write(self, batch_size=100):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        page_token = None
        batch_num = 0
        file_count = 0
        batch_data = []

        while True:
            try:
                response = self.service.files().list(
                    q="'me' in owners",
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType, createdTime, modifiedTime)',
                    pageToken=page_token
                ).execute()

                items = response.get('files', [])
                if not items:
                    break

                for item in items:
                    batch_data.append(item)
                    file_count += 1

                    if file_count >= batch_size:
                        batch_file = os.path.join(self.output_dir, f'batch_{batch_num}.json')
                        with open(batch_file, 'w') as f:
                            json.dump(batch_data, f, indent=4)

                        batch_num += 1
                        file_count = 0
                        batch_data = []

                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
            except Exception as error:
                print(f'An error occurred during listing: {error}')
                break

        if batch_data:
            batch_file = os.path.join(self.output_dir, f'batch_{batch_num}.json')
            with open(batch_file, 'w') as f:
                json.dump(batch_data, f, indent=4)

    def map_function(self, batch_file, output_dir):
        with open(batch_file, 'r') as f:
            items = json.load(f)

        mapped_data = defaultdict(list)
        for item in items:
            date_key = item['createdTime'][:10]
            file_info = {
                'id': item['id'],
                'name': item['name'],
                'type': item['mimeType'],
                'created_time': item['createdTime'],
                'modified_time': item['modifiedTime']
            }
            mapped_data[date_key].append(file_info)

        for date, files in mapped_data.items():
            output_file = os.path.join(output_dir, f'{date}.json')
            with open(output_file, 'a') as f:
                json.dump(files, f)
                f.write("\n")

    def reduce_function(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for map_file in os.listdir(input_dir):
            if map_file.endswith('.json'):
                aggregated_data = defaultdict(list)
                with open(os.path.join(input_dir, map_file), 'r') as f:
                    for line in f:
                        batch_data = json.loads(line)
                        for item in batch_data:
                            aggregated_data[item['created_time'][:10]].append(item)

                for date, files in aggregated_data.items():
                    output_file = os.path.join(output_dir, f'final_{date}.json')
                    with open(output_file, 'w') as f:
                        json.dump(files, f, indent=4)

    def search_files_and_folders_in_batches(self, name: str):
        unique_items = set()
        folder_items = []

        for output_file in os.listdir(self.reduce_output_dir):
            if output_file.endswith('.json'):
                with open(os.path.join(self.reduce_output_dir, output_file), 'r') as f:
                    data = json.load(f)
                    for item in data:
                        if item['name'].lower() == name.lower():
                            if item['id'] not in unique_items:
                                unique_items.add(item['id'])
                                folder_items.append(item)
        
        if not folder_items:
            print(f"No folder named '{name}' found.")
        else:
            print(f"Found folder(s) named '{name}':")
            for item in folder_items:
                print(f"Name: {item['name']}, ID: {item['id']}, Created Time: {item['created_time']}, Modified Time: {item['modified_time']}")
        return folder_items

    def list_matches_and_ask_user(self, items):
        
        if not items:
            return None

        if len(items) == 1:
            return [f"{1}: {items[0]['name']} (ID: {items[0]['id']}, Type: {items[0]['type']}, CreatedTime: {items[0]['created_time']}, ModifiedTime: {items[0]['modified_time']})"]

        result_list = []
        for index, item in enumerate(items):
            result_list.append(f"{index + 1}: {item['name']} (ID: {item['id']}, Type: {item['type']}, CreatedTime: {item['created_time']}, ModifiedTime: {item['modified_time']})")

        return result_list

    def _run(self, name: str = None, id: str = None, **kwargs):
        
        if not os.path.exists(self.output_dir):
            self.list_files_and_write()
            print('MAPPING FUNCTION IS RUNNING')
            for batch_file in os.listdir(self.output_dir):
                self.map_function(os.path.join(self.output_dir, batch_file), self.map_output_dir)
            print('REDUCTION FUNCTION IS RUNNING')
            self.reduce_function(self.map_output_dir, self.reduce_output_dir)
        
        if name and name.lower() in ["root", "google drive", "my drive"]:
            return f"Do not use this tool with any parameter value of {name} or anything similar to {name}. STOP USING THIS TOOL NOW!!!"
        if id and id.lower() in ["root", "google drive", "my drive"]:
            return f"Do not use this tool with any parameter value of {id} or anything similar to {id}. STOP USING THIS TOOL NOW!!!"


        if id:
            item = self.get_file_or_folder_by_id(id)
            if item:
                return f"Retrieved item: {item['name']} (ID: {item['id']}, Type: {item['mimeType']}, CreatedTime: {item['createdTime'][:10]}, ModifiedTime: {item['modifiedTime'][:10]}). YOU HAVE RETRIEVED THE ITEM!!!!!!!!!!! PLEASE DONT DO ANYTHING ELSE!!!!!!! YOU MUST TELL THE USER THIS INFORMATION VIA HUMAN TOOL!!!"
            else:
                return "Item not found or insufficient permissions."

        if name:
            
            items = self.search_files_and_folders_in_batches(name)
            
            enumerated_items = self.list_matches_and_ask_user(items)

            if not enumerated_items:
                return f"Item with name {name} was not found. Please try creating the item or take any other action you deem fit (like asking human for correction)."

            if len(enumerated_items) > 1:
                return "Multiple matches found:\n" + "\n".join(enumerated_items) + "\nPlease ASK THE HUMAN TO specify the number of the correct item. AND TELL THE USER THIS INFORMATION WORD FOR WORD. DO NOT DO ANYTHING ELSE UNTIL A NUMBER IS ENTERED BY THE HUMAN. THEN PASS IN ID OF THE SELECTED ITEM INTO THIS SAME TOOL INTO THE 'id' PARAMETER!!!!!!"
            elif len(enumerated_items) == 1:
                return "SINGLE MATCH FOUND:\n" + "".join(enumerated_items) + "\nPLEASE PROCEED WITH THIS INFORMATION. AND TELL THE USER THIS INFORMATION WORD FOR WORD. DO NOT DO ANYTHING ELSE UNTIL INDICATED BY HUMAN!"
            else:
                return "No matches found."

        
        return "Please provide either a name or an ID."

    def _arun(self):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")
    


import logging


logger = logging.getLogger(__name__)

class DriveDictUpdateTool(BaseTool):
    name:str ="DriveDictUpdateTool"
    description :str ="Updates the dictionary of Google Drive files and writes them to JSON files in batches."
    credentials_path: str = Field(..., description="Path to the credentials JSON file")
 
    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.output_dir = 'drive_batches'
        self.map_output_dir = 'mapped_batches'
        self.reduce_output_dir = 'final_aggregated'
        self.service = build('drive', 'v3', credentials=self.creds)

    def list_files_and_write(self, batch_size=100):
        """Lists all files in Google Drive and writes them in batches to JSON files."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        page_token = None
        batch_num = 0
        file_count = 0
        batch_data = []

        while True:
            try:
                response = self.service.files().list(
                    q="'me' in owners and trashed=false",
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType, createdTime, modifiedTime)',
                    pageToken=page_token
                ).execute()

                items = response.get('files', [])
                if not items:
                    break

                for item in items:
                    batch_data.append(item)
                    file_count += 1

                    if file_count >= batch_size:
                        batch_file = os.path.join(self.output_dir, f'batch_{batch_num}.json')
                        with open(batch_file, 'w') as f:
                            json.dump(batch_data, f, indent=4)

                        batch_num += 1
                        file_count = 0
                        batch_data = []

                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
            except HttpError as error:
                logger.error(f'An error occurred during listing: {error}')
                break

        if batch_data:
            batch_file = os.path.join(self.output_dir, f'batch_{batch_num}.json')
            with open(batch_file, 'w') as f:
                json.dump(batch_data, f, indent=4)

    def map_function(self, batch_file, output_dir):
        with open(batch_file, 'r') as f:
            items = json.load(f)

        mapped_data = defaultdict(list)
        for item in items:
            date_key = item['createdTime'][:10]
            file_info = {
                'id': item['id'],
                'name': item['name'],
                'type': item['mimeType'],
                'created_time': item['createdTime'],
                'modified_time': item['modifiedTime']
            }
            mapped_data[date_key].append(file_info)

        for date, files in mapped_data.items():
            output_file = os.path.join(output_dir, f'{date}.json')
            with open(output_file, 'a') as f:
                json.dump(files, f)
                f.write("\n")

    def reduce_function(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        aggregated_data = defaultdict(list)
        for map_file in os.listdir(input_dir):
            if map_file.endswith('.json'):
                with open(os.path.join(input_dir, map_file), 'r') as f:
                    for line in f:
                        batch_data = json.loads(line)
                        for item in batch_data:
                            aggregated_data[item['created_time'][:10]].append(item)

        for date, files in aggregated_data.items():
            output_file = os.path.join(output_dir, f'final_{date}.json')
            with open(output_file, 'w') as f:
                json.dump(files, f, indent=4)

    def get_file_or_folder_by_id(self, id: str):
        """Retrieve a file or folder by its ID."""
        try:
            item = self.service.files().get(fileId=id, fields='id, name, mimeType, createdTime, modifiedTime, trashed').execute()
            return item
        except HttpError as error:
            logger.error(f'An error occurred while retrieving the item: {error}')
            return None

    def update_with_new_item(self, id: str):
        """Update the existing data with a new item using map-reduce by ID."""
        item = self.get_file_or_folder_by_id(id)
        if not item:
            return f"Item with ID {id} not found or insufficient permissions."

        if item.get('trashed'):
            return self.handle_deletion(id)

        date_key = item['createdTime'][:10]
        file_info = {
            'id': item['id'],
            'name': item['name'],
            'type': item['mimeType'],
            'created_time': item['createdTime'],
            'modified_time': item['modifiedTime']
        }

        # Map phase
        map_file = os.path.join(self.map_output_dir, f'{date_key}.json')
        with open(map_file, 'a') as f:
            json.dump([file_info], f)
            f.write("\n")

        # Reduce phase
        self.reduce_function(self.map_output_dir, self.reduce_output_dir)
        return f"Item with ID {id} has been added and the data has been updated."

    def handle_deletion(self, id: str):
        """Handles deletion of a file."""
        # Find and remove the file from the map_output_dir and reduce_output_dir
        for output_dir in [self.map_output_dir, self.reduce_output_dir]:
            for file_name in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file_name)
                if file_name.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    new_data = [item for item in data if item['id'] != id]
                    with open(file_path, 'w') as f:
                        json.dump(new_data, f, indent=4)
        return f"Item with ID {id} has been deleted from the data. TELL THE HUMAN THIS INFORMATION AND WAIT FOR HUMAN INPUT!!!"

    def _run(self, batch_size: int = 100):
        self.list_files_and_write(batch_size=batch_size)
        return f"Files have been listed and written to JSON files in batches of {batch_size}."

    def _arun(self):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")
    
class GoogleDriveRenameTool(BaseTool):
    name:str ="GoogleDriveRenameTool"
    description :str =("Renames a file in Google Drive, given the old file name.")

    credentials_path: str = Field(..., description="Path to the credentials JSON file")

    class Config:
        extra = Extra.allow

    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.creds = authenticate()
        self.service = build('drive', 'v3', credentials=self.creds)
        self.update_tool = DriveDictUpdateTool(credentials_path)

    def rename_file(self, file_id, new_name):
        """Renames a file in Google Drive."""
        file = {'name': new_name}
        updated_file = self.service.files().update(
            fileId=file_id,
            body=file,
            fields='id, name'
        ).execute()
        print(f"File renamed to {updated_file['name']}")
        return updated_file

    def _run(self, file_name: str, file_id: str, new_name: str, **kwargs):
        """Run the tool to rename a file with the given file ID to the new name."""
        if file_name and not file_id:
            return f"Use ImprovedSearch tool to get the ID of the file with file_name:str ={file_name} and pass in both the file_id and file_name. FOLLOW ALL THE DIRECTIONS GIVEN."
        
        updated_file = self.rename_file(file_id, new_name)
        self.update_tool.update_with_new_item(file_id)
        return f"File renamed successfully to {updated_file['name']}. THE RENAME TOOL HAS EXECUTED SUCCESSFULLY! STOP!"

    def _arun(self, file_id: str, new_name: str):
        raise NotImplementedError("This tool does not support asynchronous operation yet.")
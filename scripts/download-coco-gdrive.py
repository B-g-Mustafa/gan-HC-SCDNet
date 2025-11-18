import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# --- Scopes ---
# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# --- Configuration ---
# The ID of the folder in your Google Drive containing the file chunks.
# To get the folder ID, navigate to the folder in Google Drive and copy the ID from the URL:
# https://drive.google.com/drive/folders/FOLDER_ID
FOLDER_ID = ""  # Replace with your actual folder ID

# The local directory on your server to save the downloaded chunks.
DOWNLOAD_DIR = "downloaded_chunks" 

def authenticate():
    """Handles user authentication and returns a service object."""
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # This will run ONCE. It will print a URL.
            # You must copy the URL, open it in your local browser, authorize,
            # and paste the resulting code back into the terminal.
            flow = InstalledAppFlow.from_client_secrets_file('creds.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    try:
        service = build('drive', 'v3', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred building the service: {error}')
        return None

def download_chunks():
    """Finds the folder and downloads all files within it."""
    service = authenticate()
    if not service:
        return

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        print(f"Created local directory: '{DOWNLOAD_DIR}'")

    try:
        # 1. Verify the folder exists and get its name
        folder_id = FOLDER_ID
        try:
            folder_info = service.files().get(fileId=folder_id, fields='name').execute()
            folder_name = folder_info.get('name', 'Unknown')
            print(f"Found folder '{folder_name}' with ID: {folder_id}")
        except HttpError as error:
            print(f"Error: Folder with ID '{folder_id}' not found or access denied.")
            print(f"Details: {error}")
            return

        # 2. List all files inside that folder
        query = f"'{folder_id}' in parents"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])

        if not files:
            print(f"No files found in folder '{FOLDER_NAME}'.")
            return

        print(f"\nStarting download of {len(files)} file(s)...")
        # 3. Download each file
        for item in files:
            file_id = item.get('id')
            file_name = item.get('name')
            local_filepath = os.path.join(DOWNLOAD_DIR, file_name)
            
            print(f"Downloading '{file_name}' to '{local_filepath}'...")
            request = service.files().get_media(fileId=file_id)
            fh = io.FileIO(local_filepath, 'wb')
            
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"  -> Download {int(status.progress() * 100)}%.")
        
        print("\n✅ All files downloaded successfully.")

    except HttpError as error:
        print(f'An error occurred: {error}')

if __name__ == '__main__':
    download_chunks()
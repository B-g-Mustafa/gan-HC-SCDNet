import os
import sys
import traceback
import argparse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# If modifying scopes, delete token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']

# Use the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CREDS_JSON_PATH = os.path.join(SCRIPT_DIR, 'creds.json')
TOKEN_JSON_PATH = os.path.join(SCRIPT_DIR, 'token.json')

def authenticate():
    """Handle Google Drive authentication and return service object."""
    creds = None
    
    # Load existing credentials if available
    if os.path.exists(TOKEN_JSON_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_JSON_PATH, SCOPES)
        except Exception as e:
            print(f"Error loading existing token: {e}")
    
    # If no valid credentials available, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                creds = None
        
        if not creds:
            if not os.path.exists(CREDS_JSON_PATH):
                print(f"Error: creds.json not found at {CREDS_JSON_PATH}")
                return None
                
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDS_JSON_PATH, SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                print(f"Error during authentication: {e}")
                return None
            
            # Save credentials for future use
            try:
                with open(TOKEN_JSON_PATH, 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                print(f"Warning: Could not save token: {e}")
    
    try:
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"Error building service: {e}")
        return None

def upload_file(filepath: str, drive_folder_id: str, new_filename: str = None):
    """
    Upload a single file to Google Drive.
    
    Args:
        filepath: Path to the local file to upload
        drive_folder_id: The Google Drive folder ID to upload to
        new_filename: Optional new name for the file in Drive (default: use original filename)
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False

    service = authenticate()
    if not service:
        print("Error: Failed to authenticate with Google Drive")
        return False

    try:
        # Verify the drive folder exists
        try:
            folder = service.files().get(fileId=drive_folder_id, fields='name').execute()
            print(f"Target folder: '{folder['name']}' (ID: {drive_folder_id})")
        except HttpError:
            print(f"Error: Could not access folder with ID '{drive_folder_id}'")
            return False

        # Prepare file metadata
        filename = new_filename or os.path.basename(filepath)
        file_metadata = {
            'name': filename,
            'parents': [drive_folder_id]
        }

        # Check if file already exists in the folder
        query = f"name='{filename}' and '{drive_folder_id}' in parents and trashed=false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        if response.get('files'):
            print(f"Warning: File '{filename}' already exists in the destination folder")
            return False

        # Create file upload
        print(f"Uploading '{filename}'...")
        media = MediaFileUpload(
            filepath,
            resumable=True,
            chunksize=1024*1024  # 1MB chunks
        )

        # Execute the upload with progress tracking
        request = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        )
        
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Progress: {int(status.progress() * 100)}%")

        print(f"✓ Upload complete! File ID: {response.get('id')}")
        return True

    except Exception as e:
        print(f"Error during upload: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload a single file to Google Drive")
    parser.add_argument('--file', '-f', required=True, default="/home/msai/birul001/gan-project/gan-HC-SCDNet/dataset_captioned.tar.gz",
                       help="Local file to upload")
    parser.add_argument('--drive-id', '-d', required=True, default="11euKusj0vU-GiyrG8WY3Ih820Fsvc_2d",
                       help="Drive folder ID to upload into")
    parser.add_argument('--name', '-n',
                       help="Optional: New filename to use in Drive")

    args = parser.parse_args()
    
    success = upload_file(args.file, args.drive_id, args.name)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nUpload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
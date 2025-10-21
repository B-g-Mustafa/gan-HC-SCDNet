import os
import io
import argparse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# --- Scopes ---
# This scope allows for read, write, and create access.
# If you change this, delete token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
    """Handles user authentication and returns a Google Drive service object."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # This will open a local browser for authentication
            flow = InstalledAppFlow.from_client_secrets_file('creds.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    try:
        service = build('drive', 'v3', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred building the service: {error}')
        return None

def upload_checkpoints(local_folder, drive_folder_id):
    """
    Uploads all files from a local folder and its subfolders to Google Drive,
    skipping files that already exist in Google Drive with the same name.
    
    Args:
        local_folder: Path to the local folder to upload
        drive_folder_id: The Google Drive folder ID (not name)
    """
    service = authenticate()
    if not service:
        print("Failed to authenticate with Google Drive.")
        return

    try:
        # 1. Verify the Drive folder exists
        try:
            folder_info = service.files().get(fileId=drive_folder_id, fields='name').execute()
            folder_name = folder_info.get('name', 'Unknown')
            print(f"✓ Target Drive folder: '{folder_name}' (ID: {drive_folder_id})")
        except HttpError as error:
            print(f"Error: Could not access folder with ID '{drive_folder_id}'")
            print(f"Details: {error}")
            return

        # 2. Get a list of all file names already in the Drive folder (recursively)
        def get_all_files_in_folder(folder_id, path=""):
            """Recursively get all files in a Drive folder and its subfolders."""
            all_files = {}
            try:
                query = f"'{folder_id}' in parents and trashed=false"
                response = service.files().list(q=query, spaces='drive', 
                                               fields='files(id, name, mimeType)', pageSize=1000).execute()
                items = response.get('files', [])
                
                for item in items:
                    file_path = f"{path}/{item['name']}" if path else item['name']
                    
                    # If it's a folder, recurse
                    if item['mimeType'] == 'application/vnd.google-apps.folder':
                        subfolder_files = get_all_files_in_folder(item['id'], file_path)
                        all_files.update(subfolder_files)
                    else:
                        # It's a file, add to dict
                        all_files[item['name']] = file_path
            except Exception as e:
                print(f"Warning: Could not list files in folder: {e}")
            
            return all_files

        existing_drive_files = get_all_files_in_folder(drive_folder_id)
        print(f"Found {len(existing_drive_files)} files already in the Drive folder (including subfolders).\n")

        # 3. Recursively collect all local files from source and subfolders
        local_files_to_upload = []
        for root, dirs, files in os.walk(local_folder):
            for filename in files:
                local_filepath = os.path.join(root, filename)
                local_files_to_upload.append((filename, local_filepath))

        print(f"Found {len(local_files_to_upload)} files to process in local folder (including subfolders).\n")

        # 4. Upload files, skipping duplicates
        uploaded_count = 0
        skipped_count = 0
        failed_count = 0

        for filename, local_filepath in local_files_to_upload:
            if filename in existing_drive_files:
                print(f"⊘ Skipping '{filename}' -> Already exists in Google Drive")
                skipped_count += 1
                continue
            
            try:
                print(f"↑ Uploading '{filename}'...")
                
                file_metadata = {'name': filename, 'parents': [drive_folder_id]}
                # resumable=True is essential for large files
                media = MediaFileUpload(local_filepath, resumable=True)
                
                request = service.files().create(body=file_metadata, media_body=media, fields='id')
                
                # Execute the upload with progress display
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        print(f"  └─ Progress: {int(status.progress() * 100)}%")
                
                print(f"  ✓ Successfully uploaded '{filename}' (ID: {response.get('id')})")
                uploaded_count += 1
            except Exception as e:
                print(f"  ✗ Failed to upload '{filename}': {e}")
                failed_count += 1

        # 5. Print summary
        print("\n" + "="*60)
        print("Upload Summary:")
        print(f"  ✓ Successfully uploaded: {uploaded_count}")
        print(f"  ⊘ Skipped (already exist): {skipped_count}")
        print(f"  ✗ Failed: {failed_count}")
        print("="*60)

    except HttpError as error:
        print(f'An HTTP error occurred: {error}')
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    # """Main function to parse arguments."""
    # parser = argparse.ArgumentParser(description="Upload model checkpoints to Google Drive, skipping existing files.")
    # parser.add_argument('source_directory', default='../../model/', help="The local directory containing the checkpoint files.")
    # parser.add_argument('drive_folder_id',default='mpctXL24tqLTw0vad2ifjlH40ofI', help="The Google Drive folder ID (not name). To get it, navigate to the folder and copy the ID from the URL: https://drive.google.com/drive/folders/FOLDER_ID")
    
    # args = parser.parse_args()

    # if not os.path.isdir(args.source_directory):
    #     print(f"Error: Local source directory '{args.source_directory}' does not exist.")
    #     return
        
    upload_checkpoints('../../model/', '1YgR-mpctXL24tqLTw0vad2ifjlH40ofI')

if __name__ == '__main__':
    main()


import os
import sys
import io
import argparse
import traceback
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

# --- Absolute paths for credentials ---
# Use the directory where this script is located as the base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CREDS_JSON_PATH = os.path.join(SCRIPT_DIR, 'creds.json')
TOKEN_JSON_PATH = os.path.join(SCRIPT_DIR, 'token.json')

print(f"[Init] Script directory: {SCRIPT_DIR}", flush=True)
print(f"[Init] Looking for creds.json at: {CREDS_JSON_PATH}", flush=True)
print(f"[Init] Token will be saved at: {TOKEN_JSON_PATH}", flush=True)

def authenticate():
    """Handles user authentication and returns a Google Drive service object."""
    creds = None
    print("[Auth] Starting authentication process...", flush=True)
    
    # Check for existing token
    if os.path.exists(TOKEN_JSON_PATH):
        print(f"[Auth] Found token.json at {TOKEN_JSON_PATH}, attempting to load credentials...", flush=True)
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_JSON_PATH, SCOPES)
            print("[Auth] Successfully loaded credentials from token.json", flush=True)
        except Exception as e:
            print(f"[Auth] Error loading token.json: {e}", flush=True)
            creds = None
    else:
        print(f"[Auth] No token.json found at {TOKEN_JSON_PATH}, will need new authentication", flush=True)
    
    # Check if credentials need refresh or new auth
    if not creds or not creds.valid:
        print("[Auth] Credentials missing or invalid, attempting to refresh or authenticate...", flush=True)
        if creds and creds.expired and creds.refresh_token:
            print("[Auth] Refreshing expired credentials...", flush=True)
            try:
                creds.refresh(Request())
                print("[Auth] Credentials refreshed successfully", flush=True)
            except Exception as e:
                print(f"[Auth] Error refreshing credentials: {e}", flush=True)
                creds = None
        else:
            # New authentication required
            print("[Auth] New authentication required. Checking for creds.json...", flush=True)
            if not os.path.exists(CREDS_JSON_PATH):
                print(f"[Auth] ERROR: creds.json not found at {CREDS_JSON_PATH}!", flush=True)
                print(f"[Auth] Current working directory: {os.getcwd()}", flush=True)
                print(f"[Auth] Script directory: {SCRIPT_DIR}", flush=True)
                print("[Auth] Please ensure creds.json exists in the script directory.", flush=True)
                return None
            
            print(f"[Auth] creds.json found at {CREDS_JSON_PATH}, initiating OAuth flow...", flush=True)
            try:
                # This will open a local browser for authentication
                flow = InstalledAppFlow.from_client_secrets_file(CREDS_JSON_PATH, SCOPES)
                print("[Auth] OAuth flow created. Opening browser for authentication...", flush=True)
                creds = flow.run_local_server(port=0)
                print("[Auth] Authentication successful!", flush=True)
            except Exception as e:
                print(f"[Auth] ERROR during OAuth flow: {e}", flush=True)
                print("[Auth] This may happen if running on a headless system without browser access.", flush=True)
                traceback.print_exc()
                return None
        
        # Save the token for future use
        if creds:
            try:
                print(f"[Auth] Saving credentials to {TOKEN_JSON_PATH}...", flush=True)
                with open(TOKEN_JSON_PATH, 'w') as token:
                    token.write(creds.to_json())
                print("[Auth] Credentials saved successfully", flush=True)
            except Exception as e:
                print(f"[Auth] Warning: Could not save token.json: {e}", flush=True)
    
    # Build the service
    try:
        print("[Auth] Building Google Drive service...", flush=True)
        service = build('drive', 'v3', credentials=creds)
        print("[Auth] Google Drive service built successfully!", flush=True)
        return service
    except HttpError as error:
        print(f'[Auth] HTTP Error building service: {error}', flush=True)
        return None
    except Exception as error:
        print(f'[Auth] Unexpected error building service: {error}', flush=True)
        traceback.print_exc()
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
        print("ERROR: Failed to authenticate with Google Drive. Check logs above for details.", flush=True)
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
    """Main function to parse arguments and call uploader.

    Usage examples:
      python upload_datapoints.py --local ./checkpoints --drive-id 1YgR-... 
      python upload_datapoints.py --local ../../model/ --drive-id 1YgR-...
    """
    print("[Main] Starting upload script...", flush=True)
    print(f"[Main] Current working directory: {os.getcwd()}", flush=True)
    
    parser = argparse.ArgumentParser(description="Upload a local folder (recursively) to a Google Drive folder."
                                     )
    parser.add_argument('--local', '-l', dest='local_folder', default='../../model/',
                        help="Local directory to upload (default: ../../model/)")
    parser.add_argument('--drive-id', '-d', dest='drive_folder_id', required=True,
                        help="Drive folder ID to upload into (required). e.g., 1YgR-...")

    args = parser.parse_args()
    print(f"[Main] Parsed arguments: local_folder={args.local_folder}, drive_folder_id={args.drive_folder_id}", flush=True)

    if not os.path.isdir(args.local_folder):
        print(f"ERROR: Local source directory '{args.local_folder}' does not exist.", flush=True)
        print(f"[Main] Current working directory: {os.getcwd()}", flush=True)
        print(f"[Main] Contents of current directory:", flush=True)
        try:
            for item in os.listdir('.'):
                print(f"  - {item}", flush=True)
        except Exception as e:
            print(f"  Could not list directory: {e}", flush=True)
        return

    print(f"[Main] Local folder '{args.local_folder}' exists. Proceeding with upload...", flush=True)
    upload_checkpoints(args.local_folder, args.drive_folder_id)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Unhandled exception in main: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

# uploading wiki train
#uploading wiki test
#upload coco train
#upload coco test

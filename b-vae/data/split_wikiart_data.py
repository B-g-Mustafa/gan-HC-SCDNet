import os
import shutil
import random
from pathlib import Path

# Configuration
WIKIART_DIR = "./wikiart"
OUTPUT_DIR = "./wikiart_split"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
TRAIN_RATIO = 0.85
TEST_RATIO = 0.15

# Create output directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def collect_all_files(root_dir):
    """
    Recursively collect all files from all subfolders.
    Returns a list of file paths.
    """
    all_files = []
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} does not exist!")
        return all_files
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    return all_files

def split_and_move_files(files, train_dir, test_dir, train_ratio):
    """
    Split files into train and test sets and move them to respective directories.
    """
    # Shuffle the files randomly
    random.shuffle(files)
    
    # Calculate split index
    split_index = int(len(files) * train_ratio)
    
    train_files = files[:split_index]
    test_files = files[split_index:]
    
    print(f"\nTotal files found: {len(files)}")
    print(f"Train files: {len(train_files)} ({train_ratio * 100}%)")
    print(f"Test files: {len(test_files)} ({(1 - train_ratio) * 100}%)")
    
    # Move train files
    print("\nMoving train files...")
    for i, file_path in enumerate(train_files, 1):
        if i % 100 == 0:
            print(f"  Moved {i}/{len(train_files)} files...")
        
        filename = os.path.basename(file_path)
        destination = os.path.join(train_dir, filename)
        
        # Handle duplicate filenames by adding a counter
        if os.path.exists(destination):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join(train_dir, f"{name}_{counter}{ext}")):
                counter += 1
            destination = os.path.join(train_dir, f"{name}_{counter}{ext}")
        
        shutil.move(file_path, destination)
    
    print(f"  Completed! Moved {len(train_files)} files to {train_dir}")
    
    # Move test files
    print("\nMoving test files...")
    for i, file_path in enumerate(test_files, 1):
        if i % 100 == 0:
            print(f"  Moved {i}/{len(test_files)} files...")
        
        filename = os.path.basename(file_path)
        destination = os.path.join(test_dir, filename)
        
        # Handle duplicate filenames by adding a counter
        if os.path.exists(destination):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join(test_dir, f"{name}_{counter}{ext}")):
                counter += 1
            destination = os.path.join(test_dir, f"{name}_{counter}{ext}")
        
        shutil.move(file_path, destination)
    
    print(f"  Completed! Moved {len(test_files)} files to {test_dir}")

def main():
    """Main function to orchestrate the file splitting process."""
    print("=" * 60)
    print("WikiArt Dataset Splitter")
    print("=" * 60)
    print(f"Source directory: {WIKIART_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train/Test ratio: {TRAIN_RATIO * 100}% / {TEST_RATIO * 100}%")
    
    # Collect all files
    print("\nCollecting files from all subfolders...")
    all_files = collect_all_files(WIKIART_DIR)
    
    if not all_files:
        print("No files found in the wikiart directory!")
        return
    
    # Split and move files
    split_and_move_files(all_files, TRAIN_DIR, TEST_DIR, TRAIN_RATIO)
    
    print("\n" + "=" * 60)
    print("Process completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

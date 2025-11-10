"""
Dataset Reorganization Script

This script reorganizes an image triplet dataset into captioned and uncaptioned subsets.
- Dataset contains: content images, style images, and synthetic images
- Two CSVs: one with 60000 rows (paths only), one with 52990 rows (paths + captions)
- Output: dataset/ folder with captioned_data/ and uncaptioned_data/ subfolders
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class DatasetReorganizer:
    def __init__(self, base_path, csv_all_data, csv_captioned_data, output_dataset_dir):
        """
        Initialize the dataset reorganizer.
        
        Args:
            base_path: Base path for resolving relative image paths in CSVs
            csv_all_data: Path to CSV with all 60000 image triplets (no captions)
            csv_captioned_data: Path to CSV with 52990 captioned image triplets
            output_dataset_dir: Output directory for reorganized dataset
        """
        self.base_path = Path(base_path)
        self.csv_all_data = csv_all_data
        self.csv_captioned_data = csv_captioned_data
        self.output_dataset_dir = Path(output_dataset_dir)
        
        # Define output subdirectories
        self.captioned_dir = self.output_dataset_dir / "captioned_data"
        self.uncaptioned_dir = self.output_dataset_dir / "uncaptioned_data"
        
        # Define image subdirectories
        self.captioned_content_dir = self.captioned_dir / "content_images"
        self.captioned_style_dir = self.captioned_dir / "style_images"
        self.captioned_synthetic_dir = self.captioned_dir / "style_transferred_images"
        
        self.uncaptioned_content_dir = self.uncaptioned_dir / "content_images"
        self.uncaptioned_style_dir = self.uncaptioned_dir / "style_images"
        self.uncaptioned_synthetic_dir = self.uncaptioned_dir / "style_transferred_images"
        
        print("=" * 80)
        print("Dataset Reorganization Configuration")
        print("=" * 80)
        print(f"Base Path: {self.base_path}")
        print(f"All Data CSV: {self.csv_all_data}")
        print(f"Captioned Data CSV: {self.csv_captioned_data}")
        print(f"Output Directory: {self.output_dataset_dir}")
        print("=" * 80)
        
    def create_directory_structure(self):
        """Create the output directory structure."""
        print("\n[Step 1/5] Creating directory structure...")
        
        directories = [
            self.captioned_content_dir,
            self.captioned_style_dir,
            self.captioned_synthetic_dir,
            self.uncaptioned_content_dir,
            self.uncaptioned_style_dir,
            self.uncaptioned_synthetic_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {directory}")
        
        print("  ✓ Directory structure created successfully!")
    
    def load_csv_files(self):
        """Load the CSV files."""
        print("\n[Step 2/5] Loading CSV files...")
        
        try:
            self.df_all = pd.read_csv(self.csv_all_data)
            print(f"  ✓ Loaded all data CSV: {len(self.df_all)} rows")
            print(f"    Columns: {list(self.df_all.columns)}")
            
            self.df_captioned = pd.read_csv(self.csv_captioned_data)
            print(f"  ✓ Loaded captioned data CSV: {len(self.df_captioned)} rows")
            print(f"    Columns: {list(self.df_captioned.columns)}")
            
            # Identify uncaptioned data
            # Assuming there's a unique identifier column (adjust as needed)
            # This will depend on your CSV structure
            self.identify_uncaptioned_data()
            
        except Exception as e:
            print(f"  ✗ Error loading CSV files: {e}")
            raise
    
    def identify_uncaptioned_data(self):
        """Identify which rows in all_data are not in captioned_data."""
        print("\n[Step 3/5] Identifying captioned vs uncaptioned data...")
        
        # Assuming the CSVs have columns like 'content_path', 'style_path', 'synthetic_path'
        # Adjust these column names based on your actual CSV structure
        
        # Create a unique identifier for each row (you may need to adjust this)
        # For example, using the synthetic image path as unique identifier
        if 'synthetic_path' in self.df_all.columns and 'synthetic_path' in self.df_captioned.columns:
            id_column = 'synthetic_path'
        elif 'Image Path' in self.df_all.columns and 'Image Path' in self.df_captioned.columns:
            id_column = 'Image Path'
        else:
            # Use the first column as identifier
            id_column = self.df_all.columns[0]
            print(f"  ⚠ Warning: Using '{id_column}' as identifier column")
        
        captioned_ids = set(self.df_captioned[id_column])
        
        self.df_all['is_captioned'] = self.df_all[id_column].isin(captioned_ids)
        
        captioned_count = self.df_all['is_captioned'].sum()
        uncaptioned_count = len(self.df_all) - captioned_count
        
        print(f"  ✓ Captioned images: {captioned_count}")
        print(f"  ✓ Uncaptioned images: {uncaptioned_count}")
        
        if captioned_count != len(self.df_captioned):
            print(f"  ⚠ Warning: Mismatch in captioned count!")
    
    def copy_images(self, df, is_captioned, content_col, style_col, synthetic_col):
        """
        Copy images from source to destination.
        
        Args:
            df: DataFrame containing image paths
            is_captioned: Boolean indicating if data is captioned
            content_col: Column name for content image paths
            style_col: Column name for style image paths
            synthetic_col: Column name for synthetic image paths
        """
        if is_captioned:
            content_dest = self.captioned_content_dir
            style_dest = self.captioned_style_dir
            synthetic_dest = self.captioned_synthetic_dir
            label = "captioned"
        else:
            content_dest = self.uncaptioned_content_dir
            style_dest = self.uncaptioned_style_dir
            synthetic_dest = self.uncaptioned_synthetic_dir
            label = "uncaptioned"
        
        print(f"\n  Copying {label} images...")
        
        new_content_paths = []
        new_style_paths = []
        new_synthetic_paths = []
        
        failed_copies = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {label} images"):
            try:
                # Content image
                content_src = self.base_path / row[content_col]
                content_filename = content_src.name
                content_dst = content_dest / content_filename
                
                # Handle duplicate filenames by adding index
                if content_dst.exists():
                    stem = content_src.stem
                    ext = content_src.suffix
                    content_dst = content_dest / f"{stem}_{idx}{ext}"
                
                shutil.copy2(content_src, content_dst)
                new_content_paths.append(str(content_dst.relative_to(self.output_dataset_dir)))
                
                # Style image
                style_src = self.base_path / row[style_col]
                style_filename = style_src.name
                style_dst = style_dest / style_filename
                
                if style_dst.exists():
                    stem = style_src.stem
                    ext = style_src.suffix
                    style_dst = style_dest / f"{stem}_{idx}{ext}"
                
                shutil.copy2(style_src, style_dst)
                new_style_paths.append(str(style_dst.relative_to(self.output_dataset_dir)))
                
                # Synthetic image
                synthetic_src = self.base_path / row[synthetic_col]
                synthetic_filename = synthetic_src.name
                synthetic_dst = synthetic_dest / synthetic_filename
                
                if synthetic_dst.exists():
                    stem = synthetic_src.stem
                    ext = synthetic_src.suffix
                    synthetic_dst = synthetic_dest / f"{stem}_{idx}{ext}"
                
                shutil.copy2(synthetic_src, synthetic_dst)
                new_synthetic_paths.append(str(synthetic_dst.relative_to(self.output_dataset_dir)))
                
            except Exception as e:
                print(f"\n  ✗ Error copying images for row {idx}: {e}")
                failed_copies.append(idx)
                new_content_paths.append("")
                new_style_paths.append("")
                new_synthetic_paths.append("")
        
        if failed_copies:
            print(f"  ⚠ Warning: {len(failed_copies)} image sets failed to copy")
            print(f"    Failed indices: {failed_copies[:10]}..." if len(failed_copies) > 10 else f"    Failed indices: {failed_copies}")
        else:
            print(f"  ✓ All {label} images copied successfully!")
        
        return new_content_paths, new_style_paths, new_synthetic_paths, failed_copies
    
    def process_and_save_csvs(self, content_col, style_col, synthetic_col):
        """
        Process data and save updated CSVs.
        
        Args:
            content_col: Column name for content image paths
            style_col: Column name for style image paths
            synthetic_col: Column name for synthetic image paths
        """
        print("\n[Step 4/5] Processing and copying images...")
        
        # Process captioned data
        df_captioned_filtered = self.df_all[self.df_all['is_captioned']].copy()
        
        # Merge with caption data to get the captions
        # Adjust the merge key based on your data
        if 'synthetic_path' in df_captioned_filtered.columns and 'synthetic_path' in self.df_captioned.columns:
            merge_key = 'synthetic_path'
        elif 'Image Path' in df_captioned_filtered.columns and 'Image Path' in self.df_captioned.columns:
            merge_key = 'Image Path'
        else:
            merge_key = df_captioned_filtered.columns[0]
        
        df_captioned_merged = df_captioned_filtered.merge(
            self.df_captioned, 
            on=merge_key, 
            how='left',
            suffixes=('', '_caption')
        )
        
        # Copy captioned images
        new_content_paths_cap, new_style_paths_cap, new_synthetic_paths_cap, failed_cap = \
            self.copy_images(df_captioned_merged, True, content_col, style_col, synthetic_col)
        
        # Update paths in captioned DataFrame
        df_captioned_merged['content_path_new'] = new_content_paths_cap
        df_captioned_merged['style_path_new'] = new_style_paths_cap
        df_captioned_merged['synthetic_path_new'] = new_synthetic_paths_cap
        
        # Save captioned CSV
        captioned_csv_path = self.captioned_dir / "captioned_metadata.csv"
        df_captioned_merged.to_csv(captioned_csv_path, index=False)
        print(f"\n  ✓ Saved captioned metadata: {captioned_csv_path}")
        
        # Process uncaptioned data
        df_uncaptioned = self.df_all[~self.df_all['is_captioned']].copy()
        
        if len(df_uncaptioned) > 0:
            # Copy uncaptioned images
            new_content_paths_uncap, new_style_paths_uncap, new_synthetic_paths_uncap, failed_uncap = \
                self.copy_images(df_uncaptioned, False, content_col, style_col, synthetic_col)
            
            # Update paths in uncaptioned DataFrame
            df_uncaptioned['content_path_new'] = new_content_paths_uncap
            df_uncaptioned['style_path_new'] = new_style_paths_uncap
            df_uncaptioned['synthetic_path_new'] = new_synthetic_paths_uncap
            
            # Save uncaptioned CSV
            uncaptioned_csv_path = self.uncaptioned_dir / "uncaptioned_metadata.csv"
            df_uncaptioned.to_csv(uncaptioned_csv_path, index=False)
            print(f"  ✓ Saved uncaptioned metadata: {uncaptioned_csv_path}")
            
            # Also save uncaptioned CSV in captioned folder (as per requirement)
            uncaptioned_ref_path = self.captioned_dir / "uncaptioned_reference.csv"
            df_uncaptioned[['content_path_new', 'style_path_new', 'synthetic_path_new']].to_csv(
                uncaptioned_ref_path, index=False
            )
            print(f"  ✓ Saved uncaptioned reference in captioned folder: {uncaptioned_ref_path}")
        else:
            print("  ℹ No uncaptioned data to process")
    
    def generate_summary(self):
        """Generate a summary of the reorganization."""
        print("\n[Step 5/5] Generating summary...")
        
        summary = []
        summary.append("=" * 80)
        summary.append("Dataset Reorganization Summary")
        summary.append("=" * 80)
        
        # Count files in each directory
        for dir_path, label in [
            (self.captioned_content_dir, "Captioned Content Images"),
            (self.captioned_style_dir, "Captioned Style Images"),
            (self.captioned_synthetic_dir, "Captioned Style-Transferred Images"),
            (self.uncaptioned_content_dir, "Uncaptioned Content Images"),
            (self.uncaptioned_style_dir, "Uncaptioned Style Images"),
            (self.uncaptioned_synthetic_dir, "Uncaptioned Style-Transferred Images"),
        ]:
            count = len(list(dir_path.glob("*"))) if dir_path.exists() else 0
            summary.append(f"{label}: {count}")
        
        # CSV files
        summary.append("\nCSV Files:")
        captioned_csv = self.captioned_dir / "captioned_metadata.csv"
        if captioned_csv.exists():
            df = pd.read_csv(captioned_csv)
            summary.append(f"  Captioned Metadata: {len(df)} rows")
        
        uncaptioned_csv = self.uncaptioned_dir / "uncaptioned_metadata.csv"
        if uncaptioned_csv.exists():
            df = pd.read_csv(uncaptioned_csv)
            summary.append(f"  Uncaptioned Metadata: {len(df)} rows")
        
        uncaptioned_ref = self.captioned_dir / "uncaptioned_reference.csv"
        if uncaptioned_ref.exists():
            df = pd.read_csv(uncaptioned_ref)
            summary.append(f"  Uncaptioned Reference (in captioned folder): {len(df)} rows")
        
        summary.append("=" * 80)
        
        summary_text = "\n".join(summary)
        print(summary_text)
        
        # Save summary to file
        summary_file = self.output_dataset_dir / "reorganization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        print(f"\n✓ Summary saved to: {summary_file}")
    
    def reorganize(self, content_col='content_path', style_col='style_path', synthetic_col='synthetic_path'):
        """
        Main method to reorganize the dataset.
        
        Args:
            content_col: Column name for content image paths (default: 'content_path')
            style_col: Column name for style image paths (default: 'style_path')
            synthetic_col: Column name for synthetic image paths (default: 'synthetic_path')
        """
        try:
            self.create_directory_structure()
            self.load_csv_files()
            self.process_and_save_csvs(content_col, style_col, synthetic_col)
            self.generate_summary()
            
            print("\n" + "=" * 80)
            print("✓ Dataset reorganization completed successfully!")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n✗ Error during reorganization: {e}")
            raise


def main():
    """
    Main function to run the dataset reorganization.
    
    IMPORTANT: Update these paths according to your dataset location!
    """
    
    # ========== CONFIGURATION - UPDATE THESE PATHS ==========
    
    # Base path for resolving relative image paths in the CSVs
    BASE_PATH = "/home/msai/birul001/BIRUL001"
    
    # Path to CSV with all 60000 image triplets (without captions)
    CSV_ALL_DATA = "/home/msai/birul001/BIRUL001/data/synthetic_dataset/csv/metadata_copy.csv"
    
    # Path to CSV with 52990 captioned image triplets
    CSV_CAPTIONED_DATA = "/home/msai/birul001/BIRUL001/captioned_meta_data_52990.csv"

    # Output directory for reorganized dataset
    OUTPUT_DATASET_DIR = "/home/msai/birul001/BIRUL001/dataset"
    
    # Column names in your CSVs (update if different)
    CONTENT_COLUMN = "content_path"      # Column name for content image paths
    STYLE_COLUMN = "style_path"          # Column name for style image paths
    SYNTHETIC_COLUMN = "synthetic_path"  # Column name for synthetic image paths
    
    # =========================================================
    
    print("\n" + "=" * 80)
    print("Starting Dataset Reorganization")
    print("=" * 80)
    print("\nPlease verify the configuration:")
    print(f"  Base Path: {BASE_PATH}")
    print(f"  All Data CSV: {CSV_ALL_DATA}")
    print(f"  Captioned Data CSV: {CSV_CAPTIONED_DATA}")
    print(f"  Output Directory: {OUTPUT_DATASET_DIR}")
    print(f"  Column Names: content='{CONTENT_COLUMN}', style='{STYLE_COLUMN}', synthetic='{SYNTHETIC_COLUMN}'")
    print("=" * 80)
    
    # Uncomment the line below to require user confirmation
    # response = input("\nProceed with these settings? (yes/no): ")
    # if response.lower() != 'yes':
    #     print("Operation cancelled.")
    #     return
    
    # Create reorganizer instance
    reorganizer = DatasetReorganizer(
        base_path=BASE_PATH,
        csv_all_data=CSV_ALL_DATA,
        csv_captioned_data=CSV_CAPTIONED_DATA,
        output_dataset_dir=OUTPUT_DATASET_DIR
    )
    
    # Run reorganization
    reorganizer.reorganize(
        content_col=CONTENT_COLUMN,
        style_col=STYLE_COLUMN,
        synthetic_col=SYNTHETIC_COLUMN
    )


if __name__ == "__main__":
    main()

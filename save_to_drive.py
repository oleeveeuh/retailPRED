#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Google Drive Backup Utility for Retail Market Dynamics Project
Zips and backs up the entire project to Google Drive
"""

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
import sys


def get_drive_path():
    """
    Get the Google Drive path for the project.
    
    Returns:
        str: Path to project directory in Google Drive
    """
    try:
        from google.colab import drive
        
        # Check if drive is mounted
        if not os.path.exists('/content/drive'):
            print("="*60)
            print("Mounting Google Drive...")
            print("="*60)
            drive.mount('/content/drive')
            print("✓ Google Drive mounted")
        
        drive_path = '/content/drive/MyDrive/retail_market_dynamics'
        
        # Create project directory if it doesn't exist
        os.makedirs(drive_path, exist_ok=True)
        
        return drive_path
        
    except ImportError:
        print("⚠ Google Colab not detected. Using local path.")
        return os.path.join(os.getcwd(), 'retail_market_dynamics_backup')
    except Exception as e:
        print(f"⚠ Error with Google Drive: {e}")
        print("Using local path instead.")
        return os.path.join(os.getcwd(), 'retail_market_dynamics_backup')


def get_project_path():
    """
    Get the path to the project folder to backup.
    
    Returns:
        str: Path to project folder
    """
    # Try Colab path first
    if os.path.exists('/content/retail_market_dynamics'):
        return '/content/retail_market_dynamics'
    # Try local path
    elif os.path.exists('retail_market_dynamics'):
        return 'retail_market_dynamics'
    # Use current directory
    else:
        return os.getcwd()


def get_dir_size(path):
    """
    Calculate total size of directory.
    
    Args:
        path (str): Directory path
        
    Returns:
        int: Total size in bytes
    """
    total_size = 0
    
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"⚠ Error calculating size: {e}")
    
    return total_size


def format_size(size_bytes):
    """
    Convert bytes to human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def create_zip_backup(source_dir, output_zip_path, exclude_patterns=None):
    """
    Create a zip file backup of the source directory.
    
    Args:
        source_dir (str): Directory to backup
        output_zip_path (str): Path to output zip file
        exclude_patterns (list): Patterns to exclude (e.g., ['*.pyc', '__pycache__'])
        
    Returns:
        bool: True if successful
    """
    print("\n" + "="*60)
    print("Creating Backup Archive...")
    print("="*60)
    
    if exclude_patterns is None:
        exclude_patterns = [
            '*.pyc',
            '__pycache__',
            '.git',
            '.ipynb_checkpoints',
            '*.log',
            '*.tmp',
            '.DS_Store'
        ]
    
    try:
        # Create zip file
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Count files for progress
            all_files = []
            for root, dirs, files in os.walk(source_dir):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not any(
                    Path(d).match(pattern) or pattern in d for pattern in exclude_patterns
                )]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    # Check if file should be excluded
                    should_exclude = False
                    for pattern in exclude_patterns:
                        if Path(file).match(pattern) or pattern in file_path:
                            should_exclude = True
                            break
                    
                    if not should_exclude:
                        all_files.append(file_path)
            
            total_files = len(all_files)
            print(f"  Adding {total_files} files to archive...")
            
            # Add files to zip
            for i, file_path in enumerate(all_files):
                if os.path.exists(file_path):
                    # Create arcname (path within zip relative to source_dir)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
                    
                    # Progress indicator
                    if (i + 1) % 100 == 0 or (i + 1) == total_files:
                        print(f"  Progress: {i + 1}/{total_files} files ({100 * (i + 1) / total_files:.1f}%)")
        
        print(f"\n✓ Backup archive created: {output_zip_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating backup: {e}")
        import traceback
        traceback.print_exc()
        return False


def backup_to_drive(source_dir=None, output_dir=None):
    """
    Main backup function.
    
    Args:
        source_dir (str): Directory to backup (default: auto-detect)
        output_dir (str): Output directory (default: Google Drive model_backups)
        
    Returns:
        str: Path to created backup file
    """
    print("\n" + "="*60)
    print("RETAIL MARKET DYNAMICS - BACKUP TO GOOGLE DRIVE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get paths
    if source_dir is None:
        source_dir = get_project_path()
    
    if output_dir is None:
        drive_path = get_drive_path()
        output_dir = os.path.join(drive_path, 'model_backups')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print source information
    print(f"\nSource directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(source_dir):
        print(f"\n✗ Error: Source directory not found: {source_dir}")
        return None
    
    # Calculate source size
    print("\nCalculating directory size...")
    source_size = get_dir_size(source_dir)
    print(f"  Source size: {format_size(source_size)}")
    
    # Create timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'retail_market_dynamics_backup_{timestamp}.zip'
    output_zip_path = os.path.join(output_dir, output_filename)
    
    print(f"\nBackup will be saved to: {output_zip_path}")
    
    # Create backup
    if not create_zip_backup(source_dir, output_zip_path):
        print("\n✗ Backup failed!")
        return None
    
    # Get backup file size
    backup_size = os.path.getsize(output_zip_path)
    
    # Print summary
    print("\n" + "="*60)
    print("BACKUP SUMMARY")
    print("="*60)
    print(f"✓ Backup completed successfully!")
    print(f"\nSource:")
    print(f"  Directory: {source_dir}")
    print(f"  Size: {format_size(source_size)}")
    print(f"\nBackup:")
    print(f"  File: {output_filename}")
    print(f"  Location: {output_zip_path}")
    print(f"  Size: {format_size(backup_size)}")
    print(f"  Compression ratio: {100 * (1 - backup_size / source_size):.1f}%")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return output_zip_path


def list_backups(output_dir=None):
    """
    List all existing backups.
    
    Args:
        output_dir (str): Directory containing backups
        
    Returns:
        list: List of backup files
    """
    if output_dir is None:
        drive_path = get_drive_path()
        output_dir = os.path.join(drive_path, 'model_backups')
    
    if not os.path.exists(output_dir):
        print(f"\nBackup directory not found: {output_dir}")
        return []
    
    backup_files = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
    backup_files.sort(reverse=True)  # Newest first
    
    if backup_files:
        print("\n" + "="*60)
        print("EXISTING BACKUPS")
        print("="*60)
        for i, backup_file in enumerate(backup_files, 1):
            file_path = os.path.join(output_dir, backup_file)
            file_size = os.path.getsize(file_path)
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            print(f"{i}. {backup_file}")
            print(f"   Size: {format_size(file_size)}")
            print(f"   Date: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    else:
        print("\nNo backups found in the directory.")
    
    return backup_files


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup Retail Market Dynamics to Google Drive')
    parser.add_argument('--source', type=str, help='Source directory to backup')
    parser.add_argument('--output', type=str, help='Output directory for backup')
    parser.add_argument('--list', action='store_true', help='List existing backups')
    
    args = parser.parse_args()
    
    if args.list:
        list_backups()
    else:
        output_file = backup_to_drive(args.source, args.output)
        
        if output_file:
            print("\n✓ Backup saved successfully!")
            print(f"  Location: {output_file}")
        else:
            print("\n✗ Backup failed!")
            sys.exit(1)


if __name__ == "__main__":
    # If run directly without args, perform backup
    if len(sys.argv) == 1:
        output_file = backup_to_drive()
        
        if output_file:
            print("\n✓ Backup completed successfully!")
            print(f"  Location: {output_file}")
        else:
            print("\n✗ Backup failed!")
            sys.exit(1)
    else:
        main()


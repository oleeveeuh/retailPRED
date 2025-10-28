#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for retail_market_dynamics project in Google Colab.
This script initializes the project structure, mounts Google Drive, and installs required libraries.
"""

import os
import sys
from pathlib import Path

# Step 1: Create folder structure
def create_folder_structure(base_path="/content/retail_market_dynamics"):
    """Create the project folder structure."""
    print("="*60)
    print("Creating project folder structure...")
    print("="*60)
    
    folders = [
        "data/raw",
        "data/processed",
        "scripts",
        "models",
        "visuals",
        "notebooks"
    ]
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"✓ Created: {folder_path}")
    
    # Create .gitkeep files in empty directories
    gitkeep_paths = [
        os.path.join(base_path, "data/raw/.gitkeep"),
        os.path.join(base_path, "data/processed/.gitkeep"),
        os.path.join(base_path, "models/.gitkeep"),
        os.path.join(base_path, "visuals/.gitkeep"),
    ]
    
    for gitkeep_path in gitkeep_paths:
        with open(gitkeep_path, 'w') as f:
            f.write("# This file ensures the directory is tracked by git\n")
    
    print(f"\n✓ Project structure created at: {base_path}")
    return base_path


# Step 2: Mount Google Drive
def mount_google_drive():
    """Mount Google Drive and set DRIVE_PATH variable."""
    print("\n" + "="*60)
    print("Mounting Google Drive...")
    print("="*60)
    
    from google.colab import drive
    
    # Mount Google Drive
    try:
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")
        
        # Set DRIVE_PATH variable
        DRIVE_PATH = "/content/drive/MyDrive/retail_market_dynamics"
        
        # Create the project directory in Google Drive
        os.makedirs(DRIVE_PATH, exist_ok=True)
        os.makedirs(os.path.join(DRIVE_PATH, "data"), exist_ok=True)
        print(f"✓ Drive path set: {DRIVE_PATH}")
        
        return DRIVE_PATH
    except Exception as e:
        print(f"⚠ Warning: Could not mount Google Drive: {e}")
        print("Continuing without Google Drive mount...")
        return None


# Step 3 & 4: Install and import required libraries
def install_libraries():
    """Install all required libraries."""
    print("\n" + "="*60)
    print("Installing required libraries...")
    print("="*60)
    
    libraries = [
        "pandas",
        "numpy",
        "requests",
        "matplotlib",
        "seaborn",
        "yfinance",
        "fredapi",
        "scikit-learn",
        "statsmodels",
        "plotly",
        "prophet",
        "pandas-datareader",
        "beautifulsoup4"
    ]
    
    failed_installs = []
    
    for lib in libraries:
        try:
            print(f"Installing {lib}...")
            os.system(f"pip install {lib} -q")
            print(f"✓ {lib} installed")
        except Exception as e:
            print(f"✗ Failed to install {lib}: {e}")
            failed_installs.append(lib)
    
    if failed_installs:
        print(f"\n⚠ Failed to install: {failed_installs}")
    else:
        print("\n✓ All libraries installed successfully!")


def import_and_verify_libraries():
    """Import libraries and print versions."""
    print("\n" + "="*60)
    print("Verifying library installations...")
    print("="*60)
    
    import warnings
    warnings.filterwarnings('ignore')
    
    libraries_and_versions = {}
    
    # Dictionary of library imports
    imports = {
        "pandas": ("pandas", "pd"),
        "numpy": ("numpy", "np"),
        "requests": ("requests", None),
        "matplotlib": ("matplotlib.pyplot", "plt"),
        "seaborn": ("seaborn", "sns"),
        "yfinance": ("yfinance", "yf"),
        "statsmodels": ("statsmodels.api", "sm"),
        "sklearn": ("sklearn", None),
        "plotly": ("plotly", "go"),
    }
    
    successful_imports = []
    failed_imports = []
    
    for lib_name, (import_name, alias) in imports.items():
        try:
            if alias:
                exec(f"import {import_name} as {alias}")
            else:
                exec(f"import {import_name}")
            
            # Get version
            try:
                version = eval(f"{alias or import_name}.__version__")
                libraries_and_versions[lib_name] = version
                print(f"✓ {lib_name}: {version}")
                successful_imports.append(lib_name)
            except:
                libraries_and_versions[lib_name] = "installed (version unknown)"
                print(f"✓ {lib_name}: installed (version unknown)")
                successful_imports.append(lib_name)
                
        except Exception as e:
            print(f"✗ {lib_name}: failed to import - {e}")
            failed_imports.append(lib_name)
    
    # Special handling for prophet and fredapi
    try:
        from prophet import Prophet
        libraries_and_versions["prophet"] = Prophet.__version__
        print(f"✓ prophet: {Prophet.__version__}")
        successful_imports.append("prophet")
    except Exception as e:
        print(f"✗ prophet: failed to import - {e}")
        failed_imports.append("prophet")
    
    try:
        from fredapi import Fred
        libraries_and_versions["fredapi"] = "installed"
        print(f"✓ fredapi: installed")
        successful_imports.append("fredapi")
    except Exception as e:
        print(f"✗ fredapi: failed to import - {e}")
        failed_imports.append("fredapi")
    
    # Print summary
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    print(f"✓ Successfully imported: {len(successful_imports)}/{len(imports)+2}")
    if failed_imports:
        print(f"✗ Failed to import: {failed_imports}")
    print("\nAll library versions:")
    for lib, version in sorted(libraries_and_versions.items()):
        print(f"  {lib}: {version}")


def main():
    """Main setup function."""
    print("\n" + "="*60)
    print("RETAIL MARKET DYNAMICS - PROJECT SETUP")
    print("="*60)
    
    # Create folder structure
    project_path = create_folder_structure()
    
    # Mount Google Drive
    DRIVE_PATH = mount_google_drive()
    
    # Install libraries
    install_libraries()
    
    # Import and verify
    import_and_verify_libraries()
    
    # Final summary
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print(f"Project path: {project_path}")
    if DRIVE_PATH:
        print(f"Drive path: {DRIVE_PATH}")
    print("\nNext steps:")
    print("1. Start working in the notebooks/ directory")
    print("2. Save data in data/raw/ (for new data) or data/processed/ (for cleaned data)")
    print("3. Save visualizations in visuals/")
    print("4. Save trained models in models/")
    print("="*60)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
RetailPRED Setup Verification Script

This script checks that your environment is properly configured for training.
Run this before training to ensure everything is set up correctly.

Usage:
    python test_setup.py
"""

import sys
import os
from pathlib import Path
import subprocess
import importlib.util

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def check_mark(passed: bool) -> str:
    """Return checkmark or X based on result"""
    return f"{Colors.GREEN}✅{Colors.END}" if passed else f"{Colors.RED}❌{Colors.END}"

def warn_mark() -> str:
    """Return warning mark"""
    return f"{Colors.YELLOW}⚠️ {Colors.END}"

def info_mark() -> str:
    """Return info mark"""
    return f"{Colors.BLUE}ℹ️ {Colors.END}"


# ============================================================================
# Check Functions
# ============================================================================

def check_train_script() -> tuple[bool, str]:
    """Check if train.py exists and is executable"""
    train_py = Path(__file__).parent / "train.py"

    if not train_py.exists():
        return False, "train.py not found"

    # Check if it's executable
    if not os.access(train_py, os.X_OK):
        return False, "train.py exists but is not executable (run: chmod +x train.py)"

    # Check basic imports work
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, '.'); from config import DATABASE_PATH, MODELS_DIR"],
            capture_output=True,
            timeout=5,
            cwd=Path(__file__).parent
        )
        if result.returncode != 0:
            return False, "train.py config import failed"
    except Exception as e:
        return False, f"train.py validation failed: {e}"

    return True, "train.py exists and is executable"


def check_directories() -> tuple[bool, list]:
    """Check if required directories exist"""
    checks = []
    project_root = Path(__file__).parent

    required_dirs = [
        ("models/", project_root / "models"),
        ("backend/ml/models/", project_root / "backend" / "ml" / "models"),
        ("data/", project_root / "data"),
        ("logs/", project_root / "logs"),
        ("project_root/data_multi_resolution/", project_root / "project_root" / "data_multi_resolution"),
    ]

    all_exist = True
    results = []

    for name, path in required_dirs:
        exists = path.exists()
        if not exists and name not in ["logs/"]:  # logs/ is auto-created
            all_exist = False
        results.append((name, exists, str(path)))

    return all_exist, results


def check_database() -> tuple[bool, str]:
    """Check if database exists and is accessible"""
    project_root = Path(__file__).parent
    db_path = project_root / "data" / "retailpred.db"

    if not db_path.exists():
        return False, "Database not found at data/retailpred.db"

    # Check if it's a valid SQLite database
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        conn.close()
        return True, f"Database found ({db_path.stat().st_size / 1024 / 1024:.1f} MB)"
    except Exception as e:
        return False, f"Database exists but is not valid: {e}"


def check_gitignore() -> tuple[bool, list]:
    """Check .gitignore configuration"""
    project_root = Path(__file__).parent
    gitignore_path = project_root / ".gitignore"

    if not gitignore_path.exists():
        return False, [("❌", ".gitignore file not found")]

    gitignore_content = gitignore_path.read_text().lower()

    # Check patterns
    checks = [
        ("*.pkl", "Should ignore *.pkl files"),
        ("*.db", "Should ignore *.db files"),
        ("*.sqlite", "Should ignore *.sqlite files"),
        ("models/", "Should NOT ignore models/ directory"),
    ]

    results = []
    issues = []

    # Check for required ignores
    has_pkl_ignore = "*.pkl" in gitignore_content or ".pkl" in gitignore_content
    has_db_ignore = "*.db" in gitignore_content or ".db" in gitignore_content

    # Check if models/ is explicitly ignored (bad!)
    models_ignored = "models/" in gitignore_content or "models" in gitignore_content.split("\n")

    if has_pkl_ignore:
        results.append(("✅", "*.pkl files are ignored"))
    else:
        results.append(("❌", "*.pkl files should be ignored"))
        issues.append("Add '*.pkl' to .gitignore")

    if has_db_ignore:
        results.append(("✅", "*.db files are ignored"))
    else:
        results.append(("❌", "*.db files should be ignored"))
        issues.append("Add '*.db' to .gitignore")

    if not models_ignored or "/models" in gitignore_content:
        results.append(("✅", "models/ directory is NOT ignored"))
    else:
        results.append(("❌", "models/ directory should NOT be ignored"))
        issues.append("Remove 'models/' from .gitignore (use backend/ml/models/*.pkl instead)")

    return len(issues) == 0, results, issues


def check_requirements() -> tuple[bool, list]:
    """Check if requirements.txt exists and key packages are available"""
    project_root = Path(__file__).parent

    # Find requirements.txt files
    req_files = [
        project_root / "requirements.txt",
        project_root / "backend" / "requirements.txt",
    ]

    found_files = [f for f in req_files if f.exists()]

    if not found_files:
        return False, [("❌", "No requirements.txt found")]

    results = []

    # Check key packages
    key_packages = {
        'lightgbm': 'LightGBM',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'statsforecast': 'statsforecast',
        'shap': 'SHAP',
    }

    missing_packages = []

    for module_name, display_name in key_packages.items():
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            results.append(("✅", f"{display_name} is installed"))
        else:
            results.append(("❌", f"{display_name} is NOT installed"))
            missing_packages.append(display_name)

    return len(missing_packages) == 0, results, missing_packages


def check_config_module() -> tuple[bool, str]:
    """Check if config.py exists and is valid"""
    config_path = Path(__file__).parent / "config.py"

    if not config_path.exists():
        return False, "config.py not found"

    try:
        # Try to import and validate
        sys.path.insert(0, str(Path(__file__).parent))
        from config import DATABASE_PATH, MODELS_DIR, BACKEND_MODELS_DIR

        # Check paths are defined
        if DATABASE_PATH is None:
            return False, "config.py exists but DATABASE_PATH is not defined"

        return True, "config.py is valid"
    except ImportError as e:
        return False, f"config.py exists but import failed: {e}"
    except Exception as e:
        return False, f"config.py validation failed: {e}"


def check_training_data() -> tuple[bool, list]:
    """Check if training data files exist"""
    project_root = Path(__file__).parent
    data_dir = project_root / "project_root" / "data_multi_resolution"

    if not data_dir.exists():
        return False, [("❌", f"Data directory not found: {data_dir}")]

    # Check for key data files
    required_files = [
        "retail_total_sales_multi_resolution.csv",
        "retail_automobile_dealers_multi_resolution.csv",
    ]

    results = []
    missing = []

    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            results.append(("✅", f"{filename} ({size:.1f} KB)"))
        else:
            results.append(("❌", f"{filename} not found"))
            missing.append(filename)

    return len(missing) == 0, results, missing


def check_symlinks() -> tuple[bool, list, list]:
    """Check if symlinks are properly set up"""
    import os
    project_root = Path(__file__).parent

    checks = [
        ("backend/data/retailpred.db", project_root / "backend" / "data" / "retailpred.db"),
        ("project_root/data/retailpred.db", project_root / "project_root" / "data" / "retailpred.db"),
    ]

    results = []
    issues = []

    for name, path in checks:
        # Use is_symlink() first - it returns True even for broken symlinks
        if path.is_symlink():
            target = Path(os.readlink(path))
            # For relative symlinks, manually resolve without using Path.resolve()
            # to avoid symlink loop issues
            if not target.is_absolute():
                # Build absolute path manually
                target_path = path.parent / target
                # Check if target exists using os.path.exists
                target_exists = os.path.exists(str(target_path))
            else:
                target_path = target
                target_exists = os.path.exists(str(target_path))

            if target_exists:
                results.append(("✅", f"{name} is a symlink to {target}"))
            else:
                results.append(("⚠️", f"{name} is a symlink but target doesn't exist"))
                issues.append(f"{name} symlink target is missing")
        elif path.is_file():
            results.append(("ℹ️", f"{name} exists as a regular file"))
            issues.append(f"{name} should be a symlink to data/retailpred.db")
        elif not path.exists():
            results.append(("❌", f"{name} does not exist"))
            issues.append(f"{name} symlink needs to be created")
        else:
            results.append(("ℹ️", f"{name} exists but is not a symlink"))
            issues.append(f"{name} should be a symlink to data/retailpred.db")

    return len(issues) == 0, results, issues


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all checks and print results"""
    print(f"{Colors.BOLD}RetailPRED Setup Verification{Colors.END}")
    print("=" * 60)
    print()

    all_passed = True
    sections = []

    # 1. Check train.py
    print(f"{Colors.BOLD}1. Training Script{Colors.END}")
    passed, msg = check_train_script()
    print(f"  {check_mark(passed)} {msg}")
    if not passed:
        all_passed = False
    print()

    # 2. Check directories
    print(f"{Colors.BOLD}2. Directory Structure{Colors.END}")
    passed, results = check_directories()
    for name, exists, path in results:
        print(f"  {check_mark(exists)} {name} - {path}")
        if not exists and name != "logs/":
            all_passed = False
    print()

    # 3. Check database
    print(f"{Colors.BOLD}3. Database{Colors.END}")
    passed, msg = check_database()
    print(f"  {check_mark(passed)} {msg}")
    if not passed:
        all_passed = False
    print()

    # 4. Check config module
    print(f"{Colors.BOLD}4. Configuration Module{Colors.END}")
    passed, msg = check_config_module()
    print(f"  {check_mark(passed)} {msg}")
    if not passed:
        all_passed = False
    print()

    # 5. Check symlinks
    print(f"{Colors.BOLD}5. Symlinks{Colors.END}")
    passed, results, issues = check_symlinks()
    for mark, msg in results:
        print(f"  {mark} {msg}")
    if not passed:
        all_passed = False
    print()

    # 6. Check requirements
    print(f"{Colors.BOLD}6. Python Dependencies{Colors.END}")
    passed, results, missing = check_requirements()
    for mark, msg in results:
        print(f"  {mark} {msg}")
    if not passed:
        all_passed = False
        print(f"  {info_mark()}To install missing packages: pip install -r requirements.txt")
    print()

    # 7. Check .gitignore
    print(f"{Colors.BOLD}7. Git Configuration{Colors.END}")
    passed, results, issues = check_gitignore()
    for mark, msg in results:
        print(f"  {mark} {msg}")
    if issues:
        all_passed = False
    print()

    # 8. Check training data
    print(f"{Colors.BOLD}8. Training Data{Colors.END}")
    passed, results, missing = check_training_data()
    for mark, msg in results:
        print(f"  {mark} {msg}")
    if not passed:
        all_passed = False
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL CHECKS PASSED!{Colors.END}")
        print()
        print("Your environment is ready for training.")
        print("Run: python train.py")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ SOME CHECKS FAILED{Colors.END}")
        print()
        print("Please fix the issues above before training.")

        # Print actionable fixes
        print()
        print(f"{Colors.BOLD}Actionable Fixes:{Colors.END}")

        # Check for missing packages
        if missing:
            print(f"\n{Colors.YELLOW}Missing Python packages:{Colors.END}")
            for pkg in missing:
                print(f"  pip install {pkg}")

        # Check for gitignore issues
        passed, results, issues = check_gitignore()
        if issues:
            print(f"\n{Colors.YELLOW}.gitignore fixes needed:{Colors.END}")
            for issue in issues:
                print(f"  - {issue}")

        # Check for directory issues
        passed, results = check_directories()
        for name, exists, path in results:
            if not exists and name != "logs/":
                print(f"\n{Colors.YELLOW}Create directory:{Colors.END}")
                print(f"  mkdir -p {path}")

        # Check for symlink issues
        passed, results, issues = check_symlinks()
        if issues:
            print(f"\n{Colors.YELLOW}Create symlinks:{Colors.END}")
            for issue in issues:
                print(f"  - {issue}")

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash
# RetailPRED Airflow Setup Script
#
# This script prepares RetailPRED for Airflow automation by:
# 1. Creating required directories
# 2. Ensuring train.py exists
# 3. Creating update_db.py for database updates
# 4. Updating .gitignore
# 5. Ensuring requirements.txt is complete
# 6. Running verification tests
#
# Usage: ./setup_for_airflow.sh
#
# This script is idempotent - safe to run multiple times

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print colored message
print_step() {
    echo -e "${BLUE}${BOLD}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Track what was done
CREATED_DIRS=()
UPDATED_FILES=()
SKIPPED_ITEMS=()

echo -e "${BOLD}======================================${NC}"
echo -e "${BOLD}  RetailPRED Airflow Setup${NC}"
echo -e "${BOLD}======================================${NC}"
echo ""
echo "Script directory: $SCRIPT_DIR"
echo ""

# ============================================================================
# 1. Create required directories
# ============================================================================

print_step "Creating required directories..."

DIRECTORIES=(
    "models"
    "logs"
    "backend/data"
)

for dir in "${DIRECTORIES[@]}"; do
    DIR_PATH="$SCRIPT_DIR/$dir"
    if [ ! -d "$DIR_PATH" ]; then
        mkdir -p "$DIR_PATH"
        print_success "Created $dir/"
        CREATED_DIRS+=("$dir/")
    else
        print_success "Exists: $dir/"
        SKIPPED_ITEMS+=("$dir/ (already exists)")
    fi

    # Add .gitkeep if models/ is empty
    if [ "$dir" = "models" ]; then
        GITKEEP="$DIR_PATH/.gitkeep"
        if [ ! -f "$GITKEEP" ]; then
            touch "$GITKEEP"
            print_success "Created models/.gitkeep"
        fi
    fi
done

echo ""

# ============================================================================
# 2. Ensure train.py exists
# ============================================================================

print_step "Checking train.py..."

TRAIN_PY="$SCRIPT_DIR/train.py"
if [ ! -f "$TRAIN_PY" ]; then
    print_warning "train.py not found. Creating template..."
    cat > "$TRAIN_PY" << 'EOF'
#!/usr/bin/env python3
"""
RetailPRED Training Script for Airflow

This is a template training script. Run ./setup_for_airflow.sh to create
the full version, or use the existing train.py from the repository.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import DATABASE_PATH, MODELS_DIR
    print(f"Configuration loaded successfully!")
    print(f"  Database: {DATABASE_PATH}")
    print(f"  Models: {MODELS_DIR}")
    print("\nTo train a model, run:")
    print("  python train.py --help")
except ImportError as e:
    print(f"Error: Could not import config: {e}")
    sys.exit(1)
EOF
    chmod +x "$TRAIN_PY"
    print_success "Created train.py (template version)"
    CREATED_DIRS+=("train.py")
else
    print_success "Exists: train.py"
    SKIPPED_ITEMS+=("train.py (already exists)")
fi

echo ""

# ============================================================================
# 3. Create update_db.py for database updates
# ============================================================================

print_step "Creating update_db.py..."

UPDATE_DB="$SCRIPT_DIR/update_db.py"
if [ ! -f "$UPDATE_DB" ]; then
    cat > "$UPDATE_DB" << 'EOF'
#!/usr/bin/env python3
"""
RetailPRED Database Update Script

This script updates the database with new predictions and metrics.
Intended to be run after model training.

Usage:
    python update_db.py --model-path models/model_latest.pkl
    python update_db.py --category total_sales
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_PATH, MODELS_DIR


def update_database(model_path: str, category: str = "total_sales") -> dict:
    """
    Update the database with model information

    Args:
        model_path: Path to the trained model file
        category: Category name

    Returns:
        Dictionary with update results
    """
    import sqlite3

    result = {
        "success": False,
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "category": category,
    }

    # Check if model exists
    model_file = Path(model_path)
    if not model_file.exists():
        # Try relative to models directory
        model_file = MODELS_DIR / Path(model_path).name
        if not model_file.exists():
            result["error"] = f"Model file not found: {model_path}"
            return result

    # Load metrics if available
    metrics_path = MODELS_DIR / "latest_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_data = json.load(f)
            metrics = metrics_data.get("metrics", {})

    # Connect to database
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Check if model_metadata table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='model_metadata'
        """)

        if cursor.fetchone():
            # Update model metadata
            cursor.execute("""
                INSERT OR REPLACE INTO model_metadata
                (model_name, model_type, training_date, metrics, file_path, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{category}_latest",
                "ensemble",
                datetime.now().isoformat(),
                json.dumps(metrics),
                str(model_file),
                1,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ))
            conn.commit()
            result["success"] = True
            result["message"] = "Database updated successfully"
        else:
            result["warning"] = "model_metadata table not found in database"

        conn.close()
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Update RetailPRED database with model information"
    )
    parser.add_argument(
        "--model-path",
        default=str(MODELS_DIR / "model_latest.pkl"),
        help="Path to trained model file"
    )
    parser.add_argument(
        "--category",
        default="total_sales",
        help="Category name"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RetailPRED Database Update")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Category: {args.category}")
    print(f"Database: {DATABASE_PATH}")
    print("=" * 60)
    print()

    result = update_database(args.model_path, args.category)

    if result.get("success"):
        print(f"✓ {result.get('message', 'Update successful')}")
        return 0
    else:
        error = result.get("error", result.get("warning", "Unknown error"))
        print(f"✗ {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
EOF
    chmod +x "$UPDATE_DB"
    print_success "Created update_db.py"
    CREATED_DIRS+=("update_db.py")
else
    print_success "Exists: update_db.py"
    SKIPPED_ITEMS+=("update_db.py (already exists)")
fi

echo ""

# ============================================================================
# 4. Update .gitignore
# ============================================================================

print_step "Updating .gitignore..."

GITIGNORE="$SCRIPT_DIR/.gitignore"
GITIGNORE_BACKUP="$SCRIPT_DIR/.gitignore.backup"

# Create .gitignore if it doesn't exist
if [ ! -f "$GITIGNORE" ]; then
    touch "$GITIGNORE"
    print_success "Created .gitignore"
    CREATED_DIRS+=(".gitignore")
else
    # Backup existing .gitignore
    cp "$GITIGNORE" "$GITIGNORE_BACKUP"
    SKIPPED_ITEMS+=(".gitignore backup created)")
fi

# Patterns to ensure are in .gitignore
PATTERNS=(
    "*.pkl"
    "*.joblib"
    "*.db"
    "*.sqlite"
    "*.sqlite3"
    "__pycache__/"
    "*.pyc"
    ".pytest_cache/"
    ".DS_Store"
    "logs/*.log"
)

# Check and add missing patterns
ADDED_PATTERNS=()
for pattern in "${PATTERNS[@]}"; do
    if ! grep -q "^$pattern" "$GITIGNORE" 2>/dev/null; then
        echo "$pattern" >> "$GITIGNORE"
        ADDED_PATTERNS+=("$pattern")
    fi
done

# Ensure models/ directory is tracked (create .gitkeep comment)
if ! grep -q "models/.gitkeep" "$GITIGNORE" 2>/dev/null; then
    echo "" >> "$GITIGNORE"
    echo "# Keep models/ directory structure but ignore .pkl files" >> "$GITIGNORE"
    echo "models/*.pkl" >> "$GITIGNORE"
    echo "!.gitkeep" >> "$GITIGNORE"
    ADDED_PATTERNS+=("models/ .gitkeep pattern")
fi

if [ ${#ADDED_PATTERNS[@]} -gt 0 ]; then
    print_success "Added to .gitignore:"
    for pattern in "${ADDED_PATTERNS[@]}"; do
        echo "    $pattern"
    done
    UPDATED_FILES+=(".gitignore")
else
    print_success ".gitignore up to date"
    SKIPPED_ITEMS+=(".gitignore (no changes needed)")
fi

# Remove backup if no changes were made
if [ ${#ADDED_PATTERNS[@]} -eq 0 ] && [ -f "$GITIGNORE_BACKUP" ]; then
    rm "$GITIGNORE_BACKUP"
fi

echo ""

# ============================================================================
# 5. Ensure requirements.txt is complete
# ============================================================================

print_step "Checking requirements.txt..."

REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
BACKEND_REQUIREMENTS="$SCRIPT_DIR/backend/requirements.txt"

# Required packages for Airflow/ML
REQUIRED_PACKAGES=(
    "pandas>="
    "numpy>="
    "scikit-learn>="
    "lightgbm>="
    "joblib>="
)

# Create or update main requirements.txt
if [ ! -f "$REQUIREMENTS" ]; then
    print_success "Creating requirements.txt"
    cat > "$REQUIREMENTS" << 'EOF'
# RetailPRED Core Requirements
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
joblib>=1.3.0

# Optional: Advanced Analytics
statsforecast>=1.5.0
shap>=0.42.0

# API Dependencies
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
EOF
    print_success "Created requirements.txt"
    CREATED_DIRS+=("requirements.txt")
else
    # Check for missing packages
    MISSING_PACKAGES=()
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! grep -q "$pkg" "$REQUIREMENTS" 2>/dev/null; then
            MISSING_PACKAGES+=("$pkg")
        fi
    done

    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        print_warning "Adding missing packages to requirements.txt:"
        for pkg in "${MISSING_PACKAGES[@]}"; do
            echo "$pkg" >> "$REQUIREMENTS"
            echo "    $pkg"
        done
        UPDATED_FILES+=("requirements.txt")
    else
        print_success "requirements.txt is complete"
        SKIPPED_ITEMS+=("requirements.txt (all packages present)")
    fi
fi

echo ""

# ============================================================================
# 6. Create symlinks for database
# ============================================================================

print_step "Setting up database symlinks..."

# Create backend/data symlink
BACKEND_DATA_DIR="$SCRIPT_DIR/backend/data"
BACKEND_DB_LINK="$BACKEND_DATA_DIR/retailpred.db"
MAIN_DB="$SCRIPT_DIR/data/retailpred.db"

mkdir -p "$BACKEND_DATA_DIR"

# Remove old symlink if it exists
if [ -L "$BACKEND_DB_LINK" ]; then
    rm "$BACKEND_DB_LINK"
fi

# Create new symlink
if [ ! -e "$BACKEND_DB_LINK" ]; then
    ln -s "../../data/retailpred.db" "$BACKEND_DB_LINK"
    print_success "Created backend/data/retailpred.db symlink"
    CREATED_DIRS+=("backend/data/retailpred.db symlink")
else
    print_success "Exists: backend/data/retailpred.db symlink"
    SKIPPED_ITEMS+=("backend/data/ symlink (already exists)")
fi

# Create project_root/data symlink
PROJECT_ROOT_DATA_DIR="$SCRIPT_DIR/project_root/data"
PROJECT_ROOT_DB_LINK="$PROJECT_ROOT_DATA_DIR/retailpred.db"

mkdir -p "$PROJECT_ROOT_DATA_DIR"

if [ -L "$PROJECT_ROOT_DB_LINK" ]; then
    rm "$PROJECT_ROOT_DB_LINK"
fi

if [ ! -e "$PROJECT_ROOT_DB_LINK" ]; then
    ln -s "../../data/retailpred.db" "$PROJECT_ROOT_DB_LINK"
    print_success "Created project_root/data/retailpred.db symlink"
    CREATED_DIRS+=("project_root/data/retailpred.db symlink")
else
    print_success "Exists: project_root/data/retailpred.db symlink"
    SKIPPED_ITEMS+=("project_root/data/ symlink (already exists)")
fi

echo ""

# ============================================================================
# 7. Run verification
# ============================================================================

print_step "Running verification tests..."

TEST_SETUP="$SCRIPT_DIR/test_setup.py"
if [ -f "$TEST_SETUP" ]; then
    if python "$TEST_SETUP"; then
        print_success "All verification checks passed!"
    else
        print_warning "Some verification checks failed. Review the output above."
        print_warning "You may need to install missing packages:"
        echo "    pip install -r requirements.txt"
    fi
else
    print_warning "test_setup.py not found. Skipping verification."
fi

echo ""
echo ""

# ============================================================================
# Summary
# ============================================================================

echo -e "${BOLD}======================================${NC}"
echo -e "${BOLD}  Setup Summary${NC}"
echo -e "${BOLD}======================================${NC}"
echo ""

if [ ${#CREATED_DIRS[@]} -gt 0 ]; then
    echo -e "${GREEN}${BOLD}Created:${NC}"
    for item in "${CREATED_DIRS[@]}"; do
        echo "  • $item"
    done
    echo ""
fi

if [ ${#UPDATED_FILES[@]} -gt 0 ]; then
    echo -e "${YELLOW}${BOLD}Updated:${NC}"
    for item in "${UPDATED_FILES[@]}"; do
        echo "  • $item"
    done
    echo ""
fi

if [ ${#SKIPPED_ITEMS[@]} -gt 0 ]; then
    echo -e "${BLUE}${BOLD}Already Exists (Skipped):${NC}"
    for item in "${SKIPPED_ITEMS[@]}"; do
        echo "  • $item"
    done
    echo ""
fi

echo -e "${BOLD}Next Steps:${NC}"
echo "  1. Install dependencies:  pip install -r requirements.txt"
echo "  2. Train a model:         python train.py --help"
echo "  3. Update database:       python update_db.py --help"
echo "  4. Verify setup:          python test_setup.py"
echo ""

echo -e "${GREEN}${BOLD}✓ Setup complete!${NC}"
echo ""

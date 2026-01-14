#!/bin/bash
# Deploy Retrained Models - Simple Version
# 
# This script:
# 1. Backs up original models
# 2. Deploys all 11 RandomForest v2 models
# 3. Deploys only 4 overfitting LGBM v2 models
# 4. Keeps 7 well-tuned LGBM models unchanged

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/backend/ml/models"

echo "================================================================================"
echo "DEPLOYING RETRAINED MODELS"
echo "================================================================================"
echo ""
echo "Deployment Plan:"
echo "  • RandomForest: 11/11 models (all v2)"
echo "  • LGBM: 4/11 models (only overfitting models)"
echo "  • LGBM: 7/11 models (keep original, already optimal)"
echo ""
echo "================================================================================"
echo ""

# Create backup directory with timestamp
BACKUP_DIR="$MODELS_DIR/backup_original_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Step 1: Backing up original models..."
echo "--------------------------------------------------------------------------------"

# Backup all RandomForest models
for model in automobile_dealers building_materials clothing_accessories electronics_and_appliances food_beverage furniture_home_furnishings gasoline_stations general_merchandise health_personal_care sporting_goods_hobby total_sales; do
    if [ -f "$MODELS_DIR/${model}_RandomForest_model.pkl" ]; then
        cp "$MODELS_DIR/${model}_RandomForest_model.pkl" "$BACKUP_DIR/"
        echo "  ✅ Backed up: ${model}_RandomForest_model.pkl"
    fi
done

# Backup all LGBM models
for model in automobile_dealers building_materials clothing_accessories electronics_and_appliances food_beverage furniture_home_furnishings gasoline_stations general_merchandise health_personal_care sporting_goods_hobby total_sales; do
    if [ -f "$MODELS_DIR/${model}_LGBM_model.pkl" ]; then
        cp "$MODELS_DIR/${model}_LGBM_model.pkl" "$BACKUP_DIR/"
        echo "  ✅ Backed up: ${model}_LGBM_model.pkl"
    fi
done

echo ""
echo "✅ Backup created at: $BACKUP_DIR"
echo ""

echo "Step 2: Deploying RandomForest v2 models (all 11)..."
echo "--------------------------------------------------------------------------------"

for model in automobile_dealers building_materials clothing_accessories electronics_and_appliances food_beverage furniture_home_furnishings gasoline_stations general_merchandise health_personal_care sporting_goods_hobby total_sales; do
    V2_FILE="$MODELS_DIR/${model}_RandomForest_model_v2.pkl"
    TARGET_FILE="$MODELS_DIR/${model}_RandomForest_model.pkl"
    
    if [ -f "$V2_FILE" ]; then
        # Remove old model
        rm -f "$TARGET_FILE"
        # Deploy v2 model
        cp "$V2_FILE" "$TARGET_FILE"
        echo "  ✅ Deployed: ${model}_RandomForest_model.pkl (v2)"
    else
        echo "  ❌ Not found: ${model}_RandomForest_model_v2.pkl"
    fi
done

echo ""
echo "Step 3: Deploying LGBM v2 models (4 overfitting models only)..."
echo "--------------------------------------------------------------------------------"

# Only deploy LGBM v2 for overfitting models
OVERFITTING_LGBM=(
    "sporting_goods_hobby"
    "furniture_home_furnishings"
    "building_materials"
    "general_merchandise"
)

for model in "${OVERFITTING_LGBM[@]}"; do
    V2_FILE="$MODELS_DIR/${model}_LGBM_model_v2.pkl"
    TARGET_FILE="$MODELS_DIR/${model}_LGBM_model.pkl"
    
    if [ -f "$V2_FILE" ]; then
        # Remove old model
        rm -f "$TARGET_FILE"
        # Deploy v2 model
        cp "$V2_FILE" "$TARGET_FILE"
        echo "  ✅ Deployed: ${model}_LGBM_model.pkl (v2)"
    else
        echo "  ❌ Not found: ${model}_LGBM_model_v2.pkl"
    fi
done

echo ""
echo "Step 4: Keeping LGBM models (7 well-tuned models unchanged)..."
echo "--------------------------------------------------------------------------------"

WELL_TUNED_LGBM=(
    "electronics_and_appliances: MASE 0.81 (already excellent)"
    "clothing_accessories: MASE 1.09 (already excellent)"
    "total_sales: MASE 1.12 (already excellent)"
    "health_personal_care: MASE 1.12 (already excellent)"
    "gasoline_stations: MASE 1.13 (already excellent)"
    "automobile_dealers: MASE 1.16 (already excellent)"
    "food_beverage: MASE 1.30 (already excellent)"
)

for model_info in "${WELL_TUNED_LGBM[@]}"; do
    model="${model_info%%:*}"
    TARGET_FILE="$MODELS_DIR/${model}_LGBM_model.pkl"
    
    if [ -f "$TARGET_FILE" ]; then
        echo "  ✅ Kept original: ${model}_LGBM_model.pkl ($model_info)"
    fi
done

echo ""
echo "================================================================================"
echo "DEPLOYMENT COMPLETE"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  • 11 RandomForest models deployed (v2)"
echo "  • 4 LGBM models deployed (v2 - overfitting models only)"
echo "  • 7 LGBM models kept (original - already optimal)"
echo "  • Total: 15 models upgraded, 7 unchanged"
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""
echo "To rollback if needed:"
echo "  cp $BACKUP_DIR/*.pkl $MODELS_DIR/"
echo ""
echo "================================================================================"

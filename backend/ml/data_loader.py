"""
Data Loader for RetailPRED
Loads and prepares retail sales data for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Default data path
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "project_root" / "data_processed" / "retail_total_sales_processed.csv"

# All 12 retail categories
RETAIL_CATEGORIES = {
    "total_retail_sales": "retail_total_sales_processed.csv",
    "automobile_dealers": "retail_automobile_dealers_processed.csv",
    "building_materials_garden": "retail_building_material_and_garden_equipment_processed.csv",
    "clothing_accessories": "retail_clothing_and_clothing_accessories_stores_processed.csv",
    "electronics_appliances": "retail_electronics_and_appliance_stores_processed.csv",
    "food_beverage_stores": "retail_food_and_beverage_stores_processed.csv",
    "furniture_home_furnishings": "retail_furniture_and_home_furnishings_stores_processed.csv",
    "gasoline_stations": "retail_gasoline_stations_processed.csv",
    "general_merchandise": "retail_general_merchandise_stores_processed.csv",
    "health_personal_care": "retail_health_and_personal_care_stores_processed.csv",
    "nonstore_retailers": "retail_nonstore_retailers_processed.csv",
    "sporting_goods_hobby": "retail_sporting_goods_hobby_and_musical_instrument_stores_processed.csv",
}

# Display names for categories
CATEGORY_DISPLAY_NAMES = {
    "total_retail_sales": "Total Retail Sales",
    "automobile_dealers": "Automobile Dealers",
    "building_materials_garden": "Building Materials & Garden",
    "clothing_accessories": "Clothing & Accessories",
    "electronics_appliances": "Electronics & Appliances",
    "food_beverage_stores": "Food & Beverage Stores",
    "furniture_home_furnishings": "Furniture & Home Furnishings",
    "gasoline_stations": "Gasoline Stations",
    "general_merchandise": "General Merchandise",
    "health_personal_care": "Health & Personal Care",
    "nonstore_retailers": "Nonstore Retailers",
    "sporting_goods_hobby": "Sporting Goods & Hobby",
}


class RetailDataLoader:
    """
    Loads and prepares retail sales data for ML training
    """

    def __init__(self, data_path: Optional[Path] = None, category: str = "total_retail_sales"):
        """
        Initialize data loader

        Args:
            data_path: Path to processed CSV file (optional, overrides category)
            category: Retail category to load (default: total_retail_sales)
        """
        if data_path:
            self.data_path = data_path
        else:
            # Get filename for category
            data_dir = Path(__file__).parent.parent.parent / "project_root" / "data_processed"
            filename = RETAIL_CATEGORIES.get(category, RETAIL_CATEGORIES["total_retail_sales"])
            self.data_path = data_dir / filename

        self.category = category
        self.data = None
        self.feature_columns = None
        self.target_column = "y"

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV

        Returns:
            DataFrame with loaded data
        """
        category_name = CATEGORY_DISPLAY_NAMES.get(self.category, self.category)
        logger.info(f"Loading {category_name} data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.data = pd.read_csv(self.data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date')

        logger.info(f"✓ Loaded {len(self.data)} records from {self.data['date'].min()} to {self.data['date'].max()}")

        return self.data

    def prepare_features_and_target(
        self,
        feature_subset: Optional[list] = None,
        drop_features: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepare features and target for training

        Args:
            feature_subset: List of specific features to use (if None, uses all available)
            drop_features: List of features to drop

        Returns:
            Tuple of (X, y, feature_names)
        """
        if self.data is None:
            self.load_data()

        # Define feature columns (exclude non-feature columns)
        exclude_cols = ['date', 'y', 'year', 'quarter', 'week_of_year']

        # Get all potential features
        all_features = [col for col in self.data.columns if col not in exclude_cols]

        # Remove features to drop
        if drop_features:
            all_features = [f for f in all_features if f not in drop_features]

        # Use subset if specified
        if feature_subset:
            feature_cols = [f for f in feature_subset if f in all_features]
        else:
            feature_cols = all_features

        # Select features
        X = self.data[feature_cols].copy()
        y = self.data[self.target_column].copy()

        # Handle any missing values
        X = X.fillna(0)
        y = y.fillna(y.mean())

        self.feature_columns = feature_cols

        logger.info(f"✓ Prepared {len(feature_cols)} features:")
        logger.info(f"  {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")

        return X, y, feature_cols

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        feature_subset: Optional[list] = None,
        drop_features: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list]:
        """
        Split data into train and test sets

        Args:
            test_size: Proportion of data for testing
            feature_subset: List of specific features to use
            drop_features: List of features to drop

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        X, y, feature_names = self.prepare_features_and_target(feature_subset, drop_features)

        # Time-series split (not random!)
        split_idx = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(f"✓ Train set: {len(X_train)} samples ({X_train.index.min()} to {X_train.index.max()})")
        logger.info(f"✓ Test set: {len(X_test)} samples ({X_test.index.min()} to {X_test.index.max()})")

        return X_train, X_test, y_train, y_test, feature_names

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the dataset

        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            self.load_data()

        return {
            "total_records": len(self.data),
            "date_range": (self.data['date'].min().strftime('%Y-%m-%d'),
                         self.data['date'].max().strftime('%Y-%m-%d')),
            "target_mean": float(self.data['y'].mean()),
            "target_std": float(self.data['y'].std()),
            "target_min": float(self.data['y'].min()),
            "target_max": float(self.data['y'].max()),
            "features_count": len([col for col in self.data.columns if col not in ['date', 'y', 'year', 'quarter', 'week_of_year']]),
        }


def load_real_data_for_training(
    test_size: float = 0.2,
    feature_subset: Optional[list] = None,
    drop_features: Optional[list] = None,
    category: str = "total_retail_sales",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list]:
    """
    Convenience function to load real retail data

    Args:
        test_size: Proportion of data for testing
        feature_subset: List of specific features to use
        drop_features: List of features to drop
        category: Retail category to load (default: total_retail_sales)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)

    Example:
        >>> X_train, X_test, y_train, y_test, features = load_real_data_for_training()
        >>> print(f"Training with {len(features)} features")
    """
    loader = RetailDataLoader(category=category)
    return loader.get_train_test_split(test_size, feature_subset, drop_features)


def get_available_categories() -> List[str]:
    """
    Get list of available retail categories

    Returns:
        List of category keys
    """
    return list(RETAIL_CATEGORIES.keys())


def get_category_display_name(category: str) -> str:
    """
    Get display name for a category

    Args:
        category: Category key

    Returns:
        Display name
    """
    return CATEGORY_DISPLAY_NAMES.get(category, category)


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=" * 60)
    print("Testing Retail Data Loader")
    print("=" * 60)

    loader = RetailDataLoader()
    summary = loader.get_data_summary()

    print("\nDataset Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nLoading train/test split...")
    X_train, X_test, y_train, y_test, features = loader.get_train_test_split(test_size=0.2)

    print(f"\nFeature set ({len(features)} features):")
    for i, feat in enumerate(features, 1):
        print(f"  {i:2d}. {feat}")

    print("\n" + "=" * 60)

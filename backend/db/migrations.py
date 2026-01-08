"""
RetailPRED Database Migration Script
Applies schema extensions to the existing database without losing data
"""

import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class RetailPREDMigration:
    """Handles database migrations for RetailPRED"""

    def __init__(self, db_path: str, schema_path: Optional[str] = None):
        """
        Initialize migration handler

        Args:
            db_path: Path to the SQLite database file
            schema_path: Optional path to schema.sql file (defaults to data/db/schema.sql)
        """
        self.db_path = db_path

        # Default schema path relative to project root
        if schema_path is None:
            project_root = Path(__file__).parent.parent.parent
            schema_path = project_root / "data" / "db" / "schema.sql"

        self.schema_path = Path(schema_path)

    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the existing database before migration

        Args:
            backup_path: Optional custom backup path

        Returns:
            Path to the backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.backup_{timestamp}"

        print(f"Creating backup at: {backup_path}")

        # Read the original database
        conn = sqlite3.connect(self.db_path)
        backup = sqlite3.connect(backup_path)

        # Backup the database
        with backup:
            conn.backup(backup)

        conn.close()
        backup.close()

        print(f"✓ Backup created successfully")
        return backup_path

    def check_existing_tables(self) -> list[str]:
        """
        Check what tables currently exist in the database

        Returns:
            List of existing table names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        conn.close()
        return tables

    def apply_schema(self, schema_content: str) -> None:
        """
        Apply the schema SQL to the database

        Args:
            schema_content: SQL schema content to apply
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Use executescript to handle the entire schema at once
            # This properly handles CREATE TABLE with IF NOT EXISTS
            cursor.executescript(schema_content)
            conn.commit()
            print("✓ Schema applied successfully")

        except Exception as e:
            conn.rollback()
            raise Exception(f"Error applying schema: {e}")
        finally:
            conn.close()

    def verify_schema(self) -> dict[str, bool]:
        """
        Verify that the new tables were created successfully

        Returns:
            Dictionary with table names and existence status
        """
        expected_tables = ["prediction_log", "model_metadata"]
        existing_tables = self.check_existing_tables()

        verification = {}
        for table in expected_tables:
            verification[table] = table in existing_tables
            status = "✓" if verification[table] else "✗"
            print(f"{status} Table '{table}': {'exists' if verification[table] else 'missing'}")

        # Check for indexes
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        expected_indexes = [
            "idx_prediction_log_model_name",
            "idx_prediction_log_prediction_date",
            "idx_prediction_log_store_id",
            "idx_prediction_log_product_id",
            "idx_prediction_log_created_at",
            "idx_model_metadata_model_name",
            "idx_model_metadata_is_active",
        ]

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
        existing_indexes = [row[0] for row in cursor.fetchall()]

        print("\nIndexes:")
        for index in expected_indexes:
            exists = index in existing_indexes
            print(f"  {'✓' if exists else '✗'} Index '{index}'")

        # Check for views
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name NOT LIKE 'sqlite_%'"
        )
        existing_views = [row[0] for row in cursor.fetchall()]

        expected_views = ["active_models", "prediction_accuracy"]

        print("\nViews:")
        for view in expected_views:
            exists = view in existing_views
            print(f"  {'✓' if exists else '✗'} View '{view}'")

        conn.close()

        return verification

    def get_table_info(self, table_name: str) -> None:
        """
        Print detailed information about a table

        Args:
            table_name: Name of the table to inspect
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        print(f"\n--- Table: {table_name} ---")
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        print("Columns:")
        for col in columns:
            col_id, name, type_, notnull, default, pk = col
            pk_str = " (PK)" if pk else ""
            null_str = " NOT NULL" if notnull else ""
            default_str = f" DEFAULT {default}" if default else ""
            print(f"  - {name}: {type_}{null_str}{default_str}{pk_str}")

        conn.close()

    def run_migration(
        self, create_backup: bool = True, verify: bool = True
    ) -> dict[str, bool]:
        """
        Run the complete migration process

        Args:
            create_backup: Whether to create a backup before migration
            verify: Whether to verify the schema after migration

        Returns:
            Dictionary with verification results
        """
        print("=" * 60)
        print("RetailPRED Database Migration")
        print("=" * 60)
        print(f"Database: {self.db_path}")
        print(f"Schema: {self.schema_path}")
        print()

        # Check if database exists
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        # Check existing tables
        print("Step 1: Checking existing database structure")
        print("-" * 40)
        existing_tables = self.check_existing_tables()
        print(f"Existing tables: {', '.join(existing_tables) if existing_tables else 'None'}")
        print()

        # Create backup
        if create_backup:
            print("Step 2: Creating database backup")
            print("-" * 40)
            backup_path = self.backup_database()
            print(f"✓ Backup saved to: {backup_path}")
            print()

        # Read schema file
        print("Step 3: Reading schema file")
        print("-" * 40)
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, "r") as f:
            schema_content = f.read()

        print(f"✓ Schema loaded from: {self.schema_path}")
        print()

        # Apply schema
        print("Step 4: Applying schema to database")
        print("-" * 40)
        self.apply_schema(schema_content)
        print()

        # Verify schema
        if verify:
            print("Step 5: Verifying migration")
            print("-" * 40)
            verification = self.verify_schema()
            print()

            # Show table details
            for table in ["prediction_log", "model_metadata"]:
                if table in existing_tables or verification.get(table, False):
                    self.get_table_info(table)

            print("\n" + "=" * 60)
            if all(verification.values()):
                print("✓ Migration completed successfully!")
            else:
                print("⚠ Migration completed with warnings")
            print("=" * 60)

            return verification

        return {}

    def rollback(self, backup_path: str) -> None:
        """
        Rollback to a backup database

        Args:
            backup_path: Path to the backup database file
        """
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        print(f"Restoring database from: {backup_path}")

        # Replace current database with backup
        import shutil

        shutil.copy2(backup_path, self.db_path)

        print("✓ Database restored successfully")


def main():
    """Main entry point for running migrations"""
    import argparse

    parser = argparse.ArgumentParser(description="RetailPRED Database Migration Tool")
    parser.add_argument(
        "--db-path",
        default="data/retailpred.db",
        help="Path to the SQLite database (default: data/retailpred.db)",
    )
    parser.add_argument(
        "--schema-path",
        default=None,
        help="Path to the schema.sql file (default: data/db/schema.sql)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup (not recommended)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after migration",
    )
    parser.add_argument(
        "--rollback",
        metavar="BACKUP_PATH",
        help="Rollback to a backup database",
    )

    args = parser.parse_args()

    try:
        migration = RetailPREDMigration(args.db_path, args.schema_path)

        if args.rollback:
            migration.rollback(args.rollback)
        else:
            migration.run_migration(
                create_backup=not args.no_backup, verify=not args.no_verify
            )

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

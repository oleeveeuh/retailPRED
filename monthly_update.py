#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monthly Data Update Script for Retail Market Dynamics Project
Automates data refresh, feature engineering, and model retraining
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import subprocess
import requests
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monthly_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_if_update_needed(data_path='data/processed/features.csv', months_old=1):
    """
    Check if data update is needed by comparing dates.
    
    Args:
        data_path (str): Path to features.csv
        months_old (int): Number of months old to trigger update
        
    Returns:
        bool: True if update is needed
    """
    logger.info("="*60)
    logger.info("Checking if update is needed...")
    logger.info("="*60)
    
    try:
        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}. Update needed.")
            return True
        
        df = pd.read_csv(data_path)
        
        if df.empty or 'date' not in df.columns:
            logger.warning("Data file is empty or missing date column. Update needed.")
            return True
        
        # Get last date
        df['date'] = pd.to_datetime(df['date'])
        last_date = df['date'].max()
        current_date = datetime.now()
        
        # Calculate months difference
        months_diff = (current_date.year - last_date.year) * 12 + \
                     (current_date.month - last_date.month)
        
        logger.info(f"Last data date: {last_date.strftime('%Y-%m-%d')}")
        logger.info(f"Current date: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"Months since last update: {months_diff}")
        
        if months_diff >= months_old:
            logger.info("✓ Update needed - data is older than threshold")
            return True
        else:
            logger.info(f"✗ No update needed - data is fresh (only {months_diff} months old)")
            return False
            
    except Exception as e:
        logger.error(f"Error checking update status: {e}")
        return True  # Default to updating if check fails


def run_script(script_name, description):
    """
    Run a Python script and capture output.
    
    Args:
        script_name (str): Name of script to run
        description (str): Description of what script does
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"{'='*60}")
    
    try:
        # Check if script exists
        if not os.path.exists(script_name):
            logger.warning(f"Script not found: {script_name}. Skipping.")
            return False
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✓ {description} completed successfully")
            if result.stdout:
                logger.info(f"Output:\n{result.stdout}")
            return True
        else:
            logger.error(f"✗ {description} failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} timed out (>10 minutes)")
        return False
    except Exception as e:
        logger.error(f"✗ Error running {description}: {e}")
        return False


def run_data_pipeline():
    """
    Run the complete data pipeline: ingestion -> cleaning -> feature engineering.
    
    Returns:
        bool: True if pipeline completed successfully
    """
    logger.info("\n" + "="*60)
    logger.info("RUNNING DATA PIPELINE")
    logger.info("="*60)
    
    success_count = 0
    total_steps = 3
    
    # Step 1: Data Ingestion
    if run_script('data_ingestion.py', 'Data Ingestion'):
        success_count += 1
    
    # Step 2: Data Cleaning
    if run_script('data_cleaning.py', 'Data Cleaning'):
        success_count += 1
    
    # Step 3: Feature Engineering
    if run_script('feature_engineering.py', 'Feature Engineering'):
        success_count += 1
    
    logger.info(f"\nPipeline Summary: {success_count}/{total_steps} steps completed")
    
    return success_count == total_steps


def merge_new_data(old_path='data/processed/features.csv', 
                   new_path='data/processed/features_new.csv',
                   output_path='data/processed/features.csv'):
    """
    Merge new data with existing features.csv.
    
    Args:
        old_path (str): Path to existing features.csv
        new_path (str): Path to newly generated features.csv
        output_path (str): Path to save merged data
        
    Returns:
        bool: True if merge successful
    """
    logger.info("\n" + "="*60)
    logger.info("Merging New Data")
    logger.info("="*60)
    
    try:
        # Load old data
        if os.path.exists(old_path):
            old_df = pd.read_csv(old_path)
            old_df['date'] = pd.to_datetime(old_df['date'])
            logger.info(f"✓ Loaded existing data: {len(old_df)} records")
        else:
            logger.info("No existing data found. Using new data only.")
            old_df = pd.DataFrame()
        
        # Load new data
        if not os.path.exists(new_path):
            logger.error(f"New data file not found: {new_path}")
            return False
        
        new_df = pd.read_csv(new_path)
        new_df['date'] = pd.to_datetime(new_df['date'])
        logger.info(f"✓ Loaded new data: {len(new_df)} records")
        
        # Merge data
        if old_df.empty:
            combined_df = new_df
        else:
            # Combine and remove duplicates
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        # Save merged data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        # Calculate statistics
        old_count = len(old_df) if not old_df.empty else 0
        new_count = len(combined_df) - old_count
        
        logger.info(f"✓ Merged data saved to: {output_path}")
        logger.info(f"  Total records: {len(combined_df)}")
        logger.info(f"  New records added: {new_count}")
        logger.info(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error merging data: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_if_retrain_needed(data_path='data/processed/features.csv'):
    """
    Check if model retraining is needed.
    
    Args:
        data_path (str): Path to features.csv
        
    Returns:
        bool: True if retraining needed
    """
    logger.info("\n" + "="*60)
    logger.info("Checking if model retraining is needed...")
    logger.info("="*60)
    
    try:
        if not os.path.exists(data_path):
            logger.info("✓ Retraining needed - no existing data")
            return True
        
        df = pd.read_csv(data_path)
        
        if df.empty:
            logger.info("✓ Retraining needed - data is empty")
            return True
        
        # Check if models exist
        model_files = ['models/linear_regression.joblib', 
                       'models/random_forest.joblib',
                       'models/prophet.joblib']
        
        models_exist = any(os.path.exists(f) for f in model_files)
        
        if not models_exist:
            logger.info("✓ Retraining needed - models don't exist")
            return True
        
        # Compare last model update time with data update time
        data_mtime = os.path.getmtime(data_path)
        model_mtime = min([os.path.getmtime(f) for f in model_files if os.path.exists(f)])
        
        if data_mtime > model_mtime:
            logger.info("✓ Retraining needed - data is newer than models")
            return True
        else:
            logger.info("✗ Retraining not needed - models are up to date")
            return False
            
    except Exception as e:
        logger.error(f"Error checking retrain status: {e}")
        return True  # Default to retraining if check fails


def retrain_models():
    """
    Retrain all models.
    
    Returns:
        bool: True if retraining successful
    """
    logger.info("\n" + "="*60)
    logger.info("Retraining Models")
    logger.info("="*60)
    
    return run_script('modeling.py', 'Model Retraining')


def send_notification(message, webhook_url=None, email=None, status='success'):
    """
    Send notification (placeholder for Slack/Email).
    
    Args:
        message (str): Message to send
        webhook_url (str): Slack webhook URL
        email (str): Email address
        status (str): 'success' or 'failure'
        
    Returns:
        bool: True if notification sent successfully
    """
    logger.info("\n" + "="*60)
    logger.info("Sending Notification")
    logger.info("="*60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emoji = "✅" if status == 'success' else "❌"
    
    notification = f"""
    {emoji} Retail Market Dynamics - Monthly Update {status.capitalize()}
    
    Timestamp: {timestamp}
    
    {message}
    """
    
    # Slack webhook notification (placeholder)
    if webhook_url:
        try:
            slack_payload = {
                "text": f"Retail Market Dynamics Update {status.capitalize()}",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{emoji} Monthly Update {status.capitalize()}"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Timestamp:* {timestamp}\n\n*Message:*\n{message}"
                        }
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=slack_payload)
            if response.status_code == 200:
                logger.info("✓ Slack notification sent")
                return True
            else:
                logger.error(f"✗ Failed to send Slack notification: {response.status_code}")
        except Exception as e:
            logger.error(f"✗ Error sending Slack notification: {e}")
    
    # Email notification (placeholder)
    if email:
        try:
            # This is a placeholder - implement actual email sending
            logger.info(f"Email notification would be sent to: {email}")
            logger.info(f"Message:\n{notification}")
            logger.info("✓ Email notification placeholder executed")
            return True
        except Exception as e:
            logger.error(f"✗ Error sending email: {e}")
    
    # Log notification to console/file
    logger.info(f"Notification:\n{notification}")
    logger.info("✓ Notification logged")
    
    return True


def generate_summary_report():
    """
    Generate a summary report of the update process.
    
    Returns:
        str: Summary report
    """
    logger.info("\n" + "="*60)
    logger.info("Generating Summary Report")
    logger.info("="*60)
    
    try:
        # Load features data
        if os.path.exists('data/processed/features.csv'):
            df = pd.read_csv('data/processed/features.csv')
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
                report = f"""
================================================================================
RETAIL MARKET DYNAMICS - MONTHLY UPDATE SUMMARY
================================================================================

Update completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SUMMARY:
  Total records: {len(df)}
  Date range: {df['date'].min()} to {df['date'].max()}
  
KEY METRICS:
"""
                
                # Add key metrics if available
                if 'retail_sales' in df.columns:
                    report += f"  Avg Retail Sales: ${df['retail_sales'].mean():,.0f}\n"
                
                if 'retail_growth' in df.columns:
                    report += f"  Avg Retail Growth: {df['retail_growth'].mean():.2f}%\n"
                
                if 'sp500_close' in df.columns:
                    report += f"  Avg S&P 500 Close: ${df['sp500_close'].mean():.2f}\n"
                
                report += "\n================================================================================\n"
                
                logger.info(report)
                return report
        
        return "No data available for summary"
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"Error generating summary: {e}"


def main():
    """Main update function."""
    logger.info("\n" + "="*60)
    logger.info("RETAIL MARKET DYNAMICS - MONTHLY UPDATE")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    update_status = 'failure'
    update_message = "Update process failed"
    
    try:
        # Step 1: Check if update is needed
        if not check_if_update_needed():
            logger.info("\n✓ No update needed. Process complete.")
            send_notification(
                "No new data available. Data is up to date.",
                status='success'
            )
            return
        
        # Step 2: Run data pipeline
        if not run_data_pipeline():
            logger.error("\n✗ Data pipeline failed")
            send_notification(
                "Data pipeline failed. Please check logs.",
                status='failure'
            )
            return
        
        # Step 3: Merge new data with existing
        # Note: We need to temporarily rename the new features.csv
        # Since running the scripts overwrites the file
        if os.path.exists('data/processed/features.csv'):
            os.rename('data/processed/features.csv', 'data/processed/features_new.csv')
        
        # The feature engineering script creates the new file
        # Now merge
        merge_new_data()
        
        # Remove temporary file
        if os.path.exists('data/processed/features_new.csv'):
            os.remove('data/processed/features_new.csv')
        
        # Step 4: Check if retraining is needed and retrain if needed
        if check_if_retrain_needed():
            logger.info("\n✓ Retraining models...")
            if retrain_models():
                logger.info("✓ Model retraining completed successfully")
            else:
                logger.warning("⚠ Model retraining failed, but update continues")
        else:
            logger.info("✓ Models are up to date, skipping retraining")
        
        # Step 5: Generate summary
        summary = generate_summary_report()
        
        # Step 6: Send notification
        update_status = 'success'
        update_message = f"Monthly update completed successfully.\n\n{summary}"
        
        send_notification(update_message, status='success')
        
        logger.info("\n✓ Monthly update completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n✗ Update failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        send_notification(
            f"Update failed with error: {str(e)}\n\nCheck logs for details.",
            status='failure'
        )
    
    finally:
        logger.info(f"\nUpdate process finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)


if __name__ == "__main__":
    main()


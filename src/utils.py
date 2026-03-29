import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPECTED_FEATURES = [
    "CustomerID", "Recency", "Frequency", "MonetaryTotal", "MonetaryAvg", "MonetaryStd",
    "MonetaryMin", "MonetaryMax", "TotalQuantity", "AvgQuantityPerTransaction", "MinQuantity",
    "MaxQuantity", "CustomerTenureDays", "FirstPurchaseDaysAgo", "PreferredDayOfWeek", "PreferredHour",
    "PreferredMonth", "WeekendPurchaseRatio", "AvgDaysBetweenPurchases", "UniqueProducts", "UniqueDescriptions",
    "AvgProductsPerTransaction", "Country", "UniqueCountries", "NegativeQuantityCount", "ZeroPriceCount", "CancelledTransactions",
    "ReturnRatio", "TotalTransactions", "UniqueInvoices", "AvgLinesPerInvoice", "Age", "Gender", "RegistrationDate", "NewsletterSubscribed",
    "LastLoginIP", "SupportTicketsCount", "SatisfactionScore", "AccountStatus", "Churn"
]

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file with error handling.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Missing raw data file at {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path} with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise

def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the dataframe contains the 52 expected features.
    """
    missing_features = [f for f in EXPECTED_FEATURES if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Schema validation failed. Missing features: {missing_features}")
        return False
    
    logger.info("Schema validation successful. All 52 features are present.")
    return True

def generate_data_report(df: pd.DataFrame):
    """
    Generate initial statistics and info report.
    """
    logger.info("--- Data Quality Report ---")
    logger.info(f"Shape: {df.shape}")
    
    # Detailed summary statistics
    desc = df.describe(include='all')
    
    # Save to reports if possible
    report_path = "reports/initial_stats.csv"
    try:
        desc.to_csv(report_path)
        logger.info(f"Initial statistics saved to {report_path}")
    except Exception as e:
        logger.error(f"Could not save report: {e}")

def detect_missing_values(df: pd.DataFrame):
    """
    Detect missing values and calculate percentages.
    """
    missing = df.isnull().sum()
    perc = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing': missing, 'Percentage': perc})
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values(by='Percentage', ascending=False)
    
    logger.info(f"Missing values detected in {len(missing_df)} columns.")
    return missing_df

def identify_outliers(df: pd.DataFrame, columns: list = None, method: str = 'iqr'):
    """
    Identify outliers and calculate skewness for numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
        
    outliers_report = {}
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            # Handle NaNs before calculation
            col_data = df[col].dropna()
            if col_data.empty:
                continue
                
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            else: # Z-score
                z_scores = (col_data - col_data.mean()) / col_data.std()
                count = (abs(z_scores) > 3).sum()
            
            skew = col_data.skew()
            outliers_report[col] = {
                'outlier_count': int(count),
                'skewness': float(skew),
                'is_highly_skewed': abs(skew) > 1
            }
            
    return outliers_report

def check_duplicates(df: pd.DataFrame, subset: list = None):
    """
    Check for duplicate rows.
    """
    duplicates_count = df.duplicated(subset=subset).sum()
    logger.info(f"Found {duplicates_count} duplicate rows.")
    return duplicates_count

def generate_quality_html_report(df: pd.DataFrame):
    """
    Generate a simple HTML quality report with outliers and skewness.
    """
    missing_df = detect_missing_values(df)
    outliers_report = identify_outliers(df)
    duplicates = check_duplicates(df, subset=['CustomerID'])
    
    # Format outliers table
    outliers_html = ""
    for col, stats in outliers_report.items():
        if stats['outlier_count'] > 0 or stats['is_highly_skewed']:
            row_style = 'style="background-color: #ffcccc;"' if stats['is_highly_skewed'] else ""
            outliers_html += f"""
            <tr {row_style}>
                <td>{col}</td>
                <td>{stats['outlier_count']}</td>
                <td>{stats['skewness']:.2f}</td>
                <td>{'Yes' if stats['is_highly_skewed'] else 'No'}</td>
            </tr>"""

    html_content = f"""
    <html>
    <head>
        <title>Professional Data Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            .summary {{ display: flex; gap: 20px; }}
            .card {{ border: 1px solid #ddd; padding: 20px; border-radius: 8px; flex: 1; background: #fdfdfd; }}
            .card h3 {{ margin-top: 0; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <h1>Retail Customers: Data Quality & Structure Analysis</h1>
        
        <div class="summary">
            <div class="card">
                <h3>Overview</h3>
                <p><b>Total Rows:</b> {len(df)}</p>
                <p><b>Features:</b> {len(df.columns)}</p>
            </div>
            <div class="card">
                <h3>Integrity</h3>
                <p><b>Duplicate IDs:</b> {duplicates}</p>
                <p><b>Missing Cells:</b> {df.isnull().sum().sum()}</p>
            </div>
        </div>
        
        <h2>Missing Values Detail</h2>
        {missing_df.to_html() if not missing_df.empty else "<p>No missing values found.</p>"}
        
        <h2>Outliers & Skewness Analysis</h2>
        <table>
            <tr>
                <th>Column</th>
                <th>Outliers (IQR)</th>
                <th>Skewness</th>
                <th>Highly Skewed?</th>
            </tr>
            {outliers_html}
        </table>
        
        <div style="margin-top: 30px; font-size: 0.8em; color: #95a5a6;">
            Report generated on: {pd.Timestamp.now()}
        </div>
    </body>
    </html>
    """
    
    report_path = "reports/data_quality_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    logger.info(f"Quality HTML report saved to {report_path}")

def generate_profiling_report(df: pd.DataFrame):
    """
    Generate an automated EDA report using ydata-profiling.
    """
    try:
        from ydata_profiling import ProfileReport
        
        logger.info("Generating profiling report. This may take a while...")
        profile = ProfileReport(df, title="Retail Customers Data Profiling Report", explorative=True)
        report_path = "reports/profile_report.html"
        profile.to_file(report_path)
        logger.info(f"Profiling report saved to {report_path}")
    except ImportError:
        logger.error("ydata-profiling not installed. Please install it with 'pip install ydata-profiling'")
    except Exception as e:
        logger.error(f"Error generating profiling report: {e}")

if __name__ == "__main__":
    logger.info("Utils script initialized.")
    # Example usage (commented out as raw data may not exist yet)
    # raw_path = "data/raw/data.csv"
    # if os.path.exists(raw_path):
    #     data = load_raw_data(raw_path)
    #     validate_schema(data)
    #     generate_data_report(data)

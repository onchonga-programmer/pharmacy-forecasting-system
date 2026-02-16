"""
Data Cleaning and Preprocessing Pipeline
=========================================
Handles data quality issues identified in EDA:
- Outlier detection and treatment
- Data transformations (log, standardization)
- Incomplete period handling
- Data validation

"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy import stats
import logging
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from config.database_config import DATABASE_URL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(project_root) / 'logs' / 'data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Main class for data cleaning and preprocessing operations"""
    
    def __init__(self, outlier_method='iqr', outlier_action='cap', log_transform=True):
        """
        Initialize DataCleaner
        
        Parameters:
        -----------
        outlier_method : str
            Method for outlier detection: 'iqr', 'zscore', or 'percentile'
        outlier_action : str
            Action for outliers: 'cap', 'remove', or 'keep'
        log_transform : bool
            Whether to apply log transformation to skewed data
        """
        self.outlier_method = outlier_method
        self.outlier_action = outlier_action
        self.log_transform = log_transform
        self.cleaning_stats = {}
        
    def load_data(self):
        """Load raw sales data from database"""
        logger.info("Loading data from database...")
        
        try:
            engine = create_engine(DATABASE_URL)
            
            query = """
            SELECT 
                sw.year,
                sw.week_number,
                sw.week_start_date,
                sw.week_end_date,
                sw.total_quantity,
                sw.avg_daily_quantity,
                d.drug_name,
                d.atc_code,
                d.drug_id
            FROM raw_data.sales_weekly_raw sw
            JOIN raw_data.drugs_raw d ON sw.drug_id = d.drug_id
            ORDER BY sw.week_start_date, d.drug_name
            """
            
            df = pd.read_sql_query(query, engine)
            
            # Convert date columns
            df['week_start_date'] = pd.to_datetime(df['week_start_date'])
            df['week_end_date'] = pd.to_datetime(df['week_end_date'])
            
            logger.info(f"✓ Loaded {len(df):,} records from {df['week_start_date'].min().date()} to {df['week_start_date'].max().date()}")
            
            self.cleaning_stats['initial_records'] = len(df)
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def detect_outliers_iqr(self, df, column='total_quantity', multiplier=1.5):
        """
        Detect outliers using IQR method
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        column : str
            Column to check for outliers
        multiplier : float
            IQR multiplier (1.5 for moderate, 3.0 for extreme)
        
        Returns:
        --------
        Series : Boolean mask for outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        logger.info(f"IQR Method: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}")
        logger.info(f"Found {outliers.sum():,} outliers ({outliers.sum()/len(df)*100:.2f}%)")
        
        return outliers, lower_bound, upper_bound
    
    def detect_outliers_zscore(self, df, column='total_quantity', threshold=3):
        """
        Detect outliers using Z-score method
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        column : str
            Column to check for outliers
        threshold : float
            Z-score threshold (typically 3)
        
        Returns:
        --------
        Series : Boolean mask for outliers
        """
        z_scores = np.abs(stats.zscore(df[column]))
        outliers = z_scores > threshold
        
        logger.info(f"Z-Score Method: Threshold={threshold}")
        logger.info(f"Found {outliers.sum():,} outliers ({outliers.sum()/len(df)*100:.2f}%)")
        
        return outliers, df[column].mean() - threshold * df[column].std(), df[column].mean() + threshold * df[column].std()
    
    def detect_outliers_percentile(self, df, column='total_quantity', lower=1, upper=99):
        """
        Detect outliers using percentile method
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        column : str
            Column to check for outliers
        lower : float
            Lower percentile (e.g., 1)
        upper : float
            Upper percentile (e.g., 99)
        
        Returns:
        --------
        Series : Boolean mask for outliers
        """
        lower_bound = df[column].quantile(lower / 100)
        upper_bound = df[column].quantile(upper / 100)
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        logger.info(f"Percentile Method: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}")
        logger.info(f"Found {outliers.sum():,} outliers ({outliers.sum()/len(df)*100:.2f}%)")
        
        return outliers, lower_bound, upper_bound
    
    def handle_outliers(self, df, column='total_quantity'):
        """
        Handle outliers based on configured method and action
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        column : str
            Column to process
        
        Returns:
        --------
        DataFrame : Processed dataframe
        """
        logger.info(f"\n{'='*60}")
        logger.info("OUTLIER DETECTION AND TREATMENT")
        logger.info(f"{'='*60}")
        logger.info(f"Method: {self.outlier_method}, Action: {self.outlier_action}")
        
        df = df.copy()
        
        # Detect outliers
        if self.outlier_method == 'iqr':
            outliers, lower_bound, upper_bound = self.detect_outliers_iqr(df, column)
        elif self.outlier_method == 'zscore':
            outliers, lower_bound, upper_bound = self.detect_outliers_zscore(df, column)
        elif self.outlier_method == 'percentile':
            outliers, lower_bound, upper_bound = self.detect_outliers_percentile(df, column)
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")
        
        self.cleaning_stats['outliers_detected'] = outliers.sum()
        self.cleaning_stats['outliers_percentage'] = outliers.sum() / len(df) * 100
        
        # Handle outliers
        if self.outlier_action == 'remove':
            df = df[~outliers]
            logger.info(f"✓ Removed {outliers.sum():,} outlier records")
            self.cleaning_stats['outliers_removed'] = outliers.sum()
            
        elif self.outlier_action == 'cap':
            original_outliers = df.loc[outliers, column].copy()
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound
            logger.info(f"✓ Capped {outliers.sum():,} outlier values")
            self.cleaning_stats['outliers_capped'] = outliers.sum()
            
        elif self.outlier_action == 'keep':
            logger.info("✓ Keeping outliers (no action)")
            self.cleaning_stats['outliers_kept'] = outliers.sum()
        else:
            raise ValueError(f"Unknown outlier action: {self.outlier_action}")
        
        return df
    
    def handle_incomplete_periods(self, df):
        """
        Handle incomplete time periods (e.g., partial year data)
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        
        Returns:
        --------
        DataFrame : Processed dataframe
        """
        logger.info(f"\n{'='*60}")
        logger.info("INCOMPLETE PERIOD HANDLING")
        logger.info(f"{'='*60}")
        
        df = df.copy()
        
        # Check for incomplete years
        year_week_counts = df.groupby('year')['week_number'].nunique()
        
        logger.info("Week counts by year:")
        for year, count in year_week_counts.items():
            status = "✓ Complete" if count >= 52 else "⚠ Incomplete"
            logger.info(f"  {year}: {count} weeks - {status}")
        
        # Flag incomplete years (< 52 weeks or last year in dataset)
        incomplete_years = year_week_counts[year_week_counts < 52].index.tolist()
        last_year = df['year'].max()
        
        if last_year not in incomplete_years:
            # Check if last year has data for full 12 months
            last_year_months = df[df['year'] == last_year]['week_start_date'].dt.month.nunique()
            if last_year_months < 12:
                incomplete_years.append(last_year)
                logger.info(f"  {last_year}: Only {last_year_months} months of data - marked incomplete")
        
        df['is_complete_year'] = ~df['year'].isin(incomplete_years)
        
        self.cleaning_stats['incomplete_years'] = incomplete_years
        self.cleaning_stats['records_in_incomplete_years'] = df[~df['is_complete_year']].shape[0]
        
        logger.info(f"\n✓ Flagged {len(incomplete_years)} incomplete year(s): {incomplete_years}")
        logger.info(f"✓ {self.cleaning_stats['records_in_incomplete_years']:,} records in incomplete periods")
        
        return df
    
    def apply_transformations(self, df, column='total_quantity'):
        """
        Apply data transformations to handle skewness
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        column : str
            Column to transform
        
        Returns:
        --------
        DataFrame : Transformed dataframe
        """
        logger.info(f"\n{'='*60}")
        logger.info("DATA TRANSFORMATIONS")
        logger.info(f"{'='*60}")
        
        df = df.copy()
        
        # Check skewness
        skewness = df[column].skew()
        logger.info(f"Original skewness: {skewness:.2f}")
        
        if self.log_transform and skewness > 1:
            # Log transformation (log1p to handle zeros)
            df[f'{column}_log'] = np.log1p(df[column])
            new_skewness = df[f'{column}_log'].skew()
            logger.info(f"✓ Applied log1p transformation")
            logger.info(f"New skewness: {new_skewness:.2f}")
            self.cleaning_stats['log_transform_applied'] = True
            self.cleaning_stats['skewness_before'] = skewness
            self.cleaning_stats['skewness_after'] = new_skewness
        else:
            logger.info("Skewness acceptable, no transformation needed")
            self.cleaning_stats['log_transform_applied'] = False
        
        # Add standardized version
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[f'{column}_scaled'] = scaler.fit_transform(df[[column]])
        logger.info(f"✓ Created standardized version")
        
        return df
    
    def validate_data(self, df):
        """
        Validate cleaned data quality
        
        Parameters:
        -----------
        df : DataFrame
            Cleaned dataframe
        
        Returns:
        --------
        dict : Validation results
        """
        logger.info(f"\n{'='*60}")
        logger.info("DATA VALIDATION")
        logger.info(f"{'='*60}")
        
        validation = {}
        
        # Missing values
        missing = df.isnull().sum().sum()
        validation['missing_values'] = missing
        logger.info(f"Missing values: {missing}")
        
        # Duplicates
        duplicates = df.duplicated(subset=['year', 'week_number', 'drug_id']).sum()
        validation['duplicates'] = duplicates
        logger.info(f"Duplicate records: {duplicates}")
        
        # Negative values
        negative = (df['total_quantity'] < 0).sum()
        validation['negative_values'] = negative
        logger.info(f"Negative values: {negative}")
        
        # Date range
        validation['date_range'] = {
            'start': df['week_start_date'].min(),
            'end': df['week_start_date'].max(),
            'total_days': (df['week_start_date'].max() - df['week_start_date'].min()).days
        }
        logger.info(f"Date range: {validation['date_range']['start'].date()} to {validation['date_range']['end'].date()}")
        
        # Data distribution
        validation['distribution'] = {
            'mean': df['total_quantity'].mean(),
            'median': df['total_quantity'].median(),
            'std': df['total_quantity'].std(),
            'skewness': df['total_quantity'].skew(),
            'kurtosis': df['total_quantity'].kurtosis()
        }
        
        logger.info(f"\nDistribution stats:")
        logger.info(f"  Mean: {validation['distribution']['mean']:.2f}")
        logger.info(f"  Median: {validation['distribution']['median']:.2f}")
        logger.info(f"  Std Dev: {validation['distribution']['std']:.2f}")
        logger.info(f"  Skewness: {validation['distribution']['skewness']:.2f}")
        
        # Quality score
        issues = missing + duplicates + negative
        quality_score = 100 - (issues / len(df) * 100)
        validation['quality_score'] = quality_score
        logger.info(f"\n✓ Data Quality Score: {quality_score:.2f}%")
        
        return validation
    
    def save_cleaned_data(self, df, output_path=None):
        """
        Save cleaned data to file
        
        Parameters:
        -----------
        df : DataFrame
            Cleaned dataframe
        output_path : str
            Output file path
        """
        if output_path is None:
            output_path = Path(project_root) / 'data' / 'processed' / 'sales_cleaned.csv'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved cleaned data to: {output_path}")
        logger.info(f"✓ Final record count: {len(df):,}")
        
        self.cleaning_stats['final_records'] = len(df)
        self.cleaning_stats['output_path'] = str(output_path)
        
        # Save cleaning statistics
        stats_path = output_path.parent / 'cleaning_stats.txt'
        with open(stats_path, 'w') as f:
            f.write("DATA CLEANING STATISTICS\n")
            f.write("="*60 + "\n\n")
            for key, value in self.cleaning_stats.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"✓ Saved cleaning statistics to: {stats_path}")
    
    def run_pipeline(self, save_output=True, output_path=None):
        """
        Run complete data cleaning pipeline
        
        Parameters:
        -----------
        save_output : bool
            Whether to save cleaned data
        output_path : str
            Output file path
        
        Returns:
        --------
        DataFrame : Cleaned dataframe
        dict : Validation results
        """
        logger.info("\n" + "="*60)
        logger.info("DATA CLEANING PIPELINE STARTED")
        logger.info("="*60 + "\n")
        
        # Load data
        df = self.load_data()
        
        # Handle outliers
        df = self.handle_outliers(df)
        
        # Handle incomplete periods
        df = self.handle_incomplete_periods(df)
        
        # Apply transformations
        df = self.apply_transformations(df)
        
        # Validate
        validation = self.validate_data(df)
        
        # Save
        if save_output:
            self.save_cleaned_data(df, output_path)
        
        logger.info("\n" + "="*60)
        logger.info("DATA CLEANING PIPELINE COMPLETED")
        logger.info("="*60 + "\n")
        
        return df, validation


def main():
    """Main execution function"""
    
    # Create cleaner with configuration
    cleaner = DataCleaner(
        outlier_method='iqr',      # Options: 'iqr', 'zscore', 'percentile'
        outlier_action='cap',       # Options: 'cap', 'remove', 'keep'
        log_transform=True          # Apply log transformation for skewed data
    )
    
    # Run pipeline
    df_cleaned, validation = cleaner.run_pipeline(save_output=True)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Initial records: {cleaner.cleaning_stats['initial_records']:,}")
    print(f"Final records: {cleaner.cleaning_stats['final_records']:,}")
    print(f"Records removed: {cleaner.cleaning_stats['initial_records'] - cleaner.cleaning_stats['final_records']:,}")
    print(f"Data quality score: {validation['quality_score']:.2f}%")
    print(f"Output saved to: {cleaner.cleaning_stats['output_path']}")
    print("="*60)
    
    return df_cleaned, validation


if __name__ == "__main__":
    df_cleaned, validation = main()

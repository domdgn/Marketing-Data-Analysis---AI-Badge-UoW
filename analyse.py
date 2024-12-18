import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_clean_data(file_path):
    """
    Load the marketing dataset and perform initial cleaning
    """
    df = pd.read_csv(file_path)
    
    # Convert percentage columns to float
    percentage_columns = ['Ad_Click_Rate', 'Conversion_Rate', 'Email_Open_Rate', 'Customer_Retention_Rate']
    df[percentage_columns] = df[percentage_columns].astype(float)
    
    # Handle any negative values in Social_Media_Followers
    df['Social_Media_Followers'] = df['Social_Media_Followers'].clip(lower=0)
    
    return df

def generate_numerical_stats(df):
    """
    Generate detailed statistics for numerical columns
    """
    numerical_cols = ['Campaign_Budget', 'Ad_Click_Rate', 'Conversion_Rate', 
                     'Social_Media_Followers', 'Email_Open_Rate', 'Customer_Retention_Rate']
    
    stats_df = df[numerical_cols].agg([
        'count', 'mean', 'median', 'std', 'min', 'max',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        'skew', 'kurtosis'
    ]).round(2)
    
    stats_df.index = ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 
                      '25th Percentile', '75th Percentile', 'Skewness', 'Kurtosis']
    
    return stats_df

def generate_categorical_stats(df):
    """
    Generate detailed statistics for categorical columns
    """
    categorical_cols = ['Platform', 'Campaign_Type', 'Target_Audience', 'Region']
    cat_stats = {}
    
    for col in categorical_cols:
        # Calculate value counts and percentages
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True).round(4) * 100
        
        # Combine counts and percentages
        stats = pd.DataFrame({
            'Count': counts,
            'Percentage': percentages
        })
        stats['Cumulative_%'] = stats['Percentage'].cumsum().round(2)
        cat_stats[col] = stats
    
    return cat_stats

def analyze_correlations(df):
    """
    Analyze correlations between numerical variables
    """
    numerical_cols = ['Campaign_Budget', 'Ad_Click_Rate', 'Conversion_Rate', 
                     'Social_Media_Followers', 'Email_Open_Rate', 'Customer_Retention_Rate']
    
    correlation_matrix = df[numerical_cols].corr().round(3)
    return correlation_matrix

def generate_campaign_performance_metrics(df):
    """
    Generate performance metrics by campaign type and platform
    """
    # Average metrics by Campaign Type
    campaign_metrics = df.groupby('Campaign_Type').agg({
        'Campaign_Budget': 'mean',
        'Ad_Click_Rate': 'mean',
        'Conversion_Rate': 'mean',
        'Email_Open_Rate': 'mean',
        'Customer_Retention_Rate': 'mean'
    }).round(2)
    
    # Average metrics by Platform
    platform_metrics = df.groupby('Platform').agg({
        'Campaign_Budget': 'mean',
        'Ad_Click_Rate': 'mean',
        'Conversion_Rate': 'mean',
        'Email_Open_Rate': 'mean',
        'Customer_Retention_Rate': 'mean'
    }).round(2)
    
    return campaign_metrics, platform_metrics

def analyze_marketing_data(file_path):
    """
    Main function to analyze marketing dataset
    """
    print("Loading and analyzing marketing data...\n")
    df = load_and_clean_data(file_path)
    
    # 1. Basic Dataset Information
    print("=== Dataset Overview ===")
    print(f"Total number of campaigns: {len(df)}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024:.2f} KB\n")
    
    # 2. Numerical Statistics
    print("=== Numerical Variables Statistics ===")
    numerical_stats = generate_numerical_stats(df)
    print(numerical_stats)
    print("\n")
    
    # 3. Categorical Statistics
    print("=== Categorical Variables Distribution ===")
    categorical_stats = generate_categorical_stats(df)
    for category, stats in categorical_stats.items():
        print(f"\n{category} Distribution:")
        print(stats)
    print("\n")
    
    # 4. Correlation Analysis
    print("=== Correlation Matrix ===")
    correlation_matrix = analyze_correlations(df)
    print(correlation_matrix)
    print("\n")
    
    # 5. Campaign Performance Metrics
    print("=== Campaign Performance Metrics ===")
    campaign_metrics, platform_metrics = generate_campaign_performance_metrics(df)
    
    print("\nAverage Metrics by Campaign Type:")
    print(campaign_metrics)
    
    print("\nAverage Metrics by Platform:")
    print(platform_metrics)
    
    # 6. Additional Insights
    print("\n=== Additional Insights ===")
    print(f"Most expensive campaign: ${df['Campaign_Budget'].max():,.2f}")
    print(f"Average campaign budget: ${df['Campaign_Budget'].mean():,.2f}")
    print(f"Most successful conversion rate: {df['Conversion_Rate'].max():.2f}%")
    print(f"Average conversion rate: {df['Conversion_Rate'].mean():.2f}%")
    
    return df

if __name__ == "__main__":
    # Clear the screen at the start
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the CSV file
    file_path = os.path.join(script_dir, 'data/Marketing_Design_Dataset.csv')
    
    try:
        df = analyze_marketing_data(file_path)
        print("\nAnalysis complete!")
    except FileNotFoundError:
        print(f"\nError: Could not find 'Marketing_Design_Dataset.csv' in {script_dir}")
        print("Please make sure the CSV file is in the same folder as this script.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    
    # Keep the window open
    input("\nPress Enter to exit...")
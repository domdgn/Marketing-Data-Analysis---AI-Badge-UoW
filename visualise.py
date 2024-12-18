import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_output_directory():
    """Create graphs directory if it doesn't exist"""
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

def load_and_prepare_data(file_path):
    """Load and prepare the marketing dataset"""
    df = pd.read_csv(file_path)
    return df

def create_visualizations(df):
    # Set the style to a default matplotlib style
    plt.style.use('default')
    # Set the color palette using seaborn
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Budget Distribution
    ax1 = plt.subplot(231)
    sns.histplot(data=df, x='Campaign_Budget', bins=30, ax=ax1)
    ax1.set_title('Campaign Budget Distribution')
    ax1.set_xlabel('Budget ($)')
    
    # 2. Platform Performance Comparison
    ax2 = plt.subplot(232)
    platform_metrics = df.groupby('Platform')[['Ad_Click_Rate', 'Conversion_Rate']].mean()
    platform_metrics.plot(kind='bar', ax=ax2)
    ax2.set_title('Platform Performance Comparison')
    ax2.set_xlabel('Platform')
    ax2.set_ylabel('Rate (%)')
    plt.xticks(rotation=45)
    
    # 3. Campaign Type Performance
    ax3 = plt.subplot(233)
    campaign_metrics = df.groupby('Campaign_Type')['Customer_Retention_Rate'].mean().sort_values()
    campaign_metrics.plot(kind='barh', ax=ax3)
    ax3.set_title('Customer Retention by Campaign Type')
    ax3.set_xlabel('Average Retention Rate (%)')
    
    # 4. Regional Performance Heatmap
    ax4 = plt.subplot(234)
    regional_metrics = df.groupby('Region')[['Ad_Click_Rate', 'Conversion_Rate', 'Email_Open_Rate']].mean()
    sns.heatmap(regional_metrics, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Regional Performance Heatmap')
    
    # 5. Audience Targeting Analysis
    ax5 = plt.subplot(235)
    audience_budget = df.groupby('Target_Audience')['Campaign_Budget'].mean().sort_values()
    audience_budget.plot(kind='barh', ax=ax5)
    ax5.set_title('Average Budget by Target Audience')
    ax5.set_xlabel('Average Budget ($)')
    
    # 6. Performance Metrics Distribution
    ax6 = plt.subplot(236)
    metrics = ['Ad_Click_Rate', 'Conversion_Rate', 'Email_Open_Rate']
    df[metrics].boxplot(ax=ax6)
    ax6.set_title('Performance Metrics Distribution')
    ax6.set_ylabel('Rate (%)')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('graphs/marketing_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation heatmap separately
    plt.figure(figsize=(10, 8))
    numerical_cols = ['Campaign_Budget', 'Ad_Click_Rate', 'Conversion_Rate', 
                     'Social_Media_Followers', 'Email_Open_Rate', 'Customer_Retention_Rate']
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Variables')
    plt.tight_layout()
    plt.savefig('graphs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        # Create graphs directory
        create_output_directory()
        
        # Load data
        df = load_and_prepare_data('data/Marketing_Design_Dataset.csv')
        
        # Create visualizations
        create_visualizations(df)
        
        print("Visualizations have been created successfully!")
        print("Check the 'graphs' folder for the visualization files.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    input("\nPress Enter to exit...")
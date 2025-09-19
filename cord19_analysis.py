# CORD-19 Data Analysis Assignment
# Student: Robinhood Waweru
# Date: 15/9/2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import re
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*70)
print("CORD-19 COVID-19 RESEARCH PAPERS DATA ANALYSIS")
print("="*70)

# ================================================================
# PART 1: DATA LOADING AND BASIC EXPLORATION
# ================================================================

print("\n" + "="*50)
print("PART 1: DATA LOADING AND BASIC EXPLORATION")
print("="*50)

def load_and_explore_data(file_path='metadata.csv'):
    """
    Load the CORD-19 dataset and perform basic exploration
    """
    try:
        print("📂 Loading CORD-19 metadata...")
        
        # For demonstration, we'll create a sample dataset if file doesn't exist
        # In practice, you would download from Kaggle
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully from {file_path}")
        except FileNotFoundError:
            print("⚠️ Creating sample dataset for demonstration...")
            df = create_sample_cord19_data()
        
        # Basic exploration
        print(f"\n📊 Dataset Overview:")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"Rows: {len(df)}")
        
        print("\n📋 First 5 rows:")
        print("-" * 60)
        print(df.head())
        
        print("\n📋 Dataset Info:")
        print("-" * 30)
        print(df.info())
        
        print("\n📋 Column Names:")
        print("-" * 20)
        for i, col in enumerate(df.columns):
            print(f"{i+1:2d}. {col}")
        
        print("\n📋 Missing Values Summary:")
        print("-" * 35)
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_summary = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent.round(2)
        }).sort_values('Missing Count', ascending=False)
        
        print(missing_summary[missing_summary['Missing Count'] > 0])
        
        print("\n📋 Basic Statistics for Numerical Columns:")
        print("-" * 45)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        else:
            print("No numerical columns found in basic dataset")
            
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_sample_cord19_data():
    """
    Create a sample CORD-19 dataset for demonstration
    This simulates the structure of the real metadata.csv file
    """
    np.random.seed(42)  # For reproducible results
    
    n_papers = 1000  # Sample size
    
    # Generate sample data
    journals = [
        'Nature', 'Science', 'Cell', 'The Lancet', 'NEJM', 'PLOS ONE',
        'BMJ', 'JAMA', 'Nature Medicine', 'Cell Host & Microbe',
        'Journal of Virology', 'Virology', 'Antiviral Research',
        'International Journal of Infectious Diseases', 'Clinical Infectious Diseases'
    ]
    
    sources = ['PMC', 'PubMed', 'WHO', 'bioRxiv', 'medRxiv', 'arXiv']
    
    # Generate dates between 2019-2023
    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime('2023-12-31')
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # COVID-related terms for titles
    covid_terms = [
        'COVID-19', 'SARS-CoV-2', 'coronavirus', 'pandemic', 'vaccine',
        'treatment', 'symptoms', 'diagnosis', 'prevention', 'immunity',
        'variant', 'transmission', 'epidemic', 'lockdown', 'mask'
    ]
    
    sample_data = {
        'cord_uid': [f'cord-{i:06d}' for i in range(n_papers)],
        'title': [f"Study of {np.random.choice(covid_terms)} in {np.random.choice(['adults', 'children', 'elderly', 'healthcare workers'])}" 
                 for _ in range(n_papers)],
        'abstract': [f"This study examines {np.random.choice(covid_terms)} and its impact on public health..." 
                    for _ in range(n_papers)],
        'publish_time': np.random.choice(date_range, n_papers),
        'authors': [f"Author{i} et al." for i in range(n_papers)],
        'journal': np.random.choice(journals, n_papers),
        'source_x': np.random.choice(sources, n_papers),
        'doi': [f"10.1000/sample.{i}" for i in range(n_papers)],
        'pmcid': [f"PMC{1000000+i}" if np.random.random() > 0.3 else None for i in range(n_papers)],
        'pubmed_id': [f"{20000000+i}" if np.random.random() > 0.4 else None for i in range(n_papers)],
        'license': np.random.choice(['CC BY', 'CC BY-NC', 'No License', None], n_papers),
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some missing values to simulate real data
    df.loc[np.random.choice(df.index, 100, replace=False), 'abstract'] = None
    df.loc[np.random.choice(df.index, 50, replace=False), 'journal'] = None
    
    return df

# ================================================================
# PART 2: DATA CLEANING AND PREPARATION
# ================================================================

def clean_and_prepare_data(df):
    """
    Clean the dataset and prepare it for analysis
    """
    print("\n" + "="*50)
    print("PART 2: DATA CLEANING AND PREPARATION")
    print("="*50)
    
    print("🧹 Starting data cleaning process...")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Convert publish_time to datetime
    print("\n📅 Converting dates...")
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['year'] = df_clean['publish_time'].dt.year
    df_clean['month'] = df_clean['publish_time'].dt.month
    df_clean['quarter'] = df_clean['publish_time'].dt.quarter
    
    # Handle missing values
    print("\n🔧 Handling missing values...")
    
    # Fill missing journals with 'Unknown'
    df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    
    # Fill missing abstracts
    df_clean['abstract'] = df_clean['abstract'].fillna('No abstract available')
    
    # Create new features
    print("\n✨ Creating new features...")
    
    # Abstract word count
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    
    # Title word count
    df_clean['title_word_count'] = df_clean['title'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    
    # Extract COVID-related keywords
    covid_keywords = ['covid', 'coronavirus', 'sars-cov-2', 'pandemic', 'vaccine']
    df_clean['covid_related'] = df_clean['title'].str.lower().str.contains('|'.join(covid_keywords), na=False)
    
    # Filter out invalid years (keep 2019-2024)
    df_clean = df_clean[df_clean['year'].between(2019, 2024)]
    
    print(f"✅ Cleaning completed!")
    print(f"Original dataset: {len(df)} rows")
    print(f"Cleaned dataset: {len(df_clean)} rows")
    print(f"Data removed: {len(df) - len(df_clean)} rows ({((len(df) - len(df_clean))/len(df)*100):.1f}%)")
    
    return df_clean

# ================================================================
# PART 3: DATA ANALYSIS AND VISUALIZATION
# ================================================================

def perform_data_analysis(df):
    """
    Perform comprehensive data analysis and create visualizations
    """
    print("\n" + "="*50)
    print("PART 3: DATA ANALYSIS AND VISUALIZATION")
    print("="*50)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Publications over time
    print("\n📊 Analyzing publication trends...")
    ax1 = plt.subplot(2, 3, 1)
    
    yearly_counts = df.groupby('year').size().reset_index(name='count')
    plt.plot(yearly_counts['year'], yearly_counts['count'], marker='o', linewidth=3, markersize=8)
    plt.title('COVID-19 Research Publications Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add trend annotation
    peak_year = yearly_counts.loc[yearly_counts['count'].idxmax(), 'year']
    peak_count = yearly_counts['count'].max()
    plt.annotate(f'Peak: {peak_count} papers', 
                xy=(peak_year, peak_count), 
                xytext=(peak_year, peak_count + 50),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center')
    
    # 2. Top journals
    print("📊 Analyzing top publishing journals...")
    ax2 = plt.subplot(2, 3, 2)
    
    top_journals = df['journal'].value_counts().head(10)
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_journals)))
    bars = plt.bar(range(len(top_journals)), top_journals.values, color=colors)
    plt.title('Top 10 Journals Publishing COVID-19 Research', fontsize=14, fontweight='bold')
    plt.xlabel('Journals', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.xticks(range(len(top_journals)), 
               [journal[:15] + '...' if len(journal) > 15 else journal 
                for journal in top_journals.index], 
               rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 3. Publication sources
    print("📊 Analyzing publication sources...")
    ax3 = plt.subplot(2, 3, 3)
    
    source_counts = df['source_x'].value_counts()
    plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Publication Sources', fontsize=14, fontweight='bold')
    
    # 4. Abstract word count distribution
    print("📊 Analyzing abstract lengths...")
    ax4 = plt.subplot(2, 3, 4)
    
    plt.hist(df['abstract_word_count'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['abstract_word_count'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["abstract_word_count"].mean():.0f} words')
    plt.axvline(df['abstract_word_count'].median(), color='orange', linestyle='--', 
                label=f'Median: {df["abstract_word_count"].median():.0f} words')
    plt.title('Distribution of Abstract Word Counts', fontsize=14, fontweight='bold')
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Monthly publication trends
    print("📊 Analyzing monthly trends...")
    ax5 = plt.subplot(2, 3, 5)
    
    monthly_counts = df.groupby(['year', 'month']).size().reset_index(name='count')
    monthly_counts['date'] = pd.to_datetime(monthly_counts[['year', 'month']].assign(day=1))
    
    plt.plot(monthly_counts['date'], monthly_counts['count'], marker='o', alpha=0.7)
    plt.title('Monthly Publication Trends', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. COVID-related vs Other research
    print("📊 Analyzing COVID-related research...")
    ax6 = plt.subplot(2, 3, 6)
    
    covid_counts = df['covid_related'].value_counts()
    labels = ['Other Research', 'COVID-related']
    colors = ['lightcoral', 'lightblue']
    
    wedges, texts, autotexts = plt.pie(covid_counts.values, labels=labels, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    plt.title('COVID-related vs Other Research', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Generate word cloud
    create_word_cloud(df)
    
    # Print analysis summary
    print_analysis_summary(df)
    
    return df

def create_word_cloud(df):
    """
    Create and display word cloud from paper titles
    """
    print("\n☁️ Creating word cloud from paper titles...")
    
    try:
        # Combine all titles
        all_titles = ' '.join(df['title'].dropna().astype(str))
        
        # Clean the text
        all_titles = re.sub(r'[^\w\s]', '', all_titles.lower())
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            max_words=100,
                            colormap='viridis').generate(all_titles)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Paper Titles', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("⚠️ WordCloud library not installed. Skipping word cloud generation.")
        print("Install with: pip install wordcloud")
    except Exception as e:
        print(f"⚠️ Error creating word cloud: {e}")

def print_analysis_summary(df):
    """
    Print comprehensive analysis summary
    """
    print("\n" + "="*50)
    print("📊 DATA ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\n📈 PUBLICATION STATISTICS:")
    print(f"• Total papers analyzed: {len(df):,}")
    print(f"• Date range: {df['year'].min()} - {df['year'].max()}")
    print(f"• Peak publication year: {df['year'].value_counts().index[0]} ({df['year'].value_counts().iloc[0]:,} papers)")
    print(f"• Average papers per year: {len(df) / (df['year'].max() - df['year'].min() + 1):.0f}")
    
    print(f"\n📚 JOURNAL STATISTICS:")
    print(f"• Total unique journals: {df['journal'].nunique()}")
    print(f"• Top journal: {df['journal'].value_counts().index[0]} ({df['journal'].value_counts().iloc[0]} papers)")
    print(f"• Average papers per journal: {len(df) / df['journal'].nunique():.1f}")
    
    print(f"\n📄 CONTENT STATISTICS:")
    print(f"• Average abstract length: {df['abstract_word_count'].mean():.0f} words")
    print(f"• Average title length: {df['title_word_count'].mean():.1f} words")
    print(f"• COVID-related papers: {df['covid_related'].sum()} ({df['covid_related'].mean()*100:.1f}%)")
    
    print(f"\n📊 SOURCE DISTRIBUTION:")
    source_stats = df['source_x'].value_counts()
    for source, count in source_stats.head(5).items():
        print(f"• {source}: {count} papers ({count/len(df)*100:.1f}%)")

# ================================================================
# MAIN EXECUTION
# ================================================================

def main():
    """
    Main function to run the complete analysis
    """
    # Part 1: Load and explore data
    df = load_and_explore_data()
    
    if df is not None:
        # Part 2: Clean and prepare data
        df_clean = clean_and_prepare_data(df)
        
        # Part 3: Perform analysis and create visualizations
        df_analyzed = perform_data_analysis(df_clean)
        
        # Save cleaned data for Streamlit app
        print("\n💾 Saving cleaned data for Streamlit app...")
        df_analyzed.to_csv('cord19_cleaned.csv', index=False)
        print("✅ Data saved as 'cord19_cleaned.csv'")
        
        print("\n🎉 Analysis completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Run the Streamlit app: streamlit run streamlit_app.py")
        print("2. Upload your code to GitHub repository")
        print("3. Submit the GitHub URL for your assignment")
        
        return df_analyzed
    else:
        print("❌ Analysis failed due to data loading issues")
        return None

if __name__ == "__main__":
    result_df = main()
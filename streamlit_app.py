# CORD-19 Data Explorer - Streamlit Web Application
# Part 4 of the CORD-19 Assignment

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .sidebar-content {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading for better performance
@st.cache_data
def load_data():
    """Load the cleaned CORD-19 dataset"""
    try:
        df = pd.read_csv('cord19_cleaned.csv')
        df['publish_time'] = pd.to_datetime(df['publish_time'])
        return df
    except FileNotFoundError:
        # Generate sample data if file doesn't exist
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_papers = 1000
    
    journals = ['Nature', 'Science', 'Cell', 'The Lancet', 'NEJM', 'PLOS ONE', 'BMJ', 'JAMA']
    sources = ['PMC', 'PubMed', 'WHO', 'bioRxiv', 'medRxiv']
    
    data = {
        'title': [f"COVID-19 Research Study {i}" for i in range(n_papers)],
        'abstract': [f"Abstract for study {i} about COVID-19..." for i in range(n_papers)],
        'publish_time': pd.date_range('2019-01-01', '2023-12-31', periods=n_papers),
        'journal': np.random.choice(journals, n_papers),
        'source_x': np.random.choice(sources, n_papers),
        'year': np.random.choice([2019, 2020, 2021, 2022, 2023], n_papers),
        'month': np.random.choice(range(1, 13), n_papers),
        'abstract_word_count': np.random.normal(200, 50, n_papers).astype(int),
        'title_word_count': np.random.normal(10, 3, n_papers).astype(int),
        'covid_related': np.random.choice([True, False], n_papers, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🦠 CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Analysis of COVID-19 Research Papers")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar controls
    st.sidebar.markdown("## 📊 Analysis Controls")
    
    # Date range selector
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_year, max_year, (min_year, max_year),
        help="Filter papers by publication year"
    )
    
    # Journal selector
    all_journals = ['All Journals'] + sorted(df['journal'].unique().tolist())
    selected_journal = st.sidebar.selectbox(
        "Select Journal",
        all_journals,
        help="Filter papers by specific journal"
    )
    
    # Source selector
    all_sources = ['All Sources'] + sorted(df['source_x'].unique().tolist())
    selected_source = st.sidebar.selectbox(
        "Select Source",
        all_sources,
        help="Filter papers by publication source"
    )
    
    # COVID-related filter
    covid_filter = st.sidebar.radio(
        "Paper Type",
        ["All Papers", "COVID-related Only", "Non-COVID Only"],
        help="Filter by COVID-19 relevance"
    )
    
    # Apply filters
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    if selected_journal != 'All Journals':
        filtered_df = filtered_df[filtered_df['journal'] == selected_journal]
    
    if selected_source != 'All Sources':
        filtered_df = filtered_df[filtered_df['source_x'] == selected_source]
    
    if covid_filter == "COVID-related Only":
        filtered_df = filtered_df[filtered_df['covid_related'] == True]
    elif covid_filter == "Non-COVID Only":
        filtered_df = filtered_df[filtered_df['covid_related'] == False]
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Papers",
            f"{len(filtered_df):,}",
            f"{len(filtered_df) - len(df):+,}" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        st.metric(
            "Unique Journals",
            f"{filtered_df['journal'].nunique():,}",
            help="Number of different journals"
        )
    
    with col3:
        st.metric(
            "Date Range",
            f"{filtered_df['year'].min()}-{filtered_df['year'].max()}",
            help="Publication year range"
        )
    
    with col4:
        covid_percentage = (filtered_df['covid_related'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric(
            "COVID-related",
            f"{covid_percentage:.1f}%",
            help="Percentage of COVID-related papers"
        )
    
    st.markdown("---")
    
    # Visualization section
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Distributions", "🔍 Analysis", "📄 Data Sample"])
    
    with tab1:
        st.subheader("Publication Trends Over Time")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Yearly trend
            yearly_counts = filtered_df.groupby('year').size().reset_index(name='count')
            
            fig_yearly = px.line(
                yearly_counts, 
                x='year', 
                y='count',
                title='Papers Published by Year',
                markers=True
            )
            fig_yearly.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of Papers",
                hovermode='x'
            )
            st.plotly_chart(fig_yearly, use_container_width=True)
        
        with col2:
            # Monthly trend (if multiple years selected)
            if year_range[1] - year_range[0] > 0:
                monthly_counts = filtered_df.groupby(['year', 'month']).size().reset_index(name='count')
                monthly_counts['date'] = pd.to_datetime(monthly_counts[['year', 'month']].assign(day=1))
                
                fig_monthly = px.line(
                    monthly_counts,
                    x='date',
                    y='count',
                    title='Monthly Publication Trends'
                )
                fig_monthly.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Papers"
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                # Monthly distribution for single year
                monthly_counts = filtered_df['month'].value_counts().sort_index()
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig_months = px.bar(
                    x=[months[i-1] for i in monthly_counts.index],
                    y=monthly_counts.values,
                    title=f'Monthly Distribution ({year_range[0]})'
                )
                st.plotly_chart(fig_months, use_container_width=True)
    
    with tab2:
        st.subheader("Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top journals
            top_journals = filtered_df['journal'].value_counts().head(10)
            
            fig_journals = px.bar(
                x=top_journals.values,
                y=top_journals.index,
                orientation='h',
                title='Top 10 Publishing Journals',
                labels={'x': 'Number of Papers', 'y': 'Journal'}
            )
            fig_journals.update_layout(height=400)
            st.plotly_chart(fig_journals, use_container_width=True)
        
        with col2:
            # Source distribution
            source_counts = filtered_df['source_x'].value_counts()
            
            fig_sources = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title='Publication Sources Distribution'
            )
            st.plotly_chart(fig_sources, use_container_width=True)
    
    with tab3:
        st.subheader("Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Abstract word count distribution
            fig_hist = px.histogram(
                filtered_df,
                x='abstract_word_count',
                nbins=30,
                title='Distribution of Abstract Word Counts'
            )
            fig_hist.add_vline(
                x=filtered_df['abstract_word_count'].mean(),
                line_dash="dash",
                annotation_text=f"Mean: {filtered_df['abstract_word_count'].mean():.0f}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # COVID vs Non-COVID comparison
            covid_comparison = filtered_df['covid_related'].value_counts()
            labels = ['Non-COVID Related', 'COVID Related']
            
            fig_covid = px.pie(
                values=covid_comparison.values,
                names=[labels[i] for i in covid_comparison.index],
                title='COVID-Related vs Other Research',
                color_discrete_sequence=['lightcoral', 'lightblue']
            )
            st.plotly_chart(fig_covid, use_container_width=True)
        
        # Word frequency analysis
        st.subheader("📝 Title Word Frequency Analysis")
        
        # Get most common words from titles
        all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
        # Clean and split words
        words = re.findall(r'\b\w+\b', all_titles.lower())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        if filtered_words:
            word_freq = Counter(filtered_words).most_common(15)
            words_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            
            fig_words = px.bar(
                words_df,
                x='Frequency',
                y='Word',
                orientation='h',
                title='Most Frequent Words in Paper Titles',
                color='Frequency',
                color_continuous_scale='viridis'
            )
            fig_words.update_layout(height=500)
            st.plotly_chart(fig_words, use_container_width=True)
    
    with tab4:
        st.subheader("📄 Sample Data")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider("Sample Size", 5, min(100, len(filtered_df)), 10)
        with col2:
            sort_by = st.selectbox("Sort By", ['year', 'journal', 'title_word_count', 'abstract_word_count'])
        
        # Display sample
        sample_df = filtered_df.sort_values(sort_by, ascending=False).head(sample_size)
        
        # Select relevant columns for display
        display_columns = ['title', 'journal', 'year', 'source_x', 'abstract_word_count', 'covid_related']
        
        st.dataframe(
            sample_df[display_columns],
            use_container_width=True,
            hide_index=True
        )
        
        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f'cord19_filtered_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv'
        )
    
    # Sidebar statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📈 Current Selection Stats")
    
    if len(filtered_df) > 0:
        st.sidebar.markdown(f"**Papers:** {len(filtered_df):,}")
        st.sidebar.markdown(f"**Avg Abstract Length:** {filtered_df['abstract_word_count'].mean():.0f} words")
        st.sidebar.markdown(f"**Most Common Journal:** {filtered_df['journal'].mode().iloc[0]}")
        st.sidebar.markdown(f"**Peak Year:** {filtered_df['year'].mode().iloc[0]}")
    else:
        st.sidebar.warning("No data matches current filters")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this App:**
    This interactive dashboard explores the CORD-19 dataset of COVID-19 research papers. 
    Use the sidebar controls to filter and analyze the data. 
    
    **Data Source:** CORD-19 Research Challenge Dataset
    **Created for:** Data Science Frameworks Assignment
    """)

# Advanced Analysis Functions
def show_advanced_analysis(df):
    """Show advanced analysis section"""
    st.subheader("🔬 Advanced Analysis")
    
    # Correlation analysis
    numeric_cols = ['abstract_word_count', 'title_word_count', 'year']
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix of Numeric Features",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Journal performance over time
    st.subheader("📚 Journal Performance Over Time")
    
    top_journals = df['journal'].value_counts().head(5).index
    journal_yearly = df[df['journal'].isin(top_journals)].groupby(['journal', 'year']).size().reset_index(name='count')
    
    fig_journal_time = px.line(
        journal_yearly,
        x='year',
        y='count',
        color='journal',
        title='Publication Trends by Top Journals',
        markers=True
    )
    st.plotly_chart(fig_journal_time, use_container_width=True)

def show_insights(df):
    """Show key insights section"""
    st.subheader("💡 Key Insights")
    
    insights = []
    
    # Peak publication year
    peak_year = df['year'].mode().iloc[0]
    peak_count = (df['year'] == peak_year).sum()
    insights.append(f"📈 **Peak Publication Year:** {peak_year} with {peak_count:,} papers")
    
    # Most productive journal
    top_journal = df['journal'].mode().iloc[0]
    top_journal_count = (df['journal'] == top_journal).sum()
    insights.append(f"📚 **Most Productive Journal:** {top_journal} ({top_journal_count:,} papers)")
    
    # Average abstract length
    avg_abstract = df['abstract_word_count'].mean()
    insights.append(f"📝 **Average Abstract Length:** {avg_abstract:.0f} words")
    
    # COVID-related percentage
    covid_pct = df['covid_related'].mean() * 100
    insights.append(f"🦠 **COVID-Related Papers:** {covid_pct:.1f}% of all papers")
    
    # Growth trend
    if df['year'].nunique() > 1:
        early_year = df['year'].min()
        late_year = df['year'].max()
        early_count = (df['year'] == early_year).sum()
        late_count = (df['year'] == late_year).sum()
        
        if late_count > early_count:
            growth = ((late_count - early_count) / early_count) * 100
            insights.append(f"📊 **Growth Trend:** {growth:+.0f}% from {early_year} to {late_year}")
    
    for insight in insights:
        st.markdown(insight)

# Run the app
if __name__ == "__main__":
    main()

# Additional utility functions for the app

@st.cache_data
def get_summary_stats(df):
    """Get summary statistics for the dataset"""
    return {
        'total_papers': len(df),
        'unique_journals': df['journal'].nunique(),
        'date_range': f"{df['year'].min()}-{df['year'].max()}",
        'avg_abstract_length': df['abstract_word_count'].mean(),
        'covid_percentage': df['covid_related'].mean() * 100,
        'top_journal': df['journal'].mode().iloc[0],
        'peak_year': df['year'].mode().iloc[0]
    }

def create_publication_timeline(df):
    """Create an enhanced publication timeline"""
    timeline_data = df.groupby(['year', 'covid_related']).size().reset_index(name='count')
    timeline_data['type'] = timeline_data['covid_related'].map({True: 'COVID-Related', False: 'Other Research'})
    
    fig = px.bar(
        timeline_data,
        x='year',
        y='count',
        color='type',
        title='Publication Timeline by Research Type',
        color_discrete_map={'COVID-Related': '#ff6b6b', 'Other Research': '#4ecdc4'}
    )
    
    return fig

def analyze_journal_diversity(df):
    """Analyze journal diversity and concentration"""
    journal_counts = df['journal'].value_counts()
    
    # Calculate concentration metrics
    total_papers = len(df)
    top_5_share = journal_counts.head(5).sum() / total_papers * 100
    top_10_share = journal_counts.head(10).sum() / total_papers * 100
    
    return {
        'total_journals': len(journal_counts),
        'top_5_concentration': top_5_share,
        'top_10_concentration': top_10_share,
        'herfindahl_index': ((journal_counts / total_papers) ** 2).sum()
    }
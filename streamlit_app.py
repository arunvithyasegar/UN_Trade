import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Guidance Tamil Nadu - Electrical Machinery Export Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Electrical Machinery and Equipment Export Analysis (2016-2024)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Guidance Tamil Nadu / Invest Tamil Nadu Assessment</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        # Try to load the CSV file
        df = pd.read_csv('https://raw.githubusercontent.com/arunvithyasegar/UN_Trade_dataset/main/TradeData_5_10_2025_12_27_1.csv', encoding='cp1252')
        
        # Use primaryValue as the main value column, fallback to fobvalue if needed
        df['export_value'] = df['primaryValue'].fillna(df['fobvalue'])
        
        # Filter for World partners only
        df = df[df['partnerDesc'] == 'World']
        
        # Filter for Export trade flow only
        df = df[df['flowDesc'] == 'Export']
        
        # Keep only essential columns
        clean_df = df[['refYear', 'reporterDesc', 'export_value']]
        
        # Filter for countries with export values above $500 million in 2024
        df_2024 = clean_df[clean_df['refYear'] == 2024]
        high_export_countries = df_2024[df_2024['export_value'] >= 500000000]['reporterDesc'].unique()
        
        # Filter the dataset to include only those countries
        filtered_df = clean_df[clean_df['reporterDesc'].isin(high_export_countries)]
        
        # Pivot the data for easier analysis
        pivot_df = filtered_df.pivot(index='reporterDesc', columns='refYear', values='export_value')
        pivot_df.columns = [str(col) for col in pivot_df.columns]  # Convert year columns to strings
        
        # Create a DataFrame for growth rates
        growth_df = pd.DataFrame(index=pivot_df.index)
        
        # Calculate YoY growth rates
        for year in range(2017, 2025):
            current_year = str(year)
            prev_year = str(year - 1)
            growth_df[f'growth_{current_year}'] = (pivot_df[current_year] - pivot_df[prev_year]) / pivot_df[prev_year]
        
        # Calculate average annual growth rate
        growth_df['avg_growth'] = growth_df.mean(axis=1)
        
        # Calculate volatility (coefficient of variation)
        std_values = pivot_df.std(axis=1)
        mean_values = pivot_df.mean(axis=1)
        growth_df['volatility'] = std_values / mean_values
        
        # Add the 2024 value for reference
        growth_df['value_2024'] = pivot_df['2024']
        
        # Classify countries into quadrants
        median_growth = growth_df['avg_growth'].median()
        median_volatility = growth_df['volatility'].median()
        
        conditions = [
            (growth_df['avg_growth'] >= median_growth) & (growth_df['volatility'] < median_volatility),
            (growth_df['avg_growth'] >= median_growth) & (growth_df['volatility'] >= median_volatility),
            (growth_df['avg_growth'] < median_growth) & (growth_df['volatility'] < median_volatility),
            (growth_df['avg_growth'] < median_growth) & (growth_df['volatility'] >= median_volatility)
        ]
        choices = ['Stable High-Growth', 'Volatile High-Growth', 'Stable Low-Growth', 'Volatile Low-Growth']
        growth_df['classification'] = np.select(conditions, choices, default='Unknown')
        
        # Calculate percentiles
        growth_df['percentile'] = growth_df['avg_growth'].rank(pct=True)
        
        return filtered_df, pivot_df, growth_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Create sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis Page",
    ["Overview", "Data Preparation", "Growth Trend Analysis", "Volatility & Classification", "Statistical Analysis"]
)

# Load data
filtered_df, pivot_df, growth_df = load_data()

if filtered_df is None or pivot_df is None or growth_df is None:
    st.error("Failed to load data. Please check the data source.")
    st.stop()

# Overview Page
if page == "Overview":
    st.markdown("<h2 class='sub-header'>Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset Information")
        st.markdown("""
        - **Source**: UN Comtrade Database
        - **Period**: 2016-2024
        - **Type**: Goods
        - **Frequency**: Annual
        - **HS Code**: 85 (Electrical machinery and equipment)
        - **Trade Flow**: Export
        - **Measure**: USD
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Key Metrics")
        total_countries = len(pivot_df)
        total_export_2024 = pivot_df['2024'].sum() / 1e9  # Convert to billions
        avg_growth = growth_df['avg_growth'].mean() * 100  # Convert to percentage
        
        st.metric("Total Countries Analyzed", f"{total_countries}")
        st.metric("Total Exports in 2024", f"${total_export_2024:.2f} Billion")
        st.metric("Average Annual Growth Rate", f"{avg_growth:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # World map of export values
    st.markdown("<h3>Global Distribution of Electrical Machinery Exports (2024)</h3>", unsafe_allow_html=True)
    
    # Prepare data for map
    map_data = pd.DataFrame({
        'Country': pivot_df.index,
        'Export_Value': pivot_df['2024'] / 1e9  # Convert to billions for better visualization
    })
    
    fig = px.choropleth(
        map_data,
        locations='Country',
        locationmode='country names',
        color='Export_Value',
        hover_name='Country',
        color_continuous_scale='Viridis',
        title='Electrical Machinery Exports by Country (2024, Billion USD)',
        labels={'Export_Value': 'Export Value (Billion USD)'}
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 10 exporters
    st.markdown("<h3>Top 10 Exporters in 2024</h3>", unsafe_allow_html=True)
    top_exporters = pivot_df.sort_values('2024', ascending=False).head(10)
    
    fig = px.bar(
        x=top_exporters.index,
        y=top_exporters['2024'] / 1e9,
        labels={'x': 'Country', 'y': 'Export Value (Billion USD)'},
        title='Top 10 Electrical Machinery Exporters in 2024',
        color=top_exporters['2024'] / 1e9,
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Data Preparation Page
elif page == "Data Preparation":
    st.markdown("<h2 class='sub-header'>Data Preparation</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
    <h3>Data Cleaning Process</h3>
    <ol>
        <li>Imported the dataset from UN Comtrade Database</li>
        <li>Filtered for HS Code 85 (Electrical machinery and equipment)</li>
        <li>Filtered for Export trade flow only</li>
        <li>Filtered for World as partner country</li>
        <li>Selected essential columns for analysis</li>
        <li>Filtered for countries with export values above $500 million in 2024</li>
        <li>Created pivot table for year-wise analysis</li>
        <li>Calculated growth rates and volatility metrics</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Show sample of raw data
    st.subheader("Sample of Cleaned Data")
    st.dataframe(filtered_df.head(10))
    
    # Show pivot table
    st.subheader("Pivot Table (Countries × Years)")
    
    # Format the pivot table for better readability
    formatted_pivot = pivot_df.copy()
    for col in formatted_pivot.columns:
        formatted_pivot[col] = formatted_pivot[col].apply(lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) else "N/A")
    
    st.dataframe(formatted_pivot)
    
    # Export option
    st.subheader("Export Cleaned Data")
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')
    
    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        label="Download Cleaned Data as CSV",
        data=csv,
        file_name='cleaned_electrical_exports_data.csv',
        mime='text/csv',
    )

# Growth Trend Analysis Page
elif page == "Growth Trend Analysis":
    st.markdown("<h2 class='sub-header'>Growth Trend Analysis</h2>", unsafe_allow_html=True)
    
    # YoY Growth Rates
    st.subheader("Year-on-Year Growth Rates")
    
    # Get growth rate columns
    growth_cols = [col for col in growth_df.columns if col.startswith('growth_')]
    
    # Format growth rates for display
    growth_display = growth_df[growth_cols].copy()
    for col in growth_display.columns:
        growth_display[col] = growth_display[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
    
    st.dataframe(growth_display)
    
    # Top countries by average growth rate
    st.subheader("Countries Ranked by Average Annual Growth Rate")
    
    top_growth = growth_df.sort_values('avg_growth', ascending=False)
    
    fig = px.bar(
        x=top_growth.index,
        y=top_growth['avg_growth'] * 100,  # Convert to percentage
        labels={'x': 'Country', 'y': 'Average Annual Growth Rate (%)'},
        title='Countries Ranked by Average Annual Growth Rate',
        color=top_growth['avg_growth'] * 100,
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Export trends for top 3 countries
    st.subheader("Export Trends for Top 3 Countries by Growth Rate")
    
    top3_countries = top_growth.head(3).index.tolist()
    
    # Create line plot for top 3 countries
    fig = go.Figure()
    
    for country in top3_countries:
        fig.add_trace(go.Scatter(
            x=pivot_df.columns,
            y=pivot_df.loc[country] / 1e9,  # Convert to billions
            mode='lines+markers',
            name=country
        ))
    
    fig.update_layout(
        title='Export Trends for Top 3 Countries by Growth Rate (2016-2024)',
        xaxis_title='Year',
        yaxis_title='Export Value (Billion USD)',
        legend_title='Country',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison with global average
    st.subheader("Growth Comparison with Global Average")
    
    # Calculate global average for each year
    global_avg = pivot_df.mean() / 1e9  # Convert to billions
    
    # Create a DataFrame for the global average
    global_avg_df = pd.DataFrame({
        'Year': global_avg.index,
        'Global Average': global_avg.values
    })
    
    # Create line plot comparing top countries with global average
    fig = go.Figure()
    
    # Add global average
    fig.add_trace(go.Scatter(
        x=global_avg_df['Year'],
        y=global_avg_df['Global Average'],
        mode='lines+markers',
        name='Global Average',
        line=dict(color='black', width=3, dash='dash')
    ))
    
    # Add top 3 countries
    for country in top3_countries:
        fig.add_trace(go.Scatter(
            x=pivot_df.columns,
            y=pivot_df.loc[country] / 1e9,
            mode='lines+markers',
            name=country
        ))
    
    fig.update_layout(
        title='Export Trends: Top 3 Countries vs. Global Average (2016-2024)',
        xaxis_title='Year',
        yaxis_title='Export Value (Billion USD)',
        legend_title='Country',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Volatility & Classification Page
elif page == "Volatility & Classification":
    st.markdown("<h2 class='sub-header'>Volatility & Classification Analysis</h2>", unsafe_allow_html=True)
    
    # Top 10 most volatile exporters
    st.subheader("Top 10 Most Volatile Exporters")
    
    top_volatile = growth_df.sort_values('volatility', ascending=False).head(10)
    
    fig = px.bar(
        x=top_volatile.index,
        y=top_volatile['volatility'],
        labels={'x': 'Country', 'y': 'Volatility (Coefficient of Variation)'},
        title='Top 10 Most Volatile Exporters',
        color=top_volatile['volatility'],
        color_continuous_scale='Reds'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Compare volatility with average growth rate
    st.subheader("Volatility vs. Average Growth Rate")
    
    fig = px.scatter(
        x=growth_df['volatility'],
        y=growth_df['avg_growth'] * 100,  # Convert to percentage
        hover_name=growth_df.index,
        labels={
            'x': 'Volatility (Coefficient of Variation)',
            'y': 'Average Annual Growth Rate (%)'
        },
        title='Volatility vs. Average Growth Rate',
        color=growth_df['value_2024'] / 1e9,  # Color by export value in 2024
        size=growth_df['value_2024'] / 1e9,  # Size by export value in 2024
        color_continuous_scale='Viridis',
        size_max=50
    )
    
    # Add quadrant lines
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1),
        x0=growth_df['volatility'].median(), y0=growth_df['avg_growth'].min() * 100,
        x1=growth_df['volatility'].median(), y1=growth_df['avg_growth'].max() * 100
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1),
        x0=growth_df['volatility'].min(), y0=growth_df['avg_growth'].median() * 100,
        x1=growth_df['volatility'].max(), y1=growth_df['avg_growth'].median() * 100
    )
    
    # Add quadrant labels
    fig.add_annotation(
        x=growth_df['volatility'].median() / 2,
        y=growth_df['avg_growth'].median() * 100 * 1.5,
        text="Stable High-Growth",
        showarrow=False,
        font=dict(size=14, color="green")
    )
    fig.add_annotation(
        x=growth_df['volatility'].median() * 1.5,
        y=growth_df['avg_growth'].median() * 100 * 1.5,
        text="Volatile High-Growth",
        showarrow=False,
        font=dict(size=14, color="orange")
    )
    fig.add_annotation(
        x=growth_df['volatility'].median() / 2,
        y=growth_df['avg_growth'].median() * 100 * 0.5,
        text="Stable Low-Growth",
        showarrow=False,
        font=dict(size=14, color="blue")
    )
    fig.add_annotation(
        x=growth_df['volatility'].median() * 1.5,
        y=growth_df['avg_growth'].median() * 100 * 0.5,
        text="Volatile Low-Growth",
        showarrow=False,
        font=dict(size=14, color="red")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Country Classification
    st.subheader("Country Classification")
    
    # Count countries in each classification
    classification_counts = growth_df['classification'].value_counts().reset_index()
    classification_counts.columns = ['Classification', 'Count']
    
    # Create pie chart
    fig = px.pie(
        classification_counts,
        values='Count',
        names='Classification',
        title='Distribution of Countries by Classification',
        color='Classification',
        color_discrete_map={
            'Stable High-Growth': 'green',
            'Volatile High-Growth': 'orange',
            'Stable Low-Growth': 'blue',
            'Volatile Low-Growth': 'red'
        }
    )
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show table of countries by classification
        st.markdown("<h4>Countries by Classification</h4>", unsafe_allow_html=True)
        
        # Create tabs for each classification
        tabs = st.tabs(['Stable High-Growth', 'Volatile High-Growth', 'Stable Low-Growth', 'Volatile Low-Growth'])
        
        for i, classification in enumerate(['Stable High-Growth', 'Volatile High-Growth', 'Stable Low-Growth', 'Volatile Low-Growth']):
            with tabs[i]:
                countries = growth_df[growth_df['classification'] == classification].sort_values('avg_growth', ascending=False)
                
                # Display countries with their growth rate and volatility
                for country in countries.index:
                    growth_val = countries.loc[country, 'avg_growth'] * 100
                    volatility_val = countries.loc[country, 'volatility']
                    export_val = countries.loc[country, 'value_2024'] / 1e9
                    
                    st.markdown(f"""
                    <div class='highlight'>
                        <b>{country}</b><br>
                        Growth Rate: {growth_val:.2f}%<br>
                        Volatility: {volatility_val:.4f}<br>
                        2024 Exports: ${export_val:.2f} Billion
                    </div>
                    """, unsafe_allow_html=True)

# Statistical Analysis Page
elif page == "Statistical Analysis":
    st.markdown("<h2 class='sub-header'>Statistical & Forecasting Analysis</h2>", unsafe_allow_html=True)
    
    # Distribution of growth rates
    st.subheader("Distribution of Average Annual Growth Rates")
    
    # Calculate mean and standard deviation for normal distribution
    mean_growth = growth_df['avg_growth'].mean()
    std_growth = growth_df['avg_growth'].std()
    
    # Create histogram with normal distribution overlay
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=growth_df['avg_growth'] * 100,  # Convert to percentage
            name="Frequency",
            nbinsx=20,
            marker_color='rgba(73, 160, 181, 0.7)'
        )
    )
    
    # Generate normal distribution curve
    x = np.linspace(
        growth_df['avg_growth'].min() * 100,
        growth_df['avg_growth'].max() * 100,
        100
    )
    y = stats.norm.pdf(x, mean_growth * 100, std_growth * 100)
    
    # Scale the normal distribution to match histogram height
    hist_max = np.histogram(growth_df['avg_growth'] * 100, bins=20)[0].max()
    pdf_max = max(y)
    scale_factor = hist_max / pdf_max
    
    # Add normal distribution curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y * scale_factor,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ),
        secondary_y=False
    )
    
    fig.update_layout(
        title='Distribution of Average Annual Growth Rates with Normal Distribution Overlay',
        xaxis_title='Average Annual Growth Rate (%)',
        yaxis_title='Frequency',
        bargap=0.1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top and Bottom Performers
    st.subheader("Top and Bottom Performers")
    
    # Identify top and bottom 10% performers
    top_performers = growth_df[growth_df['percentile'] >= 0.9].sort_values('avg_growth', ascending=False)
    bottom_performers = growth_df[growth_df['percentile'] <= 0.1].sort_values('avg_growth')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Top 10% Performers</h4>", unsafe_allow_html=True)
        
        for country in top_performers.index:
            growth_val = top_performers.loc[country, 'avg_growth'] * 100
            export_val = top_performers.loc[country, 'value_2024'] / 1e9
            
            st.markdown(f"""
            <div class='highlight' style='background-color: #DCFCE7;'>
                <b>{country}</b><br>
                Growth Rate: {growth_val:.2f}%<br>
                2024 Exports: ${export_val:.2f} Billion
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4>Bottom 10% Performers</h4>", unsafe_allow_html=True)
        
        for country in bottom_performers.index:
            growth_val = bottom_performers.loc[country, 'avg_growth'] * 100
            export_val = bottom_performers.loc[country, 'value_2024'] / 1e9
            
            st.markdown(f"""
            <div class='highlight' style='background-color: #FEE2E2;'>
                <b>{country}</b><br>
                Growth Rate: {growth_val:.2f}%<br>
                2024 Exports: ${export_val:.2f} Billion
            </div>
            """, unsafe_allow_html=True)
    
    # Visualize performance segments on histogram
    st.subheader("Performance Segments Visualization")
    
    # Create a column for performance category
    growth_df['performance_category'] = 'Average'
    growth_df.loc[growth_df['percentile'] >= 0.9, 'performance_category'] = 'Top Performers'
    growth_df.loc[growth_df['percentile'] <= 0.1, 'performance_category'] = 'Underperformers'
    
    # Create histogram with colored segments
    fig = px.histogram(
        growth_df,
        x='avg_growth',
        color='performance_category',
        nbins=20,
        labels={'avg_growth': 'Average Annual Growth Rate'},
        title='Distribution of Growth Rates by Performance Category',
        color_discrete_map={
            'Top Performers': 'green',
            'Average': 'gray',
            'Underperformers': 'red'
        }
    )
    
    # Convert x-axis to percentage
    fig.update_layout(
        xaxis_title='Average Annual Growth Rate (%)',
        xaxis=dict(tickformat='.1%')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plot of growth rates
    st.subheader("Box Plot of Growth Rates")
    
    fig = px.box(
        growth_df,
        y='avg_growth',
        points='all',
        hover_name=growth_df.index,
        labels={'avg_growth': 'Average Annual Growth Rate'},
        title='Box Plot of Average Annual Growth Rates'
    )
    
    # Convert y-axis to percentage
    fig.update_layout(
        yaxis_title='Average Annual Growth Rate (%)',
        yaxis=dict(tickformat='.1%')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights for Guidance Tamil Nadu / Invest Tamil Nadu
    st.markdown("<h3 class='sub-header'>Key Insights for Guidance Tamil Nadu / Invest Tamil Nadu</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
    <h4>Investment Opportunities</h4>
    <ul>
        <li><strong>Target Stable High-Growth Countries:</strong> These represent the most reliable investment partners for Tamil Nadu's electrical machinery sector.</li>
        <li><strong>Explore Volatile High-Growth Markets:</strong> These markets offer high growth potential but require risk mitigation strategies.</li>
        <li><strong>Identify Complementary Industries:</strong> Look for countries where Tamil Nadu can fill supply chain gaps in the electrical machinery sector.</li>
    </ul>
    
    <h4>Policy Recommendations</h4>
    <ul>
        <li><strong>Develop Targeted Export Promotion:</strong> Create country-specific export strategies based on the classification analysis.</li>
        <li><strong>Establish Risk Management Frameworks:</strong> Develop protocols for engaging with volatile markets to protect Tamil Nadu businesses.</li>
        <li><strong>Invest in Innovation:</strong> Focus R&D investments on electrical machinery subsectors showing the highest global growth.</li>
        <li><strong>Build Resilient Supply Chains:</strong> Diversify sourcing and export destinations to reduce vulnerability to market volatility.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 30px; padding: 10px; background-color: #F3F4F6; border-radius: 5px;'>
    <p>Developed for Guidance Tamil Nadu / Invest Tamil Nadu Assessment</p>
    <p>Data Source: UN Comtrade Database (HS Code 85: Electrical machinery and equipment)</p>
</div>
""", unsafe_allow_html=True)
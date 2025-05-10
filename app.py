import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Trade Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E6091;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E86C1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .highlight {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">Global Trade Data Analysis Dashboard</p>', unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    try:
        # Try to load the CSV file
        df = pd.read_csv('TradeData_5_10_2025_12_27_1_cleaned.csv', encoding='cp1252')
        
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

# Load data
filtered_df, pivot_df, growth_df = load_data()

if filtered_df is not None:
    # Add file uploader in sidebar for users to upload their own data
    st.sidebar.title("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your trade data CSV", type=["csv"])
    
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully! (Note: Currently using demo data)")
    
    # Add filters in sidebar
    st.sidebar.title("Filters")
    
    # Year filter
    years = sorted([int(col) for col in pivot_df.columns if col.isdigit()])
    start_year, end_year = st.sidebar.select_slider(
        "Select Year Range",
        options=years,
        value=(min(years), max(years))
    )
    
    # Growth rate filter
    min_growth = float(growth_df['avg_growth'].min())
    max_growth = float(growth_df['avg_growth'].max())
    growth_range = st.sidebar.slider(
        "Average Growth Rate Range",
        min_value=min_growth,
        max_value=max_growth,
        value=(min_growth, max_growth),
        format="%.2f"
    )
    
    # Classification filter
    classifications = growth_df['classification'].unique().tolist()
    selected_classifications = st.sidebar.multiselect(
        "Country Classification",
        options=classifications,
        default=classifications
    )
    
    # Filter data based on selections
    filtered_growth_df = growth_df[
        (growth_df['avg_growth'] >= growth_range[0]) & 
        (growth_df['avg_growth'] <= growth_range[1]) &
        (growth_df['classification'].isin(selected_classifications))
    ]
    
    # Number of countries to display in top/bottom lists
    num_countries = st.sidebar.slider("Number of Countries to Display", 5, 20, 10)
    
    # Top and bottom countries
    top_countries = filtered_growth_df.sort_values('avg_growth', ascending=False).head(num_countries).index.tolist()
    bottom_countries = filtered_growth_df.sort_values('avg_growth').head(num_countries).index.tolist()
    
    # Create dashboard layout with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Growth Analysis", "Country Classification", "Statistical Analysis"])
    
    with tab1:
        st.markdown('<p class="sub-header">Export Trade Data Overview</p>', unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Countries", 
                len(filtered_growth_df),
                delta=f"{len(filtered_growth_df) - len(growth_df)} from filters"
            )
        
        with col2:
            avg_growth = filtered_growth_df['avg_growth'].mean()
            st.metric(
                "Avg Annual Growth", 
                f"{avg_growth:.2%}"
            )
        
        with col3:
            avg_volatility = filtered_growth_df['volatility'].mean()
            st.metric(
                "Avg Volatility", 
                f"{avg_volatility:.4f}"
            )
        
        with col4:
            total_value = filtered_growth_df['value_2024'].sum() / 1e9
            st.metric(
                "Total 2024 Exports", 
                f"${total_value:.2f}B"
            )
        
        # Classification breakdown
        st.markdown('<p class="sub-header">Country Classification Breakdown</p>', unsafe_allow_html=True)
        
        # Create a pie chart for classification distribution
        classification_counts = filtered_growth_df['classification'].value_counts()
        fig_pie = px.pie(
            values=classification_counts.values,
            names=classification_counts.index,
            title="Distribution of Country Classifications",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Show top and bottom countries
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="sub-header">Top Growth Countries</p>', unsafe_allow_html=True)
            top_df = filtered_growth_df.loc[top_countries, ['avg_growth', 'volatility', 'classification']].sort_values('avg_growth', ascending=False)
            top_df['avg_growth'] = top_df['avg_growth'].apply(lambda x: f"{x:.2%}")
            top_df['volatility'] = top_df['volatility'].apply(lambda x: f"{x:.4f}")
            st.dataframe(top_df, use_container_width=True)
        
        with col2:
            st.markdown('<p class="sub-header">Bottom Growth Countries</p>', unsafe_allow_html=True)
            bottom_df = filtered_growth_df.loc[bottom_countries, ['avg_growth', 'volatility', 'classification']].sort_values('avg_growth')
            bottom_df['avg_growth'] = bottom_df['avg_growth'].apply(lambda x: f"{x:.2%}")
            bottom_df['volatility'] = bottom_df['volatility'].apply(lambda x: f"{x:.4f}")
            st.dataframe(bottom_df, use_container_width=True)
    
    with tab2:
        st.markdown('<p class="sub-header">Export Growth Trends Analysis</p>', unsafe_allow_html=True)
        
        # Country selector for trends
        selected_countries = st.multiselect(
            "Select Countries to Visualize Trends",
            options=filtered_growth_df.index.tolist(),
            default=top_countries[:3]
        )
        
        if selected_countries:
            # Create export value trend chart
            trend_data = filtered_df[filtered_df['reporterDesc'].isin(selected_countries)]
            trend_data['Year'] = trend_data['refYear'].astype(str)
            trend_data['export_value_billions'] = trend_data['export_value'] / 1e9
            
            fig_trend = px.line(
                trend_data,
                x='Year',
                y='export_value_billions',
                color='reporterDesc',
                markers=True,
                title=f"Export Value Trends ({start_year}-{end_year})",
                labels={'export_value_billions': 'Export Value (Billion USD)', 'reporterDesc': 'Country'}
            )
            fig_trend.update_layout(legend_title_text='Country')
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Create YoY growth rate chart
            growth_columns = [f'growth_{year}' for year in range(max(2017, start_year + 1), min(2025, end_year + 1))]
            growth_data = filtered_growth_df.loc[selected_countries, growth_columns].reset_index()
            growth_data_melted = pd.melt(
                growth_data, 
                id_vars=['reporterDesc'], 
                value_vars=growth_columns,
                var_name='Year', 
                value_name='Growth Rate'
            )
            growth_data_melted['Year'] = growth_data_melted['Year'].str.replace('growth_', '')
            
            fig_growth = px.line(
                growth_data_melted,
                x='Year',
                y='Growth Rate',
                color='reporterDesc',
                markers=True,
                title=f"Year-over-Year Growth Rates ({start_year+1}-{end_year})",
                labels={'reporterDesc': 'Country'}
            )
            fig_growth.update_layout(legend_title_text='Country')
            fig_growth.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig_growth, use_container_width=True)
        else:
            st.warning("Please select at least one country to visualize trends.")
    
    with tab3:
        st.markdown('<p class="sub-header">Country Classification Matrix</p>', unsafe_allow_html=True)
        
        # Create quadrant scatter plot
        classification_colors = {
            'Stable High-Growth': '#2ECC71',    # Green
            'Volatile High-Growth': '#F39C12',  # Orange
            'Stable Low-Growth': '#3498DB',     # Blue
            'Volatile Low-Growth': '#E74C3C'    # Red
        }
        
        # Allow user to select countries to highlight
        highlight_countries = st.multiselect(
            "Highlight Specific Countries",
            options=filtered_growth_df.index.tolist(),
            default=[]
        )
        
        fig_scatter = go.Figure()
        
        # Add data points by classification
        for classification in filtered_growth_df['classification'].unique():
            subset = filtered_growth_df[filtered_growth_df['classification'] == classification]
            
            # Split into highlighted and non-highlighted points
            if highlight_countries:
                # Non-highlighted points
                non_highlight = subset[~subset.index.isin(highlight_countries)]
                fig_scatter.add_trace(go.Scatter(
                    x=non_highlight['volatility'],
                    y=non_highlight['avg_growth'],
                    mode='markers',
                    marker=dict(
                        color=classification_colors.get(classification, '#AAAAAA'),
                        size=10,
                        opacity=0.6
                    ),
                    name=classification,
                    text=non_highlight.index,
                    hovertemplate='<b>%{text}</b><br>Growth: %{y:.2%}<br>Volatility: %{x:.4f}<extra></extra>'
                ))
                
                # Highlighted points
                highlight = subset[subset.index.isin(highlight_countries)]
                if not highlight.empty:
                    fig_scatter.add_trace(go.Scatter(
                        x=highlight['volatility'],
                        y=highlight['avg_growth'],
                        mode='markers+text',
                        marker=dict(
                            color=classification_colors.get(classification, '#AAAAAA'),
                            size=15,
                            line=dict(width=2, color='black')
                        ),
                        textposition="top center",
                        textfont=dict(size=12),
                        text=highlight.index,
                        name=f"{classification} (Highlighted)",
                        hovertemplate='<b>%{text}</b><br>Growth: %{y:.2%}<br>Volatility: %{x:.4f}<extra></extra>'
                    ))
            else:
                # No highlighting, just add all points
                fig_scatter.add_trace(go.Scatter(
                    x=subset['volatility'],
                    y=subset['avg_growth'],
                    mode='markers',
                    marker=dict(
                        color=classification_colors.get(classification, '#AAAAAA'),
                        size=12,
                        opacity=0.7
                    ),
                    name=classification,
                    text=subset.index,
                    hovertemplate='<b>%{text}</b><br>Growth: %{y:.2%}<br>Volatility: %{x:.4f}<extra></extra>'
                ))
        
        # Add quadrant lines
        fig_scatter.add_shape(
            type="line", line=dict(dash="dash", color="gray"),
            x0=growth_df['volatility'].min(), y0=growth_df['avg_growth'].median(),
            x1=growth_df['volatility'].max(), y1=growth_df['avg_growth'].median()
        )
        fig_scatter.add_shape(
            type="line", line=dict(dash="dash", color="gray"),
            x0=growth_df['volatility'].median(), y0=growth_df['avg_growth'].min(),
            x1=growth_df['volatility'].median(), y1=growth_df['avg_growth'].max()
        )
        
        # Add quadrant labels
        fig_scatter.add_annotation(
            x=growth_df['volatility'].median()/2,
            y=growth_df['avg_growth'].median() + (growth_df['avg_growth'].max() - growth_df['avg_growth'].median())/2,
            text="Stable High-Growth",
            showarrow=False,
            font=dict(size=12, color="#2ECC71")
        )
        fig_scatter.add_annotation(
            x=growth_df['volatility'].median() + (growth_df['volatility'].max() - growth_df['volatility'].median())/2,
            y=growth_df['avg_growth'].median() + (growth_df['avg_growth'].max() - growth_df['avg_growth'].median())/2,
            text="Volatile High-Growth",
            showarrow=False,
            font=dict(size=12, color="#F39C12")
        )
        fig_scatter.add_annotation(
            x=growth_df['volatility'].median()/2,
            y=growth_df['avg_growth'].median()/2,
            text="Stable Low-Growth",
            showarrow=False,
            font=dict(size=12, color="#3498DB")
        )
        fig_scatter.add_annotation(
            x=growth_df['volatility'].median() + (growth_df['volatility'].max() - growth_df['volatility'].median())/2,
            y=growth_df['avg_growth'].median()/2,
            text="Volatile Low-Growth",
            showarrow=False,
            font=dict(size=12, color="#E74C3C")
        )
        
        # Update layout
        fig_scatter.update_layout(
            title='Country Classification: Growth Rate vs Volatility',
            xaxis_title='Volatility (Coefficient of Variation)',
            yaxis_title='Average Annual Growth Rate',
            yaxis_tickformat='.0%',
            legend_title_text='Classification',
            height=600
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Show classification summary
        st.markdown('<p class="sub-header">Classification Summary</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show classification counts
            classification_counts = filtered_growth_df['classification'].value_counts().reset_index()
            classification_counts.columns = ['Classification', 'Count']
            
            fig_bar = px.bar(
                classification_counts,
                x='Classification',
                y='Count',
                color='Classification',
                color_discrete_map=classification_colors,
                title="Number of Countries by Classification"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Show average 2024 export value by classification
            class_value = filtered_growth_df.groupby('classification')['value_2024'].mean().reset_index()
            class_value['value_2024'] = class_value['value_2024'] / 1e9  # Convert to billions
            
            fig_value = px.bar(
                class_value,
                x='classification',
                y='value_2024',
                color='classification',
                color_discrete_map=classification_colors,
                title="Average 2024 Export Value by Classification (Billion USD)"
            )
            st.plotly_chart(fig_value, use_container_width=True)
    
    with tab4:
        st.markdown('<p class="sub-header">Statistical Analysis</p>', unsafe_allow_html=True)
        
        # Distribution of growth rates
        fig_hist = go.Figure()
        
        # Add histogram
        fig_hist.add_trace(go.Histogram(
            x=filtered_growth_df['avg_growth'],
            nbinsx=20,
            marker_color='lightblue',
            opacity=0.7,
            name='Growth Rate Distribution'
        ))
        
        # Add normal distribution curve
        mean = filtered_growth_df['avg_growth'].mean()
        std = filtered_growth_df['avg_growth'].std()
        x = np.linspace(filtered_growth_df['avg_growth'].min(), filtered_growth_df['avg_growth'].max(), 100)
        y = stats.norm.pdf(x, mean, std) * len(filtered_growth_df) * (filtered_growth_df['avg_growth'].max() - filtered_growth_df['avg_growth'].min()) / 20
        
        fig_hist.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Normal Distribution'
        ))
        
        # Add percentile lines
        p10 = filtered_growth_df['avg_growth'].quantile(0.1)
        p90 = filtered_growth_df['avg_growth'].quantile(0.9)
        
        fig_hist.add_shape(
            type="line", line=dict(dash="dash", color="red", width=2),
            x0=p10, y0=0, x1=p10, y1=y.max() * 1.2
        )
        
        fig_hist.add_shape(
            type="line", line=dict(dash="dash", color="green", width=2),
            x0=p90, y0=0, x1=p90, y1=y.max() * 1.2
        )
        
        fig_hist.add_shape(
            type="line", line=dict(dash="solid", color="blue", width=2),
            x0=mean, y0=0, x1=mean, y1=y.max() * 1.2
        )
        
        # Add annotations
        fig_hist.add_annotation(
            x=p10,
            y=y.max() * 1.1,
            text=f"10th Percentile: {p10:.2%}",
            showarrow=False,
            font=dict(color="red")
        )
        
        fig_hist.add_annotation(
            x=p90,
            y=y.max() * 1.1,
            text=f"90th Percentile: {p90:.2%}",
            showarrow=False,
            font=dict(color="green")
        )
        
        fig_hist.add_annotation(
            x=mean,
            y=y.max() * 1.1,
            text=f"Mean: {mean:.2%}",
            showarrow=False,
            font=dict(color="blue")
        )
        
        fig_hist.update_layout(
            title='Distribution of Average Annual Growth Rates',
            xaxis_title='Average Annual Growth Rate',
            yaxis_title='Frequency',
            xaxis_tickformat='.0%',
            height=500
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Boxplot
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(
            y=filtered_growth_df['avg_growth'],
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                color='blue',
                size=8,
                opacity=0.5
            ),
            name='Growth Rates',
            text=filtered_growth_df.index,
            hovertemplate='<b>%{text}</b><br>Growth: %{y:.2%}<extra></extra>'
        ))
        
        fig_box.update_layout(
            title='Boxplot of Average Annual Growth Rates',
            yaxis_title='Average Annual Growth Rate',
            yaxis_tickformat='.0%',
            height=400
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Statistical summary
        st.markdown('<p class="sub-header">Statistical Summary</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max', '10th Percentile', '90th Percentile'],
                'Growth Rate': [
                    f"{filtered_growth_df['avg_growth'].mean():.2%}",
                    f"{filtered_growth_df['avg_growth'].median():.2%}",
                    f"{filtered_growth_df['avg_growth'].std():.2%}",
                    f"{filtered_growth_df['avg_growth'].min():.2%}",
                    f"{filtered_growth_df['avg_growth'].max():.2%}",
                    f"{filtered_growth_df['avg_growth'].quantile(0.1):.2%}",
                    f"{filtered_growth_df['avg_growth'].quantile(0.9):.2%}"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            # Growth rate vs 2024 value scatter plot
            fig_scatter_value = px.scatter(
                filtered_growth_df.reset_index(),
                x='avg_growth',
                y='value_2024',
                color='classification',
                color_discrete_map=classification_colors,
                size='value_2024',
                size_max=50,
                hover_name='reporterDesc',
                hover_data={
                    'reporterDesc': False,
                    'avg_growth': ':.2%',
                    'value_2024': ':,.2f',
                    'classification': True
                },
                labels={
                    'avg_growth': 'Average Annual Growth Rate',
                    'value_2024': '2024 Export Value (USD)',
                    'classification': 'Classification'
                },
                title='Growth Rate vs 2024 Export Value'
            )
            
            fig_scatter_value.update_xaxes(tickformat='.0%')
            fig_scatter_value.update_yaxes(type='log')
            
            st.plotly_chart(fig_scatter_value, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f0f2f6; border-radius: 5px;">
        <p><b>Trade Data Analysis Dashboard</b> | Created with Streamlit</p>
        <p style="font-size: 0.8rem;">Data source: TradeData_5_10_2025_12_27_1_cleaned.csv</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Failed to load the data. Please check if the CSV file is available and properly formatted.")
    
    # Display demo mode message
    st.warning("""
    ### Demo Mode
    
    This dashboard is currently in demo mode. To use with real data:
    
    1. Upload your trade data CSV file using the sidebar uploader
    2. Ensure your data contains the required columns: refYear, reporterDesc, primaryValue, fobvalue, partnerDesc, flowDesc
    3. Refresh the page if needed after uploading
    
    For assistance, check the documentation or contact support.
    """)
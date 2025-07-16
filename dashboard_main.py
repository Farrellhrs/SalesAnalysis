# SALES ANALYTICS DASHBOARD - MAIN APPLICATION
"""
Interactive Sales Analytics Dashboard
Comprehensive dashboard for sales performance analysis and prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from dashboard_utils import DataProcessor, MetricsCalculator, ChartGenerator
from predictive_model import PredictiveModel

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    
    .metric-label {
        color: #7f8c8d;
        margin: 0;
    }
    
    .sidebar-filter {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class SalesDashboard:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.metrics_calc = MetricsCalculator()
        self.chart_gen = ChartGenerator()
        self.pred_model = None
        
    def load_data(self):
        """Load and prepare data"""
        try:
            # Load the sales data
            df = pd.read_csv(r"sales_visits_enriched_csv.csv")
            return self.data_processor.prepare_data(df)
        except FileNotFoundError:
            st.error("❌ File 'sales_visits_enriched_csv.csv' not found! Please make sure the file is in the same directory.")
            return None
    
    def render_sidebar(self, df):
        """Render sidebar filters"""
        st.sidebar.markdown("## 🔧 Filters & Controls")
        
        # Date range filter
        st.sidebar.markdown("### 📅 Date Range")
        date_range = st.sidebar.date_input(
            "Select Date Range:",
            value=(df['Tanggal'].min().date(), df['Tanggal'].max().date()),
            min_value=df['Tanggal'].min().date(),
            max_value=df['Tanggal'].max().date(),
            key="date_range"
        )
        
        # Segment filter
        st.sidebar.markdown("### 🏢 Customer Segment")
        segments = ['All Segments'] + list(df['Segmen'].unique())
        selected_segment = st.sidebar.selectbox("Select Segment:", segments)
        
        # Sales level filter
        st.sidebar.markdown("### 👨‍💼 Sales Level")
        levels = ['All Levels'] + list(df['Level_Sales'].unique())
        selected_level = st.sidebar.selectbox("Select Sales Level:", levels)
        
        # Progress stage filter
        st.sidebar.markdown("### 📈 Progress Stage")
        stages = ['All Stages'] + list(df['Progress'].unique())
        selected_stage = st.sidebar.selectbox("Select Progress Stage:", stages)
        
        # Customer status filter
        st.sidebar.markdown("### 👥 Customer Status")
        statuses = ['All Status'] + list(df['Status_Customer'].unique())
        selected_status = st.sidebar.selectbox("Select Customer Status:", statuses)
        
        # Sales person filter
        st.sidebar.markdown("### 👤 Sales Person")
        sales_people = ['All Sales'] + list(df['Nama_Sales'].unique())
        selected_sales = st.sidebar.selectbox("Select Sales Person:", sales_people)
        
        return {
            'date_range': date_range,
            'segment': selected_segment,
            'level': selected_level,
            'stage': selected_stage,
            'status': selected_status,
            'sales_person': selected_sales
        }
    
    def filter_data(self, df, filters):
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Date filter
        if len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            filtered_df = filtered_df[
                (filtered_df['Tanggal'].dt.date >= start_date) & 
                (filtered_df['Tanggal'].dt.date <= end_date)
            ]
        
        # Other filters
        if filters['segment'] != 'All Segments':
            filtered_df = filtered_df[filtered_df['Segmen'] == filters['segment']]
        
        if filters['level'] != 'All Levels':
            filtered_df = filtered_df[filtered_df['Level_Sales'] == filters['level']]
        
        if filters['stage'] != 'All Stages':
            filtered_df = filtered_df[filtered_df['Progress'] == filters['stage']]
        
        if filters['status'] != 'All Status':
            filtered_df = filtered_df[filtered_df['Status_Customer'] == filters['status']]
        
        if filters['sales_person'] != 'All Sales':
            filtered_df = filtered_df[filtered_df['Nama_Sales'] == filters['sales_person']]
        
        return filtered_df
    
    def render_metrics_cards(self, df):
        """Render key metrics cards"""
        metrics = self.metrics_calc.calculate_key_metrics(df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #3498db;">{metrics['total_visits']:,}</p>
                <p class="metric-label">Total Visits</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #27ae60;">{metrics['total_deals']:,}</p>
                <p class="metric-label">Total Deals</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #e74c3c;">{metrics['win_rate']:.1f}%</p>
                <p class="metric-label">Win Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #f39c12;">{metrics['avg_visits_to_close']:.1f}</p>
                <p class="metric-label">Avg Visits to Close</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #9b59b6;">Rp{metrics['total_deal_value']:.1f}B</p>
                <p class="metric-label">Total Deal Value</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_main_charts(self, df):
        """Render main dashboard charts"""
        st.markdown("## 📊 Performance Overview")
        
        # Row 1: Overview charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🏆 Win Rate by Segment")
            fig = self.chart_gen.create_segment_winrate_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Sales Funnel")
            fig = self.chart_gen.create_funnel_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("### 📈 Monthly Trend")
            fig = self.chart_gen.create_monthly_trend_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: Performance analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👨‍💼 Sales Performance")
            fig = self.chart_gen.create_sales_performance_scatter(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🔥 Deal Probability Heatmap")
            fig = self.chart_gen.create_probability_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_analysis(self, df):
        """Render detailed analysis tables and charts"""
        st.markdown("## 📋 Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Segment Analysis", "👨‍💼 Sales Performance", "🎯 Progress Analysis", "🔮 Predictive Model"])
        
        with tab1:
            self.render_segment_analysis(df)
        
        with tab2:
            self.render_sales_analysis(df)
        
        with tab3:
            self.render_progress_analysis(df)
        
        with tab4:
            self.render_predictive_analysis(df)
    
    def render_segment_analysis(self, df):
        """Render segment analysis"""
        st.markdown("### 🏢 Customer Segment Performance")
        
        segment_summary = self.metrics_calc.calculate_segment_metrics(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(segment_summary, use_container_width=True)
        
        with col2:
            fig = self.chart_gen.create_segment_comparison_chart(segment_summary)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_sales_analysis(self, df):
        """Render sales performance analysis"""
        st.markdown("### 👨‍💼 Individual Sales Performance")
        
        sales_summary = self.metrics_calc.calculate_sales_metrics(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top 15 Sales by Win Rate")
            st.dataframe(sales_summary.head(15), use_container_width=True)
        
        with col2:
            fig = self.chart_gen.create_sales_ranking_chart(sales_summary.head(10))
            st.plotly_chart(fig, use_container_width=True)
        
        # Sales level comparison
        st.markdown("#### AM vs EAM Performance Comparison")
        level_comparison = self.metrics_calc.calculate_level_comparison(df)
        st.dataframe(level_comparison, use_container_width=True)
    
    def render_progress_analysis(self, df):
        """Render progress stage analysis"""
        st.markdown("### 🎯 Sales Progress Analysis")
        
        progress_summary = self.metrics_calc.calculate_progress_metrics(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(progress_summary, use_container_width=True)
        
        with col2:
            fig = self.chart_gen.create_progress_conversion_chart(progress_summary)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_predictive_analysis(self, df):
        """Render predictive model interface"""
        st.markdown("### 🔮 Deal Probability Predictor")
        
        # Initialize predictive model if not already done
        if self.pred_model is None:
            with st.spinner("Loading predictive model..."):
                self.pred_model = PredictiveModel()
                self.pred_model.train(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Predict Deal Probability")
            
            # Input controls
            progress_options = df['Progress'].unique()
            selected_progress = st.selectbox("Progress Stage:", progress_options)
            
            visit_number = st.slider("Visit Number:", 1, 6, 3)
            
            segment_options = df['Segmen'].unique()
            selected_segment = st.selectbox("Customer Segment:", segment_options)
            
            customer_status_options = df['Status_Customer'].unique()
            selected_customer_status = st.selectbox("Customer Status:", customer_status_options)
            
            level_options = df['Level_Sales'].unique()
            selected_level = st.selectbox("Sales Level:", level_options)
            
            target_sales = st.number_input("Target Sales (Rp):", 100_000_000, 10_000_000_000, 1_000_000_000, step=100_000_000, format="%d")
            target_segmen = st.number_input("Target Segment (Rp):", 500_000_000, 50_000_000_000, 10_000_000_000, step=100_000_000, format="%d")
            
            # Predict button
            if st.button("🔮 Predict Deal Probability"):
                probability = self.pred_model.predict_probability(
                    progress=selected_progress,
                    visit_number=visit_number,
                    segment=selected_segment,
                    customer_status=selected_customer_status,
                    level_sales=selected_level,
                    target_sales=target_sales,
                    target_segmen=target_segmen
                )
                
                if probability is not None:
                    color = "green" if probability > 0.6 else "orange" if probability > 0.3 else "red"
                    percentage = probability * 100  # Convert to percentage
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: white; border-radius: 10px; margin: 1rem 0;">
                        <h2 style="color: {color}; margin: 0;">{percentage:.1f}%</h2>
                        <p style="color: #7f8c8d; margin: 0;">Deal Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Unable to calculate prediction")
        
        with col2:
            st.markdown("#### Model Performance")
            model_metrics = self.pred_model.get_model_metrics()
            st.json(model_metrics)
            
            st.markdown("#### Feature Importance")
            fig = self.chart_gen.create_feature_importance_chart(self.pred_model.get_feature_importance())
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main dashboard application"""
        # Header
        st.markdown('<h1 class="main-header">📊 Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### Interactive Sales Performance & Predictive Analytics")
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Sidebar filters
        filters = self.render_sidebar(df)
        
        # Apply filters
        filtered_df = self.filter_data(df, filters)
        
        # Display data info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Data Summary")
        st.sidebar.metric("Total Records", len(filtered_df))
        st.sidebar.metric("Date Range", f"{filtered_df['Tanggal'].min().strftime('%d/%m/%Y')} - {filtered_df['Tanggal'].max().strftime('%d/%m/%Y')}")
        
        # Main dashboard content
        if len(filtered_df) == 0:
            st.warning("⚠️ No data available for the selected filters. Please adjust your filter criteria.")
            return
        
        # Metrics cards
        self.render_metrics_cards(filtered_df)
        
        # Main charts
        self.render_main_charts(filtered_df)
        
        # Detailed analysis
        self.render_detailed_analysis(filtered_df)
        
        # Footer
        st.markdown("---")
        st.markdown("**📊 Sales Analytics Dashboard** | Built with Streamlit | Data-driven insights for sales optimization")

if __name__ == "__main__":
    dashboard = SalesDashboard()
    dashboard.run()

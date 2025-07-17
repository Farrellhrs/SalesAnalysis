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
            # Load the sales data with new filename
            df = pd.read_csv(r"sales_visits_finalbgt_enriched.csv")
            return self.data_processor.prepare_data(df)
        except FileNotFoundError:
            st.error("❌ File 'sales_visits_finalbgt_enriched.csv' not found! Please make sure the file is in the same directory.")
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
        
        # Deal status filter
        st.sidebar.markdown("### 💼 Deal Status")
        deal_statuses = ['All Deal Status'] + list(df['Status_Kontrak'].unique())
        selected_deal_status = st.sidebar.selectbox("Select Deal Status:", deal_statuses)
        
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
            'deal_status': selected_deal_status,
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
        
        if filters['deal_status'] != 'All Deal Status':
            filtered_df = filtered_df[filtered_df['Status_Kontrak'] == filters['deal_status']]
        
        if filters['sales_person'] != 'All Sales':
            filtered_df = filtered_df[filtered_df['Nama_Sales'] == filters['sales_person']]
        
        return filtered_df
    
    def render_metrics_cards(self, df):
        """Render key metrics cards"""
        metrics = self.metrics_calc.calculate_key_metrics(df)
        
        # Update metric layout to include AHT and Target Achievement
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        
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
                <p class="metric-label">Won Deals</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #f39c12;">{metrics['ongoing_deals']:,}</p>
                <p class="metric-label">Ongoing Deals</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #e74c3c;">{metrics['win_rate']:.1f}%</p>
                <p class="metric-label">Win Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #9b59b6;">Rp{metrics['total_deal_value']:.1f}M</p>
                <p class="metric-label">Won Value</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #17a2b8;">Rp{metrics['potential_value']:.1f}M</p>
                <p class="metric-label">Potential Value</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #6c757d;">{metrics['average_handling_time']:.1f}d</p>
                <p class="metric-label">Avg Handling Time</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: #17a2b8;">{metrics['ketercapaian_target']:.1f}%</p>
                <p class="metric-label">Target Achievement</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_main_charts(self, df):
        """Render main dashboard charts"""
        st.markdown("## 📊 Performance Overview")
        
        # Row 1: Overview charts
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 📊 Deal Status")
            fig = self.chart_gen.create_status_distribution_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🏆 Win Rate by Segment")
            fig = self.chart_gen.create_segment_winrate_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("### 🎯 Sales Funnel")
            fig = self.chart_gen.create_funnel_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
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
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Segment Analysis", "👨‍💼 Sales Performance", "🎯 Progress Analysis", "⏱️ Handling Time Analysis", "📈 Factor Analysis", "⏰ Timeline Analysis", "🔮 Predictive Model"])
        
        with tab1:
            self.render_segment_analysis(df)
        
        with tab2:
            self.render_sales_analysis(df)
        
        with tab3:
            self.render_progress_analysis(df)
        
        with tab4:
            self.render_aht_analysis(df)
        
        with tab5:
            self.render_factor_analysis(df)
        
        with tab6:
            self.render_timeline_analysis(df)
        
        with tab7:
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
    
    def render_aht_analysis(self, df):
        """Render Average Handling Time analysis"""
        st.markdown("### ⏱️ Average Handling Time Analysis")
        
        # Calculate overall AHT
        overall_aht = self.metrics_calc.calculate_average_handling_time(df)
        
        # Display overall AHT prominently
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; margin: 1rem 0; color: white;">
            <h2 style="margin: 0; font-size: 3rem;">{overall_aht:.1f} days</h2>
            <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">Overall Average Handling Time</p>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.7;">Time from first visit to latest visit per customer</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AHT breakdown charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👨‍💼 AHT by Salesperson")
            fig = self.chart_gen.create_aht_by_salesperson_chart(df, self.metrics_calc)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            aht_sales_data = self.metrics_calc.calculate_aht_by_salesperson(df)
            if not aht_sales_data.empty:
                st.dataframe(aht_sales_data, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏢 AHT by Segment")
            fig = self.chart_gen.create_aht_by_segment_chart(df, self.metrics_calc)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            aht_segment_data = self.metrics_calc.calculate_aht_by_segment(df)
            if not aht_segment_data.empty:
                st.dataframe(aht_segment_data, use_container_width=True)
        
        # Insights section
        st.markdown("#### 💡 Key Insights")
        
        aht_sales_data = self.metrics_calc.calculate_aht_by_salesperson(df)
        aht_segment_data = self.metrics_calc.calculate_aht_by_segment(df)
        
        if not aht_sales_data.empty and not aht_segment_data.empty:
            best_sales = aht_sales_data.iloc[0]
            worst_sales = aht_sales_data.iloc[-1]
            best_segment = aht_segment_data.iloc[0]
            worst_segment = aht_segment_data.iloc[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Fastest Handler:** {best_sales['Nama_Sales']} ({best_sales['Avg_Handling_Time']:.1f} days)")
                st.info(f"**Fastest Segment:** {best_segment['Segmen']} ({best_segment['Avg_Handling_Time']:.1f} days)")
            
            with col2:
                st.warning(f"**Slowest Handler:** {worst_sales['Nama_Sales']} ({worst_sales['Avg_Handling_Time']:.1f} days)")
                st.error(f"**Slowest Segment:** {worst_segment['Segmen']} ({worst_segment['Avg_Handling_Time']:.1f} days)")
    
    def render_factor_analysis(self, df):
        """Render factor analysis for key success factors"""
        st.markdown("### 📈 Key Success Factor Analysis")
        
        # Calculate factor analysis
        factor_data = self.metrics_calc.calculate_factor_analysis(df)
        
        # Generate charts
        charts = self.chart_gen.create_factor_analysis_charts(factor_data)
        
        # Introduction text
        st.markdown("""
        This analysis examines the top factors that influence deal success based on feature importance 
        from our predictive model: **Customer Status**, **Target Sales**, and **Customer Segment**.
        """)
        
        # Row 1: Status and Target Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(charts['status_chart'], use_container_width=True)
            
            # Status insights
            best_status = factor_data['status_analysis'].iloc[0]
            worst_status = factor_data['status_analysis'].iloc[-1]
            st.info(f"🎯 **Best performing status:** {best_status['Status_Customer']} ({best_status['Win_Rate']:.1f}% win rate)")
            st.warning(f"⚠️ **Lowest performing status:** {worst_status['Status_Customer']} ({worst_status['Win_Rate']:.1f}% win rate)")
        
        with col2:
            st.plotly_chart(charts['target_chart'], use_container_width=True)
            
            # Target insights
            best_target = factor_data['target_analysis'].iloc[0]
            worst_target = factor_data['target_analysis'].iloc[-1]
            st.success(f"💰 **Best target range:** {best_target['Target_Range']} ({best_target['Win_Rate']:.1f}% win rate)")
            st.error(f"📉 **Challenging target range:** {worst_target['Target_Range']} ({worst_target['Win_Rate']:.1f}% win rate)")
        
        # Row 2: Segment Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(charts['heatmap_chart'], use_container_width=True)
            
            # Show top combinations
            st.markdown("#### 🏆 Top Status-Segment Combinations")
            top_combinations = factor_data['combined_analysis'].head(5)
            for idx, row in top_combinations.iterrows():
                if row['Win_Rate'] > 0:
                    st.metric(
                        f"{row['Status_Customer']} + {row['Segmen']}", 
                        f"{row['Win_Rate']:.1f}%",
                        f"{row['Deals_Won']}/{row['Total_Customers']} deals"
                    )
        
        with col2:
            st.plotly_chart(charts['scatter_chart'], use_container_width=True)
            
            # Segment performance table
            st.markdown("#### 📊 Segment Performance Summary")
            segment_display = factor_data['segment_analysis'][['Segmen', 'Win_Rate', 'Avg_Deal_Value', 'Deals_Won']].copy()
            segment_display.columns = ['Segment', 'Win Rate (%)', 'Avg Deal Value (M)', 'Deals Won']
            st.dataframe(segment_display, use_container_width=True)
        
        # Key Insights Section
        st.markdown("### 💡 Strategic Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown("#### 👥 Customer Status Impact")
            status_insights = factor_data['status_analysis']
            for idx, row in status_insights.iterrows():
                if row['Win_Rate'] > 30:
                    st.success(f"✅ **{row['Status_Customer']}**: {row['Win_Rate']:.1f}% win rate")
                elif row['Win_Rate'] > 15:
                    st.warning(f"⚠️ **{row['Status_Customer']}**: {row['Win_Rate']:.1f}% win rate")
                else:
                    st.error(f"❌ **{row['Status_Customer']}**: {row['Win_Rate']:.1f}% win rate")
        
        with insight_col2:
            st.markdown("#### 💰 Target Sales Patterns")
            target_insights = factor_data['target_analysis']
            for idx, row in target_insights.iterrows():
                if row['Win_Rate'] > 30:
                    st.success(f"✅ **{row['Target_Range']}**: {row['Win_Rate']:.1f}% win rate")
                elif row['Win_Rate'] > 15:
                    st.warning(f"⚠️ **{row['Target_Range']}**: {row['Win_Rate']:.1f}% win rate")
                else:
                    st.error(f"❌ **{row['Target_Range']}**: {row['Win_Rate']:.1f}% win rate")
        
        with insight_col3:
            st.markdown("#### 🏢 Segment Opportunities")
            segment_insights = factor_data['segment_analysis'].head(3)
            for idx, row in segment_insights.iterrows():
                st.success(f"🎯 **{row['Segmen']}**")
                st.write(f"• Win Rate: {row['Win_Rate']:.1f}%")
                st.write(f"• Avg Value: {row['Avg_Deal_Value']:.1f}M IDR")
                st.write("---")
    
    def render_timeline_analysis(self, df):
        """Render timeline progression analysis for sales processes"""
        st.markdown("### ⏰ Sales Timeline Progression Analysis")
        st.markdown("**Analisis berapa lama menuju kontrak:** Melacak timeline proses sales dari Inisiasi hingga setiap tahap berikutnya")
        
        # Calculate timeline data
        timeline_data = self.metrics_calc.calculate_sales_timeline_analysis(df)
        progression_summary = self.metrics_calc.calculate_stage_progression_summary(df)
        
        if timeline_data.empty:
            st.warning("⚠️ No timeline data available. Need multiple visits per customer to analyze progression.")
            return
        
        # Overview metrics
        st.markdown("#### 📊 Timeline Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate overall metrics
        total_sales_analyzed = len(timeline_data)
        avg_deal_closure = timeline_data['Avg_Deal_Closure_Days'].dropna().mean()
        fastest_closer = timeline_data.dropna(subset=['Avg_Deal_Closure_Days']).nsmallest(1, 'Avg_Deal_Closure_Days')
        slowest_closer = timeline_data.dropna(subset=['Avg_Deal_Closure_Days']).nlargest(1, 'Avg_Deal_Closure_Days')
        
        with col1:
            st.metric("👥 Sales Analyzed", total_sales_analyzed)
        
        with col2:
            if not pd.isna(avg_deal_closure):
                st.metric("⏱️ Avg Deal Closure", f"{avg_deal_closure:.1f} days")
            else:
                st.metric("⏱️ Avg Deal Closure", "No data")
        
        with col3:
            if not fastest_closer.empty:
                st.metric("🚀 Fastest Closer", f"{fastest_closer.iloc[0]['Nama_Sales']}", f"{fastest_closer.iloc[0]['Avg_Deal_Closure_Days']:.1f} days")
            else:
                st.metric("🚀 Fastest Closer", "No data")
        
        with col4:
            if not slowest_closer.empty:
                st.metric("🐌 Slowest Closer", f"{slowest_closer.iloc[0]['Nama_Sales']}", f"{slowest_closer.iloc[0]['Avg_Deal_Closure_Days']:.1f} days")
            else:
                st.metric("🐌 Slowest Closer", "No data")
        
        # Charts Section
        st.markdown("#### 📈 Timeline Visualizations")
        
        # Row 1: Stage Progression Charts
        col1, col2 = st.columns(2)
        
        with col1:
            timeline_chart = self.chart_gen.create_sales_timeline_chart(timeline_data)
            st.plotly_chart(timeline_chart, use_container_width=True)
            
            # Stage progression insights
            if progression_summary:
                st.markdown("**⏱️ Average Days from Inisiasi:**")
                for stage, metrics in progression_summary.items():
                    st.write(f"• **{stage}**: {metrics['avg_days']} days (median: {metrics['median_days']})")
        
        with col2:
            closure_chart = self.chart_gen.create_deal_closure_timeline_chart(timeline_data)
            st.plotly_chart(closure_chart, use_container_width=True)
            
            # Deal closure insights
            closure_data = timeline_data.dropna(subset=['Avg_Deal_Closure_Days'])
            if not closure_data.empty:
                st.markdown("**🎯 Deal Closure Performance:**")
                fast_closers = closure_data.nsmallest(3, 'Avg_Deal_Closure_Days')
                for idx, row in fast_closers.iterrows():
                    st.success(f"✅ **{row['Nama_Sales']}**: {row['Avg_Deal_Closure_Days']:.1f} days")
        
        # Row 2: Heatmap and Detailed Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            heatmap_chart = self.chart_gen.create_stage_progression_heatmap(timeline_data)
            st.plotly_chart(heatmap_chart, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Detailed Timeline Data")
            
            # Display timeline data table with available columns
            base_cols = ['Nama_Sales', 'Level_Sales', 'Total_Customers', 'Avg_Deal_Closure_Days']
            stage_cols = [col for col in timeline_data.columns if col.startswith('Days_to_')]
            
            display_cols = []
            for col in base_cols:
                if col in timeline_data.columns:
                    display_cols.append(col)
            
            # Add up to 3 stage columns
            display_cols.extend(stage_cols[:3])
            
            display_data = timeline_data[display_cols].copy()
            
            # Rename columns for better display
            column_mapping = {
                'Nama_Sales': 'Salesperson',
                'Level_Sales': 'Level',
                'Total_Customers': 'Customers',
                'Avg_Deal_Closure_Days': 'Avg Closure (days)'
            }
            
            # Add dynamic mappings for stage columns
            for col in stage_cols[:3]:
                if col in display_data.columns:
                    stage_name = col.replace('Days_to_', '').replace('_', ' ')
                    column_mapping[col] = f'To {stage_name}'
            
            display_data = display_data.rename(columns=column_mapping)
            st.dataframe(display_data, use_container_width=True)
        
        # Performance Insights Section
        st.markdown("#### 💡 Timeline Performance Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown("#### 🏃‍♂️ Speed Champions")
            # Check if Presentasi column exists (with correct name)
            presentasi_col = None
            for col in timeline_data.columns:
                if 'Presentasi' in col and col.startswith('Days_to_'):
                    presentasi_col = col
                    break
            
            if presentasi_col and not timeline_data[presentasi_col].isna().all():
                fast_progression = timeline_data.dropna(subset=[presentasi_col]).nsmallest(3, presentasi_col)
                for idx, row in fast_progression.iterrows():
                    st.success(f"⚡ **{row['Nama_Sales']}**: {row[presentasi_col]:.1f} days to Presentasi")
            else:
                st.info("No Presentasi progression data available")
        
        with insight_col2:
            st.markdown("#### 🎯 Consistent Performers")
            # Find salespeople with good progression across multiple stages
            # Check which columns actually exist before using them
            available_stage_cols = [col for col in timeline_data.columns if col.startswith('Days_to_')]
            
            if len(available_stage_cols) >= 2:
                # Use the first two available stage columns for consistency analysis
                consistent_performers = timeline_data.dropna(subset=available_stage_cols[:2])
                if not consistent_performers.empty:
                    # Calculate average progression across available stages
                    consistent_performers['Avg_Progression'] = consistent_performers[available_stage_cols[:2]].mean(axis=1)
                    top_consistent = consistent_performers.nsmallest(3, 'Avg_Progression')
                    for idx, row in top_consistent.iterrows():
                        st.info(f"🎯 **{row['Nama_Sales']}**: {row['Avg_Progression']:.1f} avg days per stage")
                else:
                    st.info("No data available for consistency analysis")
            else:
                st.info("Insufficient stage data for consistency analysis")
        
        with insight_col3:
            st.markdown("#### 📈 Improvement Opportunities")
            slow_progression = timeline_data.dropna(subset=['Avg_Deal_Closure_Days']).nlargest(3, 'Avg_Deal_Closure_Days')
            for idx, row in slow_progression.iterrows():
                st.warning(f"📈 **{row['Nama_Sales']}**: {row['Avg_Deal_Closure_Days']:.1f} days closure time")
    
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
            
            nilai_kontrak = st.number_input("Contract Value (Rp):", 10_000_000, 5_000_000_000, 100_000_000, step=10_000_000, format="%d")
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
                    nilai_kontrak=nilai_kontrak,
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
            if model_metrics:
                st.json(model_metrics)
            else:
                st.info("Model metrics not available - insufficient data for training")
            
            st.markdown("#### Feature Importance")
            feature_importance = self.pred_model.get_feature_importance()
            fig = self.chart_gen.create_feature_importance_chart(feature_importance)
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

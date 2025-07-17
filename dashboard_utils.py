# DASHBOARD UTILITIES
"""
Utility classes for data processing, metrics calculation, and chart generation
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataProcessor:
    """Data processing utilities"""
    
    def prepare_data(self, df):
        """Prepare data for dashboard"""
        # Convert date column
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        
        # Create additional columns
        df['Month'] = df['Tanggal'].dt.strftime('%Y-%m')
        df['Deal_Binary'] = (df['Status_Kontrak'] == 'Deal').astype(int)
        df['Quarter'] = df['Tanggal'].dt.quarter
        df['Week'] = df['Tanggal'].dt.isocalendar().week
        
        # Handle new data structure with Nilai_Kontrak
        # Scale monetary values to millions for better readability
        df['Nilai_Kontrak_Millions'] = df['Nilai_Kontrak'] / 1e6
        df['Target_Sales_Millions'] = df['Target_Sales'] / 1e6
        df['Target_Segmen_Millions'] = df['Target_Segmen'] / 1e6
        
        # Map status kontrak for clarity
        df['Status_Kontrak_Label'] = df['Status_Kontrak'].map({
            'Berpotensi Deal': 'Ongoing',
            'Deal': 'Won',
            'Cancel': 'Lost'
        })
        
        return df

class MetricsCalculator:
    """Calculate various business metrics"""
    
    def calculate_key_metrics(self, df):
        """Calculate key performance metrics"""
        total_visits = len(df)
        # Count unique customers who have deals (not total deal visits)
        total_deals = df[df['Status_Kontrak'] == 'Deal']['ID_Customer'].nunique()
        win_rate = (total_deals / df['ID_Customer'].nunique() * 100) if df['ID_Customer'].nunique() > 0 else 0
        
        deal_df = df[df['Status_Kontrak'] == 'Deal']
        avg_visits_to_close = deal_df['Kunjungan_Ke'].mean() if len(deal_df) > 0 else 0
        # Use Nilai_Kontrak for actual deal value
        total_deal_value = deal_df['Nilai_Kontrak'].sum() / 1e6 if len(deal_df) > 0 else 0
        
        # Calculate additional metrics
        # Count unique customers for ongoing and cancelled deals (not total visits)
        ongoing_deals = df[df['Status_Kontrak'] == 'Berpotensi Deal']['ID_Customer'].nunique()
        cancelled_deals = df[df['Status_Kontrak'] == 'Cancel']['ID_Customer'].nunique()
        
        # Potential value from ongoing deals (take max contract value per customer to avoid double counting)
        ongoing_df = df[df['Status_Kontrak'] == 'Berpotensi Deal']
        if len(ongoing_df) > 0:
            potential_value = ongoing_df.groupby('ID_Customer')['Nilai_Kontrak'].max().sum() / 1e6
        else:
            potential_value = 0
        
        # Calculate Average Handling Time (AHT)
        aht = self.calculate_average_handling_time(df)
        
        # Calculate Ketercapaian Target (Deal Achievement Rate)
        ketercapaian_target = self.calculate_ketercapaian_target(df)
        
        return {
            'total_visits': total_visits,
            'total_deals': total_deals,
            'win_rate': win_rate,
            'avg_visits_to_close': avg_visits_to_close,
            'total_deal_value': total_deal_value,
            'ongoing_deals': ongoing_deals,
            'cancelled_deals': cancelled_deals,
            'potential_value': potential_value,
            'average_handling_time': aht,
            'ketercapaian_target': ketercapaian_target
        }
    
    def calculate_average_handling_time(self, df):
        """Calculate Average Handling Time (AHT) for customers"""
        # Group by customer and calculate date range for each customer
        customer_aht = df.groupby('ID_Customer')['Tanggal'].agg(['min', 'max']).reset_index()
        
        # Calculate days between first and last visit for each customer
        customer_aht['handling_days'] = (customer_aht['max'] - customer_aht['min']).dt.days
        
        # Calculate overall average handling time
        overall_aht = customer_aht['handling_days'].mean()
        
        return overall_aht if not pd.isna(overall_aht) else 0
    
    def calculate_aht_by_salesperson(self, df):
        """Calculate AHT breakdown by salesperson"""
        # Get the primary salesperson for each customer (most frequent)
        customer_sales = df.groupby('ID_Customer')['Nama_Sales'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).reset_index()
        customer_sales.columns = ['ID_Customer', 'Primary_Sales']
        
        # Calculate AHT per customer
        customer_aht = df.groupby('ID_Customer')['Tanggal'].agg(['min', 'max']).reset_index()
        customer_aht['handling_days'] = (customer_aht['max'] - customer_aht['min']).dt.days
        
        # Merge with salesperson data
        aht_by_sales = customer_aht.merge(customer_sales, on='ID_Customer')
        
        # Calculate average AHT per salesperson
        sales_aht = aht_by_sales.groupby('Primary_Sales')['handling_days'].mean().reset_index()
        sales_aht.columns = ['Nama_Sales', 'Avg_Handling_Time']
        sales_aht['Avg_Handling_Time'] = sales_aht['Avg_Handling_Time'].round(1)
        
        return sales_aht.sort_values('Avg_Handling_Time')
    
    def calculate_aht_by_segment(self, df):
        """Calculate AHT breakdown by segment"""
        # Get the primary segment for each customer (most frequent)
        customer_segment = df.groupby('ID_Customer')['Segmen'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).reset_index()
        customer_segment.columns = ['ID_Customer', 'Primary_Segment']
        
        # Calculate AHT per customer
        customer_aht = df.groupby('ID_Customer')['Tanggal'].agg(['min', 'max']).reset_index()
        customer_aht['handling_days'] = (customer_aht['max'] - customer_aht['min']).dt.days
        
        # Merge with segment data
        aht_by_segment = customer_aht.merge(customer_segment, on='ID_Customer')
        
        # Calculate average AHT per segment
        segment_aht = aht_by_segment.groupby('Primary_Segment')['handling_days'].mean().reset_index()
        segment_aht.columns = ['Segmen', 'Avg_Handling_Time']
        segment_aht['Avg_Handling_Time'] = segment_aht['Avg_Handling_Time'].round(1)
        
        return segment_aht.sort_values('Avg_Handling_Time')
    
    def calculate_sales_timeline_analysis(self, df):
        """Calculate timeline progression analysis for each salesperson"""
        # Define the standard sales progression stages
        stage_order = ['Inisiasi', 'Presentasi', 'Penawaran Harga', 'Negosiasi', 'Paska Deal']
        
        timeline_results = []
        
        # Get unique salespeople
        salespeople = df['Nama_Sales'].unique()
        
        for sales in salespeople:
            sales_df = df[df['Nama_Sales'] == sales].copy()
            
            # Get customers handled by this salesperson
            customers = sales_df['ID_Customer'].unique()
            
            stage_times = {stage: [] for stage in stage_order}
            stage_transitions = {}
            
            for customer in customers:
                customer_df = sales_df[sales_df['ID_Customer'] == customer].sort_values('Tanggal')
                
                if len(customer_df) < 2:  # Skip single-visit customers
                    continue
                
                # Track progression through stages
                customer_stages = customer_df['Progress'].tolist()
                customer_dates = customer_df['Tanggal'].tolist()
                
                # Calculate time from first visit (Inisiasi) to each subsequent stage
                if 'Inisiasi' in customer_stages:
                    inisiasi_date = customer_dates[customer_stages.index('Inisiasi')]
                    
                    for i, stage in enumerate(customer_stages):
                        if stage != 'Inisiasi':
                            days_from_inisiasi = (customer_dates[i] - inisiasi_date).days
                            stage_times[stage].append(days_from_inisiasi)
                
                # Calculate stage-to-stage transitions
                for i in range(len(customer_stages) - 1):
                    from_stage = customer_stages[i]
                    to_stage = customer_stages[i + 1]
                    days_transition = (customer_dates[i + 1] - customer_dates[i]).days
                    
                    transition_key = f"{from_stage} → {to_stage}"
                    if transition_key not in stage_transitions:
                        stage_transitions[transition_key] = []
                    stage_transitions[transition_key].append(days_transition)
            
            # Calculate averages for this salesperson
            sales_timeline = {
                'Nama_Sales': sales,
                'Level_Sales': sales_df['Level_Sales'].iloc[0],
                'Total_Customers': len(customers)
            }
            
            # Average days from Inisiasi to each stage
            for stage in stage_order[1:]:  # Skip Inisiasi itself
                if stage_times[stage]:
                    avg_days = np.mean(stage_times[stage])
                    sales_timeline[f'Days_to_{stage}'] = round(avg_days, 1)
                else:
                    sales_timeline[f'Days_to_{stage}'] = None
            
            # Average transition times
            for transition, times in stage_transitions.items():
                if times:
                    avg_transition = np.mean(times)
                    clean_transition = transition.replace(' → ', '_to_').replace(' ', '_')
                    sales_timeline[f'Transition_{clean_transition}'] = round(avg_transition, 1)
            
            # Calculate deal closure metrics
            deal_customers = sales_df[sales_df['Status_Kontrak'] == 'Deal']['ID_Customer'].unique()
            if len(deal_customers) > 0:
                deal_closure_times = []
                for customer in deal_customers:
                    customer_df = sales_df[sales_df['ID_Customer'] == customer].sort_values('Tanggal')
                    if len(customer_df) > 1:
                        total_days = (customer_df['Tanggal'].max() - customer_df['Tanggal'].min()).days
                        deal_closure_times.append(total_days)
                
                if deal_closure_times:
                    sales_timeline['Avg_Deal_Closure_Days'] = round(np.mean(deal_closure_times), 1)
                    sales_timeline['Min_Deal_Closure_Days'] = min(deal_closure_times)
                    sales_timeline['Max_Deal_Closure_Days'] = max(deal_closure_times)
                else:
                    sales_timeline['Avg_Deal_Closure_Days'] = None
                    sales_timeline['Min_Deal_Closure_Days'] = None
                    sales_timeline['Max_Deal_Closure_Days'] = None
            else:
                sales_timeline['Avg_Deal_Closure_Days'] = None
                sales_timeline['Min_Deal_Closure_Days'] = None
                sales_timeline['Max_Deal_Closure_Days'] = None
            
            timeline_results.append(sales_timeline)
        
        return pd.DataFrame(timeline_results)
    
    def calculate_stage_progression_summary(self, df):
        """Calculate overall stage progression patterns"""
        stage_order = ['Inisiasi', 'Presentasi', 'Penawaran Harga', 'Negosiasi', 'Paska Deal']
        
        # Overall progression times from Inisiasi
        progression_summary = {}
        
        customers = df['ID_Customer'].unique()
        
        for stage in stage_order[1:]:  # Skip Inisiasi itself
            stage_times = []
            
            for customer in customers:
                customer_df = df[df['ID_Customer'] == customer].sort_values('Tanggal')
                customer_stages = customer_df['Progress'].tolist()
                customer_dates = customer_df['Tanggal'].tolist()
                
                if 'Inisiasi' in customer_stages and stage in customer_stages:
                    inisiasi_date = customer_dates[customer_stages.index('Inisiasi')]
                    stage_date = customer_dates[customer_stages.index(stage)]
                    days_diff = (stage_date - inisiasi_date).days
                    stage_times.append(days_diff)
            
            if stage_times:
                progression_summary[stage] = {
                    'avg_days': round(np.mean(stage_times), 1),
                    'median_days': round(np.median(stage_times), 1),
                    'min_days': min(stage_times),
                    'max_days': max(stage_times),
                    'count': len(stage_times)
                }
        
        return progression_summary
    
    def calculate_ketercapaian_target(self, df):
        """Calculate target achievement rate (Deal Value / Target Sales)"""
        # Get deals only
        deal_df = df[df['Status_Kontrak'] == 'Deal']
        
        if len(deal_df) == 0:
            return 0
        
        # Calculate total deal value and target sales for deals
        total_deal_value = deal_df['Nilai_Kontrak'].sum()
        total_target_sales = deal_df['Target_Sales'].sum()
        
        # Calculate achievement rate as percentage
        if total_target_sales > 0:
            ketercapaian = (total_deal_value / total_target_sales) * 100
        else:
            ketercapaian = 0
            
        return ketercapaian
    
    def calculate_factor_analysis(self, df):
        """Analyze key factors affecting deal success"""
        analysis_results = {}
        
        # 1. Status Customer Analysis
        status_analysis = df.groupby('Status_Customer').agg({
            'ID_Customer': 'nunique',
            'Status_Kontrak': lambda x: (x == 'Deal').sum()
        }).reset_index()
        status_analysis.columns = ['Status_Customer', 'Total_Customers', 'Deals_Won']
        status_analysis['Win_Rate'] = (status_analysis['Deals_Won'] / status_analysis['Total_Customers'] * 100).round(1)
        status_analysis = status_analysis.sort_values('Win_Rate', ascending=False)
        
        # 2. Target Sales Range Analysis
        df_copy = df.copy()
        # Create target sales ranges for analysis
        df_copy['Target_Range'] = pd.cut(df_copy['Target_Sales'], 
                                       bins=[0, 800000000, 1000000000, 1300000000, float('inf')],
                                       labels=['< 800M', '800M-1B', '1B-1.3B', '> 1.3B'])
        
        target_analysis = df_copy.groupby('Target_Range').agg({
            'ID_Customer': 'nunique',
            'Status_Kontrak': lambda x: (x == 'Deal').sum()
        }).reset_index()
        target_analysis.columns = ['Target_Range', 'Total_Customers', 'Deals_Won']
        target_analysis['Win_Rate'] = (target_analysis['Deals_Won'] / target_analysis['Total_Customers'] * 100).round(1)
        target_analysis = target_analysis.sort_values('Win_Rate', ascending=False)
        
        # 3. Segment Analysis
        segment_analysis = df.groupby('Segmen').agg({
            'ID_Customer': 'nunique',
            'Status_Kontrak': lambda x: (x == 'Deal').sum(),
            'Nilai_Kontrak': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum()
        }).reset_index()
        segment_analysis.columns = ['Segmen', 'Total_Customers', 'Deals_Won', 'Deal_Value']
        segment_analysis['Win_Rate'] = (segment_analysis['Deals_Won'] / segment_analysis['Total_Customers'] * 100).round(1)
        segment_analysis['Avg_Deal_Value'] = (segment_analysis['Deal_Value'] / segment_analysis['Deals_Won'] / 1e6).round(2)
        segment_analysis = segment_analysis.sort_values('Win_Rate', ascending=False)
        
        # 4. Combined Factor Analysis (Status + Segment)
        combined_analysis = df.groupby(['Status_Customer', 'Segmen']).agg({
            'ID_Customer': 'nunique',
            'Status_Kontrak': lambda x: (x == 'Deal').sum()
        }).reset_index()
        combined_analysis.columns = ['Status_Customer', 'Segmen', 'Total_Customers', 'Deals_Won']
        combined_analysis['Win_Rate'] = (combined_analysis['Deals_Won'] / combined_analysis['Total_Customers'] * 100).round(1)
        combined_analysis = combined_analysis.sort_values('Win_Rate', ascending=False)
        
        return {
            'status_analysis': status_analysis,
            'target_analysis': target_analysis,
            'segment_analysis': segment_analysis,
            'combined_analysis': combined_analysis
        }
    
    def calculate_segment_metrics(self, df):
        """Calculate segment-wise performance metrics"""
        segment_summary = df.groupby('Segmen').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()],
            'Nilai_Kontrak': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum(),
            'Target_Sales': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum(),
            'Kunjungan_Ke': 'mean'
        }).round(2)
        
        segment_summary.columns = ['Total_Visits', 'Total_Deals', 'Deal_Value', 'Target_Sales_Total', 'Avg_Visits']
        segment_summary['Win_Rate'] = (segment_summary['Total_Deals'] / segment_summary['Total_Visits'] * 100).round(1)
        segment_summary['Deal_Value_Millions'] = (segment_summary['Deal_Value'] / 1e6).round(2)
        
        # Calculate Target Achievement by segment
        segment_summary['Target_Achievement'] = ((segment_summary['Deal_Value'] / segment_summary['Target_Sales_Total']) * 100).round(1)
        segment_summary['Target_Achievement'] = segment_summary['Target_Achievement'].fillna(0)
        
        # Calculate Average Handling Time by segment
        segment_aht = self.calculate_aht_by_segment(df)
        aht_dict = dict(zip(segment_aht['Segmen'], segment_aht['Avg_Handling_Time']))
        
        segment_summary = segment_summary.reset_index()
        segment_summary['Avg_Handling_Time'] = segment_summary['Segmen'].map(aht_dict).fillna(0)
        segment_summary = segment_summary.sort_values('Win_Rate', ascending=False)
        
        return segment_summary[['Segmen', 'Total_Visits', 'Total_Deals', 'Win_Rate', 'Deal_Value_Millions', 'Target_Achievement', 'Avg_Handling_Time', 'Avg_Visits']]
    
    def calculate_sales_metrics(self, df):
        """Calculate individual sales performance metrics"""
        sales_summary = df.groupby('Nama_Sales').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()],
            'Level_Sales': 'first',
            'Nilai_Kontrak': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum(),
            'Target_Sales': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum(),
            'Segmen': lambda x: ', '.join(x.unique()[:3])  # Top 3 segments
        }).round(2)
        
        sales_summary.columns = ['Total_Visits', 'Total_Deals', 'Level', 'Deal_Value', 'Target_Sales_Total', 'Segments']
        sales_summary['Win_Rate'] = (sales_summary['Total_Deals'] / sales_summary['Total_Visits'] * 100).round(1)
        sales_summary['Deal_Value_Millions'] = (sales_summary['Deal_Value'] / 1e6).round(2)
        
        # Calculate Target Achievement for each salesperson
        sales_summary['Target_Achievement'] = ((sales_summary['Deal_Value'] / sales_summary['Target_Sales_Total']) * 100).round(1)
        sales_summary['Target_Achievement'] = sales_summary['Target_Achievement'].fillna(0)
        
        # Calculate Average Handling Time for each salesperson
        aht_data = self.calculate_aht_by_salesperson(df)
        aht_dict = dict(zip(aht_data['Nama_Sales'], aht_data['Avg_Handling_Time']))
        
        sales_summary = sales_summary.reset_index()
        sales_summary['Avg_Handling_Time'] = sales_summary['Nama_Sales'].map(aht_dict).fillna(0).round(1)
        
        sales_summary = sales_summary.sort_values('Win_Rate', ascending=False)
        
        return sales_summary[['Nama_Sales', 'Level', 'Total_Visits', 'Total_Deals', 'Win_Rate', 'Deal_Value_Millions', 'Target_Achievement', 'Avg_Handling_Time', 'Segments']]
    
    def calculate_level_comparison(self, df):
        """Compare AM vs EAM performance"""
        level_comparison = df.groupby('Level_Sales').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()],
            'Nama_Sales': 'nunique',
            'Nilai_Kontrak': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum(),
            'Target_Sales': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum(),
            'Kunjungan_Ke': 'mean'
        }).round(2)
        
        level_comparison.columns = ['Total_Visits', 'Total_Deals', 'Unique_Sales', 'Deal_Value', 'Target_Sales_Total', 'Avg_Visits']
        level_comparison['Win_Rate'] = (level_comparison['Total_Deals'] / level_comparison['Total_Visits'] * 100).round(1)
        level_comparison['Deals_per_Sales'] = (level_comparison['Total_Deals'] / level_comparison['Unique_Sales']).round(1)
        level_comparison['Visits_per_Sales'] = (level_comparison['Total_Visits'] / level_comparison['Unique_Sales']).round(1)
        level_comparison['Deal_Value_Millions'] = (level_comparison['Deal_Value'] / 1e6).round(2)
        
        # Calculate Target Achievement by level
        level_comparison['Target_Achievement'] = ((level_comparison['Deal_Value'] / level_comparison['Target_Sales_Total']) * 100).round(1)
        level_comparison['Target_Achievement'] = level_comparison['Target_Achievement'].fillna(0)
        
        # Calculate Average Handling Time by level
        level_aht = df.groupby(['Level_Sales', 'ID_Customer'])['Tanggal'].agg(['min', 'max']).reset_index()
        level_aht['handling_days'] = (level_aht['max'] - level_aht['min']).dt.days
        level_aht_avg = level_aht.groupby('Level_Sales')['handling_days'].mean().round(1)
        
        level_comparison = level_comparison.reset_index()
        level_comparison['Avg_Handling_Time'] = level_comparison['Level_Sales'].map(level_aht_avg).fillna(0)
        
        return level_comparison[['Level_Sales', 'Total_Visits', 'Total_Deals', 'Unique_Sales', 'Win_Rate', 'Deal_Value_Millions', 'Target_Achievement', 'Avg_Handling_Time', 'Deals_per_Sales', 'Visits_per_Sales']]
    
    def calculate_progress_metrics(self, df):
        """Calculate progress stage metrics"""
        progress_summary = df.groupby('Progress').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()],
            'Kunjungan_Ke': 'mean',
            'Nilai_Kontrak': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum()
        }).round(2)
        
        progress_summary.columns = ['Total_Visits', 'Total_Deals', 'Avg_Visit_Number', 'Deal_Value']
        progress_summary['Conversion_Rate'] = (progress_summary['Total_Deals'] / progress_summary['Total_Visits'] * 100).round(1)
        progress_summary['Deal_Value_Millions'] = (progress_summary['Deal_Value'] / 1e6).round(2)
        
        # Reorder by logical progression
        stage_order = ['Inisiasi', 'Presentasi', 'Penawaran Harga', 'Negosiasi', 'Paska Deal']
        progress_summary = progress_summary.reindex([stage for stage in stage_order if stage in progress_summary.index])
        
        return progress_summary.reset_index()

class ChartGenerator:
    """Generate various charts for dashboard"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#9b59b6'
        }
    
    def create_segment_winrate_chart(self, df):
        """Create segment win rate bar chart"""
        segment_data = df.groupby('Segmen').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()]
        })
        segment_data.columns = ['Total_Visits', 'Total_Deals']
        segment_data['Win_Rate'] = (segment_data['Total_Deals'] / segment_data['Total_Visits'] * 100).round(1)
        segment_data = segment_data.reset_index().sort_values('Win_Rate', ascending=False)
        
        fig = px.bar(
            segment_data, 
            x='Segmen', 
            y='Win_Rate',
            color='Win_Rate',
            color_continuous_scale='RdYlGn',
            title='Win Rate by Customer Segment'
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Customer Segment",
            yaxis_title="Win Rate (%)",
            font=dict(size=12)
        )
        
        return fig
    
    def create_funnel_chart(self, df):
        """Create sales funnel chart"""
        # Calculate funnel data
        stage_order = ['Inisiasi', 'Presentasi', 'Penawaran Harga', 'Negosiasi', 'Paska Deal']
        funnel_data = []
        
        for stage in stage_order:
            stage_df = df[df['Progress'] == stage]
            if len(stage_df) > 0:
                funnel_data.append({
                    'Stage': stage,
                    'Count': len(stage_df),
                    'Deals': stage_df['Deal_Binary'].sum()
                })
        
        funnel_df = pd.DataFrame(funnel_data)
        
        fig = go.Figure(go.Funnel(
            y=funnel_df['Stage'],
            x=funnel_df['Count'],
            textinfo="value+percent initial",
            marker=dict(color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60'])
        ))
        
        fig.update_layout(
            title='Sales Funnel Analysis',
            font=dict(size=12)
        )
        
        return fig
    
    def create_monthly_trend_chart(self, df):
        """Create monthly trend chart"""
        monthly_data = df.groupby('Month').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()]
        }).reset_index()
        monthly_data.columns = ['Month', 'Total_Visits', 'Total_Deals']
        monthly_data['Win_Rate'] = (monthly_data['Total_Deals'] / monthly_data['Total_Visits'] * 100).round(1)
        
        fig = go.Figure()
        
        # Add win rate line
        fig.add_trace(go.Scatter(
            x=monthly_data['Month'],
            y=monthly_data['Win_Rate'],
            mode='lines+markers',
            name='Win Rate (%)',
            line=dict(color=self.color_palette['primary'], width=3),
            marker=dict(size=8)
        ))
        
        # Add visits bar
        fig.add_trace(go.Bar(
            x=monthly_data['Month'],
            y=monthly_data['Total_Visits'],
            name='Total Visits',
            yaxis='y2',
            opacity=0.6,
            marker_color=self.color_palette['info']
        ))
        
        fig.update_layout(
            title='Monthly Performance Trend',
            xaxis_title='Month',
            yaxis=dict(title='Win Rate (%)', side='left'),
            yaxis2=dict(title='Total Visits', side='right', overlaying='y'),
            legend=dict(x=0.01, y=0.99),
            font=dict(size=12)
        )
        
        return fig
    
    def create_sales_performance_scatter(self, df):
        """Create sales performance scatter plot"""
        sales_data = df.groupby('Nama_Sales').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()],
            'Level_Sales': 'first',
            'Nilai_Kontrak': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum()
        }).round(2)
        
        sales_data.columns = ['Total_Visits', 'Total_Deals', 'Level', 'Deal_Value']
        sales_data['Win_Rate'] = (sales_data['Total_Deals'] / sales_data['Total_Visits'] * 100).round(1)
        sales_data['Deal_Value_Millions'] = (sales_data['Deal_Value'] / 1e6).round(2)
        sales_data = sales_data.reset_index()
        
        fig = px.scatter(
            sales_data.head(20),  # Top 20 performers
            x='Total_Visits',
            y='Total_Deals',
            size='Deal_Value_Millions',
            color='Win_Rate',
            hover_name='Nama_Sales',
            hover_data=['Level'],
            color_continuous_scale='RdYlGn',
            title='Sales Performance: Visits vs Deals (Deal Value in Millions IDR)'
        )
        
        fig.update_layout(
            xaxis_title="Total Visits",
            yaxis_title="Total Deals",
            font=dict(size=12)
        )
        
        return fig
    
    def create_probability_heatmap(self, df):
        """Create deal probability heatmap"""
        # Create pivot table for heatmap
        heatmap_data = df.pivot_table(
            values='Deal_Binary',
            index='Progress',
            columns='Kunjungan_Ke',
            aggfunc='mean'
        ).fillna(0)
        
        fig = px.imshow(
            heatmap_data,
            title='Deal Probability: Progress Stage vs Visit Number',
            color_continuous_scale='RdYlGn',
            aspect='auto',
            labels=dict(x="Visit Number", y="Progress Stage", color="Deal Probability")
        )
        
        fig.update_layout(font=dict(size=12))
        
        return fig
    
    def create_segment_comparison_chart(self, segment_summary):
        """Create segment comparison chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Win Rate', 'Total Visits', 'Total Deals', 'Deal Value (Millions IDR)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Win Rate
        fig.add_trace(
            go.Bar(x=segment_summary['Segmen'], y=segment_summary['Win_Rate'], 
                   name='Win Rate', marker_color=self.color_palette['success']),
            row=1, col=1
        )
        
        # Total Visits
        fig.add_trace(
            go.Bar(x=segment_summary['Segmen'], y=segment_summary['Total_Visits'], 
                   name='Total Visits', marker_color=self.color_palette['primary']),
            row=1, col=2
        )
        
        # Total Deals
        fig.add_trace(
            go.Bar(x=segment_summary['Segmen'], y=segment_summary['Total_Deals'], 
                   name='Total Deals', marker_color=self.color_palette['warning']),
            row=2, col=1
        )
        
        # Deal Value
        fig.add_trace(
            go.Bar(x=segment_summary['Segmen'], y=segment_summary['Deal_Value_Millions'], 
                   name='Deal Value (Millions IDR)', marker_color=self.color_palette['danger']),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Segment Performance Comparison',
            showlegend=False,
            height=600,
            font=dict(size=12)
        )
        
        return fig
    
    def create_sales_ranking_chart(self, sales_summary):
        """Create sales ranking chart"""
        fig = px.bar(
            sales_summary,
            x='Win_Rate',
            y='Nama_Sales',
            orientation='h',
            color='Level',
            title='Top 10 Sales by Win Rate',
            color_discrete_map={'AM': self.color_palette['primary'], 'EAM': self.color_palette['success']}
        )
        
        fig.update_layout(
            xaxis_title="Win Rate (%)",
            yaxis_title="Sales Person",
            yaxis={'categoryorder': 'total ascending'},
            font=dict(size=12)
        )
        
        return fig
    
    def create_progress_conversion_chart(self, progress_summary):
        """Create progress conversion chart"""
        fig = go.Figure()
        
        # Add conversion rate bars
        fig.add_trace(go.Bar(
            x=progress_summary['Progress'],
            y=progress_summary['Conversion_Rate'],
            name='Conversion Rate (%)',
            marker_color=self.color_palette['success']
        ))
        
        # Add average visit number line
        fig.add_trace(go.Scatter(
            x=progress_summary['Progress'],
            y=progress_summary['Avg_Visit_Number'] * 20,  # Scale for visibility
            mode='lines+markers',
            name='Avg Visit Number (×20)',
            yaxis='y2',
            line=dict(color=self.color_palette['danger'], width=3)
        ))
        
        fig.update_layout(
            title='Progress Stage Analysis',
            xaxis_title='Progress Stage',
            yaxis=dict(title='Conversion Rate (%)', side='left'),
            yaxis2=dict(title='Average Visit Number', side='right', overlaying='y'),
            font=dict(size=12)
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance):
        """Create feature importance chart"""
        # Handle case when feature_importance is None or empty
        if feature_importance is None or feature_importance.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No feature importance data available<br>Model may need more data for training",
                xref="paper", yref="paper",
                x=0.5, y=0.5, 
                showarrow=False,
                font=dict(size=14),
                align="center"
            )
            fig.update_layout(
                title='Feature Importance (Predictive Model)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400,
                font=dict(size=12)
            )
            return fig
        
        # Create the actual chart if data is available
        fig = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Feature Importance (Predictive Model)',
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Features",
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            font=dict(size=12)
        )
        
        return fig

    def create_factor_analysis_charts(self, factor_data):
        """Create comprehensive factor analysis charts"""
        
        # 1. Status Customer Impact Chart
        status_fig = px.bar(
            factor_data['status_analysis'],
            x='Status_Customer',
            y='Win_Rate',
            color='Win_Rate',
            color_continuous_scale='RdYlGn',
            title='Win Rate by Customer Status',
            labels={'Win_Rate': 'Win Rate (%)', 'Status_Customer': 'Customer Status'},
            text='Win_Rate'
        )
        status_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        status_fig.update_layout(height=400, showlegend=False)
        
        # 2. Target Sales Range Impact Chart
        target_fig = px.bar(
            factor_data['target_analysis'],
            x='Target_Range',
            y='Win_Rate',
            color='Win_Rate',
            color_continuous_scale='RdYlGn',
            title='Win Rate by Target Sales Range',
            labels={'Win_Rate': 'Win Rate (%)', 'Target_Range': 'Target Sales Range'},
            text='Win_Rate'
        )
        target_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        target_fig.update_layout(height=400, showlegend=False)
        
        # 3. Segment Performance Heatmap
        segment_pivot = factor_data['combined_analysis'].pivot(
            index='Status_Customer', 
            columns='Segmen', 
            values='Win_Rate'
        ).fillna(0)
        
        heatmap_fig = px.imshow(
            segment_pivot,
            title='Win Rate Heatmap: Customer Status vs Segment',
            labels=dict(x="Segment", y="Customer Status", color="Win Rate (%)"),
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        heatmap_fig.update_layout(height=400)
        
        # 4. Segment Value vs Win Rate Scatter
        scatter_fig = px.scatter(
            factor_data['segment_analysis'],
            x='Win_Rate',
            y='Avg_Deal_Value',
            size='Total_Customers',
            color='Segmen',
            title='Segment Performance: Win Rate vs Average Deal Value',
            labels={'Win_Rate': 'Win Rate (%)', 'Avg_Deal_Value': 'Avg Deal Value (M IDR)'},
            hover_data=['Deals_Won']
        )
        scatter_fig.update_layout(height=400)
        
        return {
            'status_chart': status_fig,
            'target_chart': target_fig,
            'heatmap_chart': heatmap_fig,
            'scatter_chart': scatter_fig
        }
    
    def create_status_distribution_chart(self, df):
        """Create deal status distribution chart"""
        status_counts = df['Status_Kontrak'].value_counts()
        status_mapping = {
            'Berpotensi Deal': 'Ongoing',
            'Deal': 'Won',
            'Cancel': 'Lost'
        }
        
        # Apply mapping and calculate values
        status_data = []
        colors = ['#f39c12', '#27ae60', '#e74c3c']  # Orange, Green, Red
        
        for status, color in zip(['Berpotensi Deal', 'Deal', 'Cancel'], colors):
            count = status_counts.get(status, 0)
            total_value = df[df['Status_Kontrak'] == status]['Nilai_Kontrak'].sum() / 1e6
            status_data.append({
                'Status': status_mapping.get(status, status),
                'Count': count,
                'Total_Value': total_value,
                'Color': color
            })
        
        status_df = pd.DataFrame(status_data)
        
        # Create subplot for count and value
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Deal Count by Status', 'Deal Value by Status (Millions IDR)'),
            specs=[[{"type": "domain"}, {"type": "domain"}]]
        )
        
        # Count pie chart
        fig.add_trace(
            go.Pie(
                labels=status_df['Status'],
                values=status_df['Count'],
                marker_colors=status_df['Color'],
                hole=0.3,
                name="Count"
            ),
            row=1, col=1
        )
        
        # Value pie chart
        fig.add_trace(
            go.Pie(
                labels=status_df['Status'],
                values=status_df['Total_Value'],
                marker_colors=status_df['Color'],
                hole=0.3,
                name="Value"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Deal Status Distribution Overview',
            showlegend=True,
            height=400,
            font=dict(size=12)
        )
        
        return fig

    def create_aht_by_salesperson_chart(self, df, metrics_calc):
        """Create AHT breakdown by salesperson chart"""
        aht_data = metrics_calc.calculate_aht_by_salesperson(df)
        
        if aht_data.empty:
            # Return empty chart if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No AHT data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Average Handling Time by Salesperson")
            return fig
        
        fig = px.bar(
            aht_data,
            x='Nama_Sales',
            y='Avg_Handling_Time',
            title='Average Handling Time by Salesperson',
            labels={'Avg_Handling_Time': 'Average Days', 'Nama_Sales': 'Salesperson'},
            color='Avg_Handling_Time',
            color_continuous_scale='RdYlBu_r',  # Red for high, Blue for low
            text='Avg_Handling_Time'
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}d',
            textposition='outside'
        )
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            font=dict(size=12),
            showlegend=False
        )
        
        return fig
    
    def create_aht_by_segment_chart(self, df, metrics_calc):
        """Create AHT breakdown by segment chart"""
        aht_data = metrics_calc.calculate_aht_by_segment(df)
        
        if aht_data.empty:
            # Return empty chart if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No AHT data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Average Handling Time by Segment")
            return fig
        
        fig = px.bar(
            aht_data,
            x='Segmen',
            y='Avg_Handling_Time',
            title='Average Handling Time by Segment',
            labels={'Avg_Handling_Time': 'Average Days', 'Segmen': 'Segment'},
            color='Avg_Handling_Time',
            color_continuous_scale='RdYlBu_r',  # Red for high, Blue for low
            text='Avg_Handling_Time'
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}d',
            textposition='outside'
        )
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            font=dict(size=12),
            showlegend=False
        )
        
        return fig
    
    def create_sales_timeline_chart(self, timeline_data):
        """Create sales timeline progression chart"""
        if timeline_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No timeline data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Sales Timeline Analysis")
            return fig
        
        # Create timeline chart showing progression from Inisiasi to each stage
        stages = ['Days_to_Presentasi', 'Days_to_Penawaran_Harga', 'Days_to_Negosiasi', 'Days_to_Paska_Deal']
        stage_labels = ['Presentasi', 'Penawaran Harga', 'Negosiasi', 'Paska Deal']
        
        fig = go.Figure()
        
        for i, (stage, label) in enumerate(zip(stages, stage_labels)):
            if stage in timeline_data.columns:
                # Filter out None values
                stage_data = timeline_data[timeline_data[stage].notna()]
                if not stage_data.empty:
                    fig.add_trace(go.Box(
                        y=stage_data[stage],
                        name=label,
                        boxpoints='outliers',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
        
        fig.update_layout(
            title='Sales Timeline: Days from Inisiasi to Each Stage',
            yaxis_title='Days from Inisiasi',
            xaxis_title='Sales Stages',
            height=500,
            font=dict(size=12)
        )
        
        return fig
    
    def create_deal_closure_timeline_chart(self, timeline_data):
        """Create deal closure timeline comparison chart"""
        if timeline_data.empty or 'Avg_Deal_Closure_Days' not in timeline_data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No deal closure data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Deal Closure Timeline")
            return fig
        
        # Filter sales with deal closure data
        closure_data = timeline_data[timeline_data['Avg_Deal_Closure_Days'].notna()].copy()
        
        if closure_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No sales with completed deals found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Deal Closure Timeline")
            return fig
        
        # Sort by closure time
        closure_data = closure_data.sort_values('Avg_Deal_Closure_Days')
        
        fig = px.bar(
            closure_data,
            x='Nama_Sales',
            y='Avg_Deal_Closure_Days',
            color='Level_Sales',
            title='Average Deal Closure Time by Salesperson',
            labels={'Avg_Deal_Closure_Days': 'Average Days to Close Deal', 'Nama_Sales': 'Salesperson'},
            color_discrete_map={'AM': '#3498db', 'EAM': '#27ae60'},
            text='Avg_Deal_Closure_Days'
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}d',
            textposition='outside'
        )
        
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            font=dict(size=12)
        )
        
        return fig
    
    def create_stage_progression_heatmap(self, timeline_data):
        """Create stage progression efficiency heatmap"""
        if timeline_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No timeline data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Stage Progression Heatmap")
            return fig
        
        # Select transition columns
        transition_cols = [col for col in timeline_data.columns if 'Transition_' in col]
        
        if not transition_cols:
            fig = go.Figure()
            fig.add_annotation(
                text="No transition data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Stage Progression Heatmap")
            return fig
        
        # Prepare data for heatmap
        heatmap_data = []
        for _, row in timeline_data.iterrows():
            sales_transitions = []
            for col in transition_cols:
                if pd.notna(row[col]):
                    # Clean column name for display
                    transition_name = col.replace('Transition_', '').replace('_to_', ' → ').replace('_', ' ')
                    sales_transitions.append({
                        'Salesperson': row['Nama_Sales'],
                        'Transition': transition_name,
                        'Days': row[col]
                    })
            heatmap_data.extend(sales_transitions)
        
        if not heatmap_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid transition data found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Stage Progression Heatmap")
            return fig
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Create pivot table
        pivot_data = heatmap_df.pivot(index='Salesperson', columns='Transition', values='Days')
        
        fig = px.imshow(
            pivot_data,
            title='Stage Transition Efficiency Heatmap (Days)',
            labels=dict(x="Stage Transitions", y="Salesperson", color="Days"),
            color_continuous_scale='RdYlBu_r',  # Red for slow, Blue for fast
            aspect='auto'
        )
        
        fig.update_layout(
            height=600,
            font=dict(size=12)
        )
        
        return fig

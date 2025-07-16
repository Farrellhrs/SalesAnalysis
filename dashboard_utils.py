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
        
        # Scale monetary values to billions
        df['Target_Sales_Billions'] = df['Target_Sales'] / 1e9
        df['Target_Segmen_Billions'] = df['Target_Segmen'] / 1e9
        
        return df

class MetricsCalculator:
    """Calculate various business metrics"""
    
    def calculate_key_metrics(self, df):
        """Calculate key performance metrics"""
        total_visits = len(df)
        total_deals = df['Deal_Binary'].sum()
        win_rate = (total_deals / total_visits * 100) if total_visits > 0 else 0
        
        deal_df = df[df['Status_Kontrak'] == 'Deal']
        avg_visits_to_close = deal_df['Kunjungan_Ke'].mean() if len(deal_df) > 0 else 0
        total_deal_value = deal_df['Target_Sales'].sum() / 1e9 if len(deal_df) > 0 else 0
        
        return {
            'total_visits': total_visits,
            'total_deals': total_deals,
            'win_rate': win_rate,
            'avg_visits_to_close': avg_visits_to_close,
            'total_deal_value': total_deal_value
        }
    
    def calculate_segment_metrics(self, df):
        """Calculate segment-wise performance metrics"""
        segment_summary = df.groupby('Segmen').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()],
            'Target_Sales': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum(),
            'Kunjungan_Ke': 'mean'
        }).round(2)
        
        segment_summary.columns = ['Total_Visits', 'Total_Deals', 'Deal_Value', 'Avg_Visits']
        segment_summary['Win_Rate'] = (segment_summary['Total_Deals'] / segment_summary['Total_Visits'] * 100).round(1)
        segment_summary['Deal_Value_Billions'] = (segment_summary['Deal_Value'] / 1e9).round(2)
        segment_summary = segment_summary.reset_index().sort_values('Win_Rate', ascending=False)
        
        return segment_summary[['Segmen', 'Total_Visits', 'Total_Deals', 'Win_Rate', 'Deal_Value_Billions', 'Avg_Visits']]
    
    def calculate_sales_metrics(self, df):
        """Calculate individual sales performance metrics"""
        sales_summary = df.groupby('Nama_Sales').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()],
            'Level_Sales': 'first',
            'Target_Sales': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum(),
            'Segmen': lambda x: ', '.join(x.unique()[:3])  # Top 3 segments
        }).round(2)
        
        sales_summary.columns = ['Total_Visits', 'Total_Deals', 'Level', 'Deal_Value', 'Segments']
        sales_summary['Win_Rate'] = (sales_summary['Total_Deals'] / sales_summary['Total_Visits'] * 100).round(1)
        sales_summary['Deal_Value_Billions'] = (sales_summary['Deal_Value'] / 1e9).round(2)
        sales_summary = sales_summary.reset_index().sort_values('Win_Rate', ascending=False)
        
        return sales_summary[['Nama_Sales', 'Level', 'Total_Visits', 'Total_Deals', 'Win_Rate', 'Deal_Value_Billions', 'Segments']]
    
    def calculate_level_comparison(self, df):
        """Compare AM vs EAM performance"""
        level_comparison = df.groupby('Level_Sales').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()],
            'Nama_Sales': 'nunique',
            'Target_Sales': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum(),
            'Kunjungan_Ke': 'mean'
        }).round(2)
        
        level_comparison.columns = ['Total_Visits', 'Total_Deals', 'Unique_Sales', 'Deal_Value', 'Avg_Visits']
        level_comparison['Win_Rate'] = (level_comparison['Total_Deals'] / level_comparison['Total_Visits'] * 100).round(1)
        level_comparison['Deals_per_Sales'] = (level_comparison['Total_Deals'] / level_comparison['Unique_Sales']).round(1)
        level_comparison['Visits_per_Sales'] = (level_comparison['Total_Visits'] / level_comparison['Unique_Sales']).round(1)
        level_comparison['Deal_Value_Billions'] = (level_comparison['Deal_Value'] / 1e9).round(2)
        
        return level_comparison.reset_index()
    
    def calculate_progress_metrics(self, df):
        """Calculate progress stage metrics"""
        progress_summary = df.groupby('Progress').agg({
            'Status_Kontrak': ['count', lambda x: (x == 'Deal').sum()],
            'Kunjungan_Ke': 'mean',
            'Target_Sales': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum()
        }).round(2)
        
        progress_summary.columns = ['Total_Visits', 'Total_Deals', 'Avg_Visit_Number', 'Deal_Value']
        progress_summary['Conversion_Rate'] = (progress_summary['Total_Deals'] / progress_summary['Total_Visits'] * 100).round(1)
        progress_summary['Deal_Value_Billions'] = (progress_summary['Deal_Value'] / 1e9).round(2)
        
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
            'Target_Sales': lambda x: x[df.loc[x.index, 'Status_Kontrak'] == 'Deal'].sum()
        }).round(2)
        
        sales_data.columns = ['Total_Visits', 'Total_Deals', 'Level', 'Deal_Value']
        sales_data['Win_Rate'] = (sales_data['Total_Deals'] / sales_data['Total_Visits'] * 100).round(1)
        sales_data['Deal_Value_Billions'] = (sales_data['Deal_Value'] / 1e9).round(2)
        sales_data = sales_data.reset_index()
        
        fig = px.scatter(
            sales_data.head(20),  # Top 20 performers
            x='Total_Visits',
            y='Total_Deals',
            size='Deal_Value_Billions',
            color='Win_Rate',
            hover_name='Nama_Sales',
            hover_data=['Level'],
            color_continuous_scale='RdYlGn',
            title='Sales Performance: Visits vs Deals'
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
            subplot_titles=('Win Rate', 'Total Visits', 'Total Deals', 'Deal Value (Billions)'),
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
            go.Bar(x=segment_summary['Segmen'], y=segment_summary['Deal_Value_Billions'], 
                   name='Deal Value', marker_color=self.color_palette['danger']),
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

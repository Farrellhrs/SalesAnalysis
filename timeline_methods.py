    
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

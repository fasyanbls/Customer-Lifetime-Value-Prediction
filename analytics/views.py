import json
import plotly
import plotly.graph_objs as go
import pandas as pd
from django.shortcuts import render
from django.db.models import Sum, Avg, Count, Max, Min

from .models import FactCustomerCLV, DimCustomer, MLModelLog
from .management.commands.run_etl import run_full_etl
from .ml_model import (
    train_clv_model, predict_clv, get_feature_importance,
    predict_all_customers, compare_models,
)

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Georgia, serif', color='#3d2b1f', size=12),
)
PLOTLY_CFG = {'responsive': True, 'displayModeBar': False}


def _fig_json(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# DASHBOARD
def dashboard(request):
    facts = FactCustomerCLV.objects.select_related('customer')

    # KPIs
    kpis = facts.aggregate(
        total_revenue   = Sum('total_payments'),
        avg_clv         = Avg('clv_predicted'),
        total_customers = Count('customer', distinct=True),
        avg_recency     = Avg('recency_days'),
    )

    # Segment distribution
    segments = list(
        facts.values('clv_segment')
             .annotate(count=Count('pk'), revenue=Sum('total_payments'))
             .order_by('-revenue')
    )

    # Chart 1: Revenue by segment (bar)
    seg_labels  = [s['clv_segment'] for s in segments]
    seg_revenue = [float(s['revenue'] or 0) for s in segments]
    seg_counts  = [s['count'] for s in segments]
    seg_colors  = {'High': '#8B3A3A', 'Medium': '#8B6F4E', 'Low': '#C4956A'}
    fig_seg = go.Figure(data=[
        go.Bar(
            x=seg_labels,
            y=seg_revenue,
            marker_color=[seg_colors.get(l, '#C4956A') for l in seg_labels],
            text=[f'${v:,.0f}<br>{c} customers' for v, c in zip(seg_revenue, seg_counts)],
            textposition='outside',
        )
    ])
    fig_seg.update_layout(
        **PLOTLY_BASE,
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
        yaxis_title='Total Revenue ($)',
        showlegend=False,
    )

    # Chart 2: Customer segments donut 
    fig_pie = go.Figure(data=[
        go.Pie(
            labels=seg_labels,
            values=seg_counts,
            marker_colors=[seg_colors.get(l, '#C4956A') for l in seg_labels],
            hole=0.5,
            textinfo='label+percent',
        )
    ])
    fig_pie.update_layout(
        **PLOTLY_BASE,
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        showlegend=True,
    )

    # Chart 3: CLV by recency 
    rfm_data = list(
        facts.values('recency_days', 'clv_predicted', 'clv_segment', 'frequency')[:300]
    )
    rfm_df = pd.DataFrame(rfm_data)
    fig_rfm = go.Figure()
    for seg, color in seg_colors.items():
        sub = rfm_df[rfm_df['clv_segment'] == seg] if not rfm_df.empty else pd.DataFrame()
        if not sub.empty:
            fig_rfm.add_trace(go.Scatter(
                x=sub['recency_days'],
                y=sub['clv_predicted'].astype(float),
                mode='markers',
                name=seg,
                marker=dict(color=color, size=7, opacity=0.7),
                hovertemplate='Recency: %{x}d<br>CLV: $%{y:.2f}<extra></extra>',
            ))
    fig_rfm.update_layout(
        **PLOTLY_BASE,
        margin=dict(l=50, r=20, t=20, b=50),
        height=300,
        xaxis_title='Recency (days since last rental)',
        yaxis_title='Predicted CLV ($)',
    )

    top_customers = facts.order_by('-clv_predicted')[:10]

    context = {
        'kpis':          kpis,
        'segments':      segments,
        'top_customers': top_customers,
        'chart_seg':     _fig_json(fig_seg),
        'chart_pie':     _fig_json(fig_pie),
        'chart_rfm':     _fig_json(fig_rfm),
    }
    return render(request, 'analytics/dashboard.html', context)


# ETL
def etl_view(request):
    result = None
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'run_etl':
            result = run_full_etl()
        elif action == 'predict_all':
            result = predict_all_customers()
    return render(request, 'analytics/etl.html', {'result': result})


# ML MODEL
def model_view(request):
    result       = None
    compare_result = None

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'train':
            result = train_clv_model()
        elif action == 'compare':
            compare_result = compare_models()

    # Feature importance chart
    fi       = get_feature_importance()
    chart_fi = None
    if fi:
        feat_labels = {
            'recency_days':         'Recency (days)',
            'frequency':            'Frequency',
            'monetary_avg':         'Monetary Avg ($)',
            'customer_tenure_days': 'Customer Tenure',
            'payment_count':        'Payment Count',
        }
        labels = [feat_labels.get(k, k) for k in fi.keys()]
        values = [round(v * 100, 2) for v in fi.values()]
        colors = ['#8B3A3A', '#8B6F4E', '#C4956A', '#D4A97A', '#E8C99A']
        fig_fi = go.Figure(data=[
            go.Bar(
                x=values, y=labels, orientation='h',
                marker_color=colors[:len(labels)],
                text=[f'{v:.1f}%' for v in values],
                textposition='outside',
            )
        ])
        fig_fi.update_layout(
            **PLOTLY_BASE,
            margin=dict(l=140, r=60, t=20, b=40),
            height=260,
            xaxis_title='Importance (%)',
            showlegend=False,
        )
        chart_fi = _fig_json(fig_fi)

    # Actual vs Predicted chart 
    chart_avp = None
    if result and result.get('success') and result.get('actual'):
        actual    = result['actual']
        predicted = result['predicted']
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=actual, y=predicted,
            mode='markers',
            marker=dict(color='#8B6F4E', size=7, opacity=0.7),
            name='Predictions',
            hovertemplate='Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>',
        ))
        # Perfect prediction line
        max_val = max(max(actual), max(predicted))
        fig_avp.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines',
            line=dict(color='#8B3A3A', dash='dash', width=1.5),
            name='Perfect Prediction',
        ))
        fig_avp.update_layout(
            **PLOTLY_BASE,
            margin=dict(l=50, r=20, t=20, b=50),
            height=300,
            xaxis_title='Actual CLV ($)',
            yaxis_title='Predicted CLV ($)',
        )
        chart_avp = _fig_json(fig_avp)

    # Model comparison chart 
    chart_compare = None
    if compare_result and compare_result.get('success'):
        models  = list(compare_result['results'].keys())
        r2_vals = [compare_result['results'][m]['r2']  for m in models]
        mae_vals= [compare_result['results'][m]['mae'] for m in models]
        cv_vals = [compare_result['results'][m]['cv_r2_mean'] for m in models]

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name='R² (test)', x=models, y=r2_vals,
            marker_color='#8B6F4E',
            text=[f'{v:.3f}' for v in r2_vals],
            textposition='outside',
        ))
        fig_cmp.add_trace(go.Bar(
            name='CV R² (mean)', x=models, y=cv_vals,
            marker_color='#C4956A',
            text=[f'{v:.3f}' for v in cv_vals],
            textposition='outside',
        ))
        fig_cmp.update_layout(
            **PLOTLY_BASE,
            margin=dict(l=40, r=20, t=20, b=40),
            height=300,
            barmode='group',
            yaxis_title='R² Score (higher = better)',
            yaxis_range=[0, 1.1],
        )
        chart_compare = _fig_json(fig_cmp)

    logs = MLModelLog.objects.order_by('-trained_at')[:10]

    context = {
        'result':         result,
        'compare_result': compare_result,
        'chart_fi':       chart_fi,
        'chart_avp':      chart_avp,
        'chart_compare':  chart_compare,
        'logs':           logs,
    }
    return render(request, 'analytics/model.html', context)


# PREDICTION
def prediction_view(request):
    prediction = None
    if request.method == 'POST':
        try:
            prediction = predict_clv(
                recency_days  = int(request.POST.get('recency_days', 0)),
                frequency     = int(request.POST.get('frequency', 0)),
                monetary_avg  = float(request.POST.get('monetary_avg', 0)),
                tenure_days   = int(request.POST.get('tenure_days', 0)),
                payment_count = int(request.POST.get('payment_count', 0)),
            )
            
            if 'error' not in prediction:
                val = prediction['predicted_clv']
                p33 = prediction['p33']
                p66 = prediction['p66']
                max_val = max(p66 * 1.5, val * 1.2)
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    number={'prefix': "$"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "CLV Segment Position", 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, max_val]},
                        'bar': {'color': "#3d2b1f"},
                        'steps': [
                            {'range': [0, p33], 'color': "#E8C99A"},      # Low
                            {'range': [p33, p66], 'color': "#C4956A"},    # Medium
                            {'range': [p66, max_val], 'color': "#8B3A3A"} # High
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': val
                        }
                    }
                ))
                fig_gauge.update_layout(
                    **PLOTLY_BASE,
                    margin=dict(l=40, r=40, t=40, b=20),
                    height=250,
                )
                prediction['chart_gauge'] = _fig_json(fig_gauge)
                
        except Exception as e:
            prediction = {'error': str(e)}

    return render(request, 'analytics/prediction.html', {'prediction': prediction})

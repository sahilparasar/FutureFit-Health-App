import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def generate_sample_data():
    """Generate sample health assessment data"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Height': np.random.normal(170, 10, n_samples),
        'Weight': np.random.normal(70, 15, n_samples),
        'Hip': np.random.normal(100, 8, n_samples),
        'Waist': np.random.normal(85, 12, n_samples),
        'Hand_Grip_Strength': np.random.normal(35, 8, n_samples),
        'Curl_Ups': np.random.poisson(25, n_samples),
        'Pushups': np.random.poisson(20, n_samples),
        'Step_Test_Heart_Rate': np.random.normal(110, 20, n_samples),
        'Balance_Test_Seconds': np.random.normal(45, 15, n_samples),
        'Flexibility_cm': np.random.normal(25, 8, n_samples),
    }
    
    return pd.DataFrame(data)

def create_risk_radar_chart(risk_probabilities):
    """Create radar chart for risk assessment"""
    categories = ['Cardiovascular', 'Musculoskeletal', 'Metabolic', 'Functional Decline']
    
    fig = go.Figure(data=go.Scatterpolar(
        r=risk_probabilities,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Health Risk Assessment Radar"
    )
    
    return fig

def create_fitness_dashboard(health_data):
    """Create comprehensive fitness dashboard"""
    # Calculate metrics
    bmi = health_data['Weight'] / ((health_data['Height'] / 100) ** 2)
    wh_ratio = health_data['Waist'] / health_data['Hip']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('BMI Distribution', 'Waist-to-Hip Ratio', 
                       'Strength Metrics', 'Flexibility & Balance'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # BMI histogram
    fig.add_trace(go.Histogram(x=bmi, name='BMI'), row=1, col=1)
    
    # WH Ratio histogram
    fig.add_trace(go.Histogram(x=wh_ratio, name='WH Ratio'), row=1, col=2)
    
    # Strength metrics bar
    strength_metrics = ['Hand Grip', 'Pushups', 'Curl Ups']
    strength_values = [
        health_data['Hand_Grip_Strength'].iloc[0],
        health_data['Pushups'].iloc[0],
        health_data['Curl_Ups'].iloc[0]
    ]
    fig.add_trace(go.Bar(x=strength_metrics, y=strength_values, name='Strength'), row=2, col=1)
    
    # Flexibility vs Balance scatter
    fig.add_trace(go.Scatter(
        x=[health_data['Flexibility_cm'].iloc[0]],
        y=[health_data['Balance_Test_Seconds'].iloc[0]],
        mode='markers',
        name='Your Score',
        marker=dict(size=15, color='red')
    ), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    return fig
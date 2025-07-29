# Agricultural Monitoring Platform - Streamlit Frontend
# Complete web interface for the agricultural monitoring system

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import io
from PIL import Image
import base64
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import asyncio
import aiohttp

# Page configuration
st.set_page_config(
    page_title="Agricultural Monitoring Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .alert-card {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .info-card {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 100%);
    }
    .stSelectbox > div > div {
        background-color: #F1F8E9;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Helper functions
def safe_json_serialize(obj):
    """Safely serialize objects to JSON, handling numpy types and other non-serializable objects"""
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, tuple):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif obj is None:
        return None
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return str(obj)

class AgriAPI:
    """API client for agricultural monitoring backend"""
    
    @staticmethod
    def get_dashboard_summary():
        """Get dashboard summary from API"""
        try:
            response = requests.get(f"{API_BASE_URL}/dashboard/summary")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to fetch dashboard data: {str(e)}")
            return None
    
    @staticmethod
    def create_field(field_data):
        """Create a new field"""
        try:
            response = requests.post(f"{API_BASE_URL}/fields/", json=field_data)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Failed to create field: {str(e)}")
            return None
    
    @staticmethod
    def analyze_field(field_id, image_file):
        """Submit field analysis"""
        try:
            files = {"image": image_file}
            data = {"field_id": field_id}
            response = requests.post(f"{API_BASE_URL}/analyze/", files=files, data=data)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Failed to analyze field: {str(e)}")
            return None
    
    @staticmethod
    def get_field_history(field_id):
        """Get field analysis history"""
        try:
            response = requests.get(f"{API_BASE_URL}/fields/{field_id}/history")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Failed to fetch field history: {str(e)}")
            return None

def create_sample_data():
    """Create sample data for demonstration"""
    return {
        "total_fields": 25,
        "total_area_hectares": 450.5,
        "average_health_score": 0.78,
        "total_predicted_yield": 1250000,
        "disease_alerts": 3,
        "recent_analyses": 15
    }

def create_sample_fields():
    """Create sample field data"""
    return [
        {"id": 1, "name": "North Field", "crop_type": "Wheat", "area": 25.5, "health": 0.85, "status": "Healthy"},
        {"id": 2, "name": "South Field", "crop_type": "Corn", "area": 18.2, "health": 0.72, "status": "Monitoring"},
        {"id": 3, "name": "East Field", "crop_type": "Soybeans", "area": 32.1, "health": 0.91, "status": "Excellent"},
        {"id": 4, "name": "West Field", "crop_type": "Rice", "area": 28.7, "health": 0.45, "status": "Alert"},
        {"id": 5, "name": "Central Field", "crop_type": "Cotton", "area": 22.3, "health": 0.68, "status": "Fair"},
    ]

def create_sample_history():
    """Create sample historical data"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
    base_health = 0.8
    
    history = []
    for i, date in enumerate(dates):
        # Add seasonal variation and some noise
        seasonal_factor = 0.2 * np.sin(2 * np.pi * i / 52) + 0.1 * np.random.randn()
        health = np.clip(base_health + seasonal_factor, 0.2, 1.0)
        
        history.append({
            'date': date,
            'health_score': health,
            'predicted_yield': health * 50000 + np.random.normal(0, 5000),
            'ndvi_avg': health * 0.8 + 0.1,
            'disease_detected': np.random.random() < 0.1
        })
    
    return pd.DataFrame(history)

def get_health_color(health_score):
    """Get color based on health score"""
    if health_score >= 0.8:
        return "#4CAF50"  # Green
    elif health_score >= 0.6:
        return "#FF9800"  # Orange
    elif health_score >= 0.4:
        return "#FF5722"  # Red-Orange
    else:
        return "#F44336"  # Red

def get_status_emoji(status):
    """Get emoji based on status"""
    emoji_map = {
        "Excellent": "🟢",
        "Healthy": "✅",
        "Monitoring": "🟡",
        "Fair": "🟠",
        "Alert": "🔴"
    }
    return emoji_map.get(status, "⚪")

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Agricultural Monitoring Platform</h1>', unsafe_allow_html=True)
    
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Field Management", "Analysis & Monitoring", "Historical Data", "Alerts & Reports", "Settings"],
        index=["Dashboard", "Field Management", "Analysis & Monitoring", "Historical Data", "Alerts & Reports", "Settings"].index(st.session_state.current_page)
    )
    
    # Update session state when selectbox changes
    if page != st.session_state.current_page:
        st.session_state.current_page = page
    
    # Page routing - use session state
    current_page = st.session_state.current_page
    if current_page == "Dashboard":
        show_dashboard()
    elif current_page == "Field Management":
        show_field_management()
    elif current_page == "Analysis & Monitoring":
        show_analysis_monitoring()
    elif current_page == "Historical Data":
        show_historical_data()
    elif current_page == "Alerts & Reports":
        show_alerts_reports()
    elif current_page == "Settings":
        show_settings()

def show_dashboard():
    """Display main dashboard"""
    st.header("📊 Farm Overview Dashboard")
    
    # Fetch data (with fallback to sample data)
    dashboard_data = AgriAPI.get_dashboard_summary()
    if not dashboard_data:
        dashboard_data = create_sample_data()
        st.info("📡 Using demo data - Connect to backend for live data")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>🏞️ Total Fields</h3><h2>{dashboard_data["total_fields"]}</h2></div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div class="metric-card"><h3>📏 Total Area</h3><h2>{dashboard_data["total_area_hectares"]:.1f} ha</h2></div>',
            unsafe_allow_html=True
        )
    
    with col3:
        health_score = dashboard_data["average_health_score"]
        health_color = get_health_color(health_score)
        st.markdown(
            f'<div class="metric-card"><h3>💚 Avg Health</h3><h2 style="color: {health_color}">{health_score:.2f}</h2></div>',
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f'<div class="metric-card"><h3>📈 Predicted Yield</h3><h2>{dashboard_data["total_predicted_yield"]:,.0f} kg</h2></div>',
            unsafe_allow_html=True
        )
    
    # Alerts section
    if dashboard_data["disease_alerts"] > 0:
        st.markdown(
            f'<div class="alert-card"><h3>⚠️ Active Alerts</h3><p>{dashboard_data["disease_alerts"]} fields require immediate attention</p></div>',
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏞️ Field Health Status")
        
        # Create field status data
        fields = create_sample_fields()
        status_counts = pd.DataFrame(fields)['status'].value_counts()
        
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Field Status Distribution",
            color_discrete_map={
                'Excellent': '#4CAF50',
                'Healthy': '#8BC34A',
                'Monitoring': '#FF9800',
                'Fair': '#FF5722',
                'Alert': '#F44336'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Crop Type Distribution")
        
        crop_data = pd.DataFrame(fields)
        crop_area = crop_data.groupby('crop_type')['area'].sum()
        
        fig = px.bar(
            x=crop_area.index,
            y=crop_area.values,
            title="Area by Crop Type (hectares)",
            color=crop_area.values,
            color_continuous_scale="Greens"
        )
        fig.update_layout(showlegend=False, xaxis_title="Crop Type", yaxis_title="Area (ha)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Field overview table
    st.subheader("🌾 Field Overview")
    
    df_fields = pd.DataFrame(fields)
    df_fields['Status'] = df_fields.apply(lambda row: f"{get_status_emoji(row['status'])} {row['status']}", axis=1)
    df_fields['Health Score'] = df_fields['health'].apply(lambda x: f"{x:.2f}")
    df_fields['Area (ha)'] = df_fields['area'].apply(lambda x: f"{x:.1f}")
    
    display_df = df_fields[['name', 'crop_type', 'Area (ha)', 'Health Score', 'Status']].copy()
    display_df.columns = ['Field Name', 'Crop Type', 'Area (ha)', 'Health Score', 'Status']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Recent activity
    st.subheader("📝 Recent Activity")
    recent_activities = [
        "🔍 North Field analyzed - Health score: 0.85",
        "⚠️ West Field showing stress indicators",
        "✅ Disease treatment applied to South Field",
        "📊 Weekly yield prediction updated",
        "🌡️ Weather alert: High temperatures expected"
    ]
    
    for activity in recent_activities:
        st.write(f"• {activity}")

def show_field_management():
    """Display field management interface"""
    st.header("🏞️ Field Management")
    
    tab1, tab2, tab3 = st.tabs(["📋 Field List", "➕ Add New Field", "🗺️ Field Map"])
    
    with tab1:
        st.subheader("Existing Fields")
        
        fields = create_sample_fields()
        
        for field in fields:
            with st.expander(f"{field['name']} - {field['crop_type']} ({field['area']:.1f} ha)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Health Score", f"{field['health']:.2f}")
                
                with col2:
                    st.metric("Status", field['status'])
                
                with col3:
                    st.metric("Area", f"{field['area']:.1f} ha")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"🔍 Analyze", key=f"analyze_{field['id']}"):
                        st.info("Redirecting to analysis page...")
                
                with col2:
                    if st.button(f"📊 History", key=f"history_{field['id']}"):
                        st.info("Loading historical data...")
                
                with col3:
                    if st.button(f"✏️ Edit", key=f"edit_{field['id']}"):
                        st.info("Edit functionality coming soon...")
    
    with tab2:
        st.subheader("Add New Field")
        
        with st.form("new_field_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                field_name = st.text_input("Field Name*", placeholder="e.g., North Field")
                crop_type = st.selectbox("Crop Type*", 
                    ["Wheat", "Corn", "Rice", "Soybeans", "Cotton", "Barley", "Other"])
                area = st.number_input("Area (hectares)*", min_value=0.1, value=10.0, step=0.1)
            
            with col2:
                location = st.text_input("Location", placeholder="e.g., GPS coordinates or address")
                planting_date = st.date_input("Planting Date")
                notes = st.text_area("Notes", placeholder="Additional information about the field")
            
            # Field boundary (simplified)
            st.subheader("Field Boundary")
            st.info("📍 For demo purposes, use sample coordinates. In production, integrate with map drawing tools.")
            geometry = st.text_area("Geometry (WKT format)", 
                value="POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))",
                help="Well-Known Text format for field boundary")
            
            submitted = st.form_submit_button("Create Field", type="primary")
            
            if submitted:
                if field_name and crop_type and area:
                    field_data = {
                        "name": field_name,
                        "crop_type": crop_type,
                        "area_hectares": area,
                        "geometry": geometry
                    }
                    
                    result = AgriAPI.create_field(field_data)
                    if result:
                        st.success(f"✅ Field '{field_name}' created successfully!")
                    else:
                        st.error("❌ Failed to create field. Please check your connection to the backend.")
                else:
                    st.error("❌ Please fill in all required fields marked with *")
    
    with tab3:
        st.subheader("Field Locations Map")
        
        # Create a sample map
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
        
        # Add sample field markers
        sample_locations = [
            {"name": "North Field", "lat": 40.720, "lon": -74.000, "health": 0.85, "crop": "Wheat"},
            {"name": "South Field", "lat": 40.710, "lon": -74.010, "health": 0.72, "crop": "Corn"},
            {"name": "East Field", "lat": 40.715, "lon": -73.990, "health": 0.91, "crop": "Soybeans"},
            {"name": "West Field", "lat": 40.708, "lon": -74.020, "health": 0.45, "crop": "Rice"},
        ]
        
        for location in sample_locations:
            color = 'green' if location['health'] > 0.8 else 'orange' if location['health'] > 0.6 else 'red'
            
            folium.Marker(
                [location['lat'], location['lon']],
                popup=f"<b>{location['name']}</b><br>Crop: {location['crop']}<br>Health: {location['health']:.2f}",
                tooltip=location['name'],
                icon=folium.Icon(color=color, icon='leaf')
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=500)
        
        if map_data['last_object_clicked_popup']:
            st.info(f"Selected: {map_data['last_object_clicked_popup']}")

def show_analysis_monitoring():
    """Display analysis and monitoring interface"""
    st.header("🔬 Analysis & Monitoring")
    
    tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "🔄 Real-time Monitoring", "📊 Batch Processing"])
    
    with tab1:
        st.subheader("Field Image Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Field selection
            fields = create_sample_fields()
            field_options = {f"{f['name']} - {f['crop_type']}": f['id'] for f in fields}
            selected_field_name = st.selectbox("Select Field", list(field_options.keys()))
            field_id = field_options[selected_field_name]
            
            # Display selected field info
            selected_field = next(f for f in fields if f['id'] == field_id)
            st.info(f"**Field:** {selected_field['name']}\n**Crop:** {selected_field['crop_type']}\n**Area:** {selected_field['area']:.1f} ha\n**Current Health:** {selected_field['health']:.2f}")
            
            # Image upload
            uploaded_file = st.file_uploader(
                "Upload Field Image",
                type=['jpg', 'jpeg', 'png', 'tiff', 'tif'],
                help="Upload drone or satellite imagery of your field"
            )
            
            # Analysis options
            st.subheader("Analysis Options")
            analyze_health = st.checkbox("Health Assessment", value=True)
            analyze_disease = st.checkbox("Disease Detection", value=True)
            analyze_yield = st.checkbox("Yield Prediction", value=True)
            analyze_segmentation = st.checkbox("Crop Segmentation", value=True)
            
            # Analysis button - only enabled when image is uploaded
            analysis_button = st.button(
                "🔍 Start Analysis", 
                type="primary", 
                disabled=not uploaded_file,
                use_container_width=True
            )
        
        with col2:
            if uploaded_file is not None:
                # Display uploaded image
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Field Image: {uploaded_file.name}", use_column_width=True)
                
                # Show image details
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"**Format:** {image.format}")
                st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
                
                # Perform analysis when button is clicked
                if analysis_button:
                    with st.spinner("🔄 Analyzing image... This may take a few minutes."):
                        # Show progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Simulate analysis steps
                        steps = [
                            "Loading image...",
                            "Preprocessing data...",
                            "Running segmentation model...",
                            "Calculating health metrics...",
                            "Detecting diseases...",
                            "Predicting yield...",
                            "Generating recommendations..."
                        ]
                        
                        analysis_results = {}
                        
                        for i, step in enumerate(steps):
                            status_text.text(step)
                            progress_bar.progress((i + 1) / len(steps))
                            
                            # Simulate processing time
                            import time
                            time.sleep(0.8)
                            
                            # Generate realistic results based on analysis step
                            if "health" in step.lower():
                                analysis_results['health_score'] = float(np.random.uniform(0.45, 0.92))
                                analysis_results['ndvi_avg'] = float(analysis_results['health_score'] * 0.75 + np.random.uniform(-0.1, 0.1))
                            elif "disease" in step.lower():
                                analysis_results['disease_detected'] = bool(np.random.choice([True, False], p=[0.3, 0.7]))
                                if analysis_results['disease_detected']:
                                    analysis_results['disease_type'] = str(np.random.choice(['leaf_spot', 'blight', 'rust', 'wilt']))
                                    analysis_results['disease_severity'] = str(np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.3, 0.2]))
                                else:
                                    analysis_results['disease_type'] = None
                                    analysis_results['disease_severity'] = None
                            elif "yield" in step.lower():
                                base_yield = float(selected_field['area'] * 3000)  # kg per hectare
                                health_factor = analysis_results.get('health_score', 0.7)
                                analysis_results['predicted_yield'] = float(base_yield * health_factor * np.random.uniform(0.8, 1.2))
                        
                        # Generate recommendations based on results
                        recommendations = []
                        health_score = analysis_results.get('health_score', 0.7)
                        
                        if health_score < 0.5:
                            recommendations.extend([
                                "🚨 Immediate intervention required",
                                "💧 Check irrigation system",
                                "🧪 Conduct soil testing"
                            ])
                        elif health_score < 0.7:
                            recommendations.extend([
                                "⚠️ Monitor field closely",
                                "🌱 Consider nutrient supplementation"
                            ])
                        else:
                            recommendations.append("✅ Field is performing well")
                        
                        if analysis_results.get('disease_detected'):
                            disease_type = analysis_results.get('disease_type', 'unknown')
                            severity = analysis_results.get('disease_severity', 'low')
                            recommendations.extend([
                                f"🦠 {disease_type.title()} detected - {severity} severity",
                                "💊 Apply appropriate treatment",
                                "🔍 Monitor spread to adjacent areas"
                            ])
                        
                        if analysis_results.get('ndvi_avg', 0) < 0.4:
                            recommendations.append("🌿 Low vegetation index - check plant stress")
                        
                        analysis_results['recommendations'] = recommendations
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Store results in session state
                        st.session_state.analysis_results = analysis_results
                        st.session_state.analyzed_image = uploaded_file.name
                        st.session_state.analyzed_field = selected_field_name
                        
                        st.success("✅ Analysis complete!")
                
                # Display analysis results if available
                if (hasattr(st.session_state, 'analysis_results') and 
                    hasattr(st.session_state, 'analyzed_image') and 
                    st.session_state.analyzed_image == uploaded_file.name):
                    
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    results = st.session_state.analysis_results
                    
                    # Key metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        health_score = results.get('health_score', 0)
                        health_color = get_health_color(health_score)
                        st.metric(
                            "🌱 Health Score",
                            f"{health_score:.3f}",
                            delta=f"{health_score - selected_field['health']:.3f}" if 'health' in selected_field else None
                        )
                    
                    with col2:
                        ndvi_avg = results.get('ndvi_avg', 0)
                        st.metric("🌿 NDVI Average", f"{ndvi_avg:.3f}")
                    
                    with col3:
                        predicted_yield = results.get('predicted_yield', 0)
                        st.metric("📈 Predicted Yield", f"{predicted_yield:,.0f} kg")
                    
                    # Disease detection
                    if results.get('disease_detected'):
                        disease_type = results.get('disease_type', 'Unknown')
                        severity = results.get('disease_severity', 'Unknown')
                        st.markdown(
                            f'<div class="alert-card"><h4>🦠 Disease Detected</h4><p><strong>Type:</strong> {disease_type.title()}<br><strong>Severity:</strong> {severity.title()}</p></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="success-card"><h4>✅ No Disease Detected</h4><p>Field appears healthy with no signs of disease</p></div>',
                            unsafe_allow_html=True
                        )
                    
                    # Health visualization
                    fig_health = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = health_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Field Health Score"},
                        delta = {'reference': selected_field['health']},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': health_color},
                            'steps': [
                                {'range': [0, 0.4], 'color': "lightgray"},
                                {'range': [0.4, 0.7], 'color': "yellow"},
                                {'range': [0.7, 1], 'color': "lightgreen"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.9}}))
                    
                    fig_health.update_layout(height=300)
                    st.plotly_chart(fig_health, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    for i, rec in enumerate(results.get('recommendations', []), 1):
                        st.write(f"{i}. {rec}")
                    
                    # Action buttons
                    st.subheader("🎯 Actions")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("💾 Save Results", use_container_width=True):
                            # Here you would call the API to save results
                            st.success("✅ Results saved to database!")
                    
                    with col2:
                        if st.button("📧 Send Report", use_container_width=True):
                            st.success("📧 Report sent to stakeholders!")
                    
                    with col3:
                        if st.button("📅 Schedule Follow-up", use_container_width=True):
                            st.success("📅 Follow-up analysis scheduled!")
                    
                    with col4:
                        # Download detailed report
                        report_data = {
                            "field": st.session_state.analyzed_field,
                            "analysis_date": datetime.now().isoformat(),
                            "image_file": uploaded_file.name,
                            "results": safe_json_serialize(results)
                        }
                        
                        try:
                            report_json = json.dumps(report_data, indent=2)
                            st.download_button(
                                "📥 Download Report",
                                data=report_json,
                                file_name=f"analysis_report_{field_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error creating report: {str(e)}")
                            # Fallback: show the data structure for debugging
                            with st.expander("Debug: Report Data Structure"):
                                st.write(report_data)
                    
                    # API Integration section
                    st.markdown("---")
                    st.subheader("🔗 API Integration")
                    
                    with st.expander("📡 Send to Backend API"):
                        st.info("Click below to send this analysis to your backend API for storage and further processing.")
                        
                        if st.button("🚀 Submit to API", type="primary"):
                            try:
                                # Convert image to bytes for API submission
                                img_byte_arr = io.BytesIO()
                                image.save(img_byte_arr, format='JPEG')
                                img_byte_arr.seek(0)
                                
                                # Call the API
                                with st.spinner("Submitting to API..."):
                                    result = AgriAPI.analyze_field(field_id, img_byte_arr)
                                    
                                    if result:
                                        st.success("✅ Successfully submitted to API!")
                                        st.json(result)
                                    else:
                                        st.error("❌ Failed to submit to API. Check backend connection.")
                            
                            except Exception as e:
                                st.error(f"❌ Error submitting to API: {str(e)}")
            
            else:
                # Show upload instruction when no image is selected
                st.info("👆 Please upload an image to begin analysis")
                
                # Show sample images for demo
                st.subheader("📷 Sample Images")
                st.write("Use these sample images to test the analysis:")
                
                sample_images = [
                    {"name": "Healthy Wheat Field", "description": "High NDVI, good health score expected"},
                    {"name": "Diseased Corn Field", "description": "Shows signs of leaf blight"},
                    {"name": "Drought-Stressed Soybeans", "description": "Low moisture, stress indicators"},
                    {"name": "Satellite Image", "description": "Multi-spectral analysis capability"}
                ]
                
                for img in sample_images:
                    st.write(f"• **{img['name']}**: {img['description']}")
    
    with tab2:
        st.subheader("Real-time Monitoring Dashboard")
        
        # Simulate real-time data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌡️ Temperature", "24°C", "+2°C")
        with col2:
            st.metric("💧 Humidity", "65%", "-5%")
        with col3:
            st.metric("🌬️ Wind Speed", "12 km/h", "+3 km/h")
        with col4:
            st.metric("☀️ Solar Radiation", "850 W/m²", "+50 W/m²")
        
        # Live sensor data chart
        st.subheader("📈 Live Sensor Data")
        
        # Generate sample time series data
        times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             end=datetime.now(), freq='h')
        
        sensor_data = pd.DataFrame({
            'Time': times,
            'Temperature': 20 + 10 * np.sin(np.linspace(0, 4*np.pi, len(times))) + np.random.normal(0, 1, len(times)),
            'Humidity': 60 + 20 * np.cos(np.linspace(0, 4*np.pi, len(times))) + np.random.normal(0, 2, len(times)),
            'Soil_Moisture': 30 + 15 * np.sin(np.linspace(0, 2*np.pi, len(times))) + np.random.normal(0, 1.5, len(times))
        })
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Temperature (°C)', 'Humidity (%)', 'Soil Moisture (%)'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(go.Scatter(x=sensor_data['Time'], y=sensor_data['Temperature'], 
                                name='Temperature', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=sensor_data['Time'], y=sensor_data['Humidity'], 
                                name='Humidity', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=sensor_data['Time'], y=sensor_data['Soil_Moisture'], 
                                name='Soil Moisture', line=dict(color='brown')), row=3, col=1)
        
        fig.update_layout(height=600, showlegend=False, title_text="24-Hour Sensor Readings")
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert system
        st.subheader("🚨 Active Alerts")
        alerts = [
            {"level": "High", "message": "West Field: Soil moisture below critical threshold", "time": "2 hours ago"},
            {"level": "Medium", "message": "North Field: Temperature spike detected", "time": "4 hours ago"},
            {"level": "Low", "message": "East Field: Wind speed increasing", "time": "6 hours ago"}
        ]
        
        for alert in alerts:
            alert_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[alert['level']]
            st.write(f"{alert_color} **{alert['level']}**: {alert['message']} _{alert['time']}_")
    
    with tab3:
        st.subheader("Batch Processing")
        
        st.info("🔄 Process multiple images or fields simultaneously")
        
        # Batch upload
        uploaded_files = st.file_uploader(
            "Upload Multiple Images",
            type=['jpg', 'jpeg', 'png', 'tiff', 'tif'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} files uploaded")
            
            # Processing options
            col1, col2 = st.columns(2)
            with col1:
                processing_type = st.selectbox(
                    "Processing Type",
                    ["Health Assessment", "Disease Detection", "Full Analysis", "Segmentation Only"]
                )
            
            with col2:
                output_format = st.selectbox(
                    "Output Format",
                    ["JSON", "CSV", "PDF Report", "Excel"]
                )
            
            if st.button("🚀 Start Batch Processing", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    time.sleep(1)  # Simulate processing time
                
                st.success(f"✅ Batch processing complete! {len(uploaded_files)} files processed.")
                
                # Show sample results table
                results_data = []
                for i, file in enumerate(uploaded_files):
                    results_data.append({
                        'File': file.name,
                        'Health Score': np.random.uniform(0.4, 0.9),
                        'Disease Detected': np.random.choice([True, False]),
                        'NDVI': np.random.uniform(0.3, 0.8),
                        'Status': np.random.choice(['Healthy', 'Monitoring', 'Alert'])
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"batch_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def show_historical_data():
    """Display historical data and trends"""
    st.header("📈 Historical Data & Trends")
    
    # Generate sample historical data
    history_df = create_sample_history()
    
    tab1, tab2, tab3 = st.tabs(["📊 Trends Analysis", "🏞️ Field Comparison", "📋 Data Export"])
    
    with tab1:
        st.subheader("Field Performance Trends")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=180))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Filter data
        mask = (history_df['date'] >= pd.Timestamp(start_date)) & (history_df['date'] <= pd.Timestamp(end_date))
        filtered_df = history_df.loc[mask]
        
        # Health score trend
        fig_health = px.line(filtered_df, x='date', y='health_score', 
                            title='Health Score Trend Over Time',
                            labels={'health_score': 'Health Score', 'date': 'Date'})
        fig_health.add_hline(y=0.8, line_dash="dash", line_color="green", 
                            annotation_text="Excellent Threshold")
        fig_health.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                            annotation_text="Good Threshold")
        fig_health.add_hline(y=0.4, line_dash="dash", line_color="red", 
                            annotation_text="Alert Threshold")
        st.plotly_chart(fig_health, use_container_width=True)
        
        # NDVI and Yield correlation
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ndvi = px.line(filtered_df, x='date', y='ndvi_avg',
                              title='NDVI Average Over Time',
                              labels={'ndvi_avg': 'NDVI', 'date': 'Date'})
            fig_ndvi.update_traces(line_color='green')
            st.plotly_chart(fig_ndvi, use_container_width=True)
        
        with col2:
            fig_yield = px.line(filtered_df, x='date', y='predicted_yield',
                               title='Predicted Yield Trend',
                               labels={'predicted_yield': 'Yield (kg)', 'date': 'Date'})
            fig_yield.update_traces(line_color='gold')
            st.plotly_chart(fig_yield, use_container_width=True)
        
        # Disease occurrence timeline
        disease_events = filtered_df[filtered_df['disease_detected'] == True]
        if not disease_events.empty:
            st.subheader("🦠 Disease Detection Timeline")
            fig_disease = px.scatter(disease_events, x='date', y='health_score',
                                   title='Disease Detection Events',
                                   labels={'health_score': 'Health Score at Detection', 'date': 'Date'},
                                   color_discrete_sequence=['red'])
            fig_disease.update_traces(marker_size=10)
            st.plotly_chart(fig_disease, use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_health = filtered_df['health_score'].mean()
            st.metric("Average Health Score", f"{avg_health:.3f}")
        
        with col2:
            avg_ndvi = filtered_df['ndvi_avg'].mean()
            st.metric("Average NDVI", f"{avg_ndvi:.3f}")
        
        with col3:
            total_yield = filtered_df['predicted_yield'].sum() / 1000  # Convert to tons
            st.metric("Total Predicted Yield", f"{total_yield:.1f} tons")
        
        with col4:
            disease_rate = (filtered_df['disease_detected'].sum() / len(filtered_df)) * 100
            st.metric("Disease Detection Rate", f"{disease_rate:.1f}%")
    
    with tab2:
        st.subheader("Field Comparison Analysis")
        
        # Multi-field comparison
        fields = create_sample_fields()
        selected_fields = st.multiselect(
            "Select Fields to Compare",
            [f"{f['name']} - {f['crop_type']}" for f in fields],
            default=[f"{fields[0]['name']} - {fields[0]['crop_type']}", 
                    f"{fields[1]['name']} - {fields[1]['crop_type']}"]
        )
        
        if selected_fields:
            # Generate comparison data for selected fields
            comparison_data = []
            for field_name in selected_fields:
                # Generate sample data for each field
                dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
                base_health = np.random.uniform(0.5, 0.9)
                
                for date in dates[-12:]:  # Last 12 weeks
                    comparison_data.append({
                        'Field': field_name,
                        'Date': date,
                        'Health Score': base_health + np.random.normal(0, 0.1),
                        'NDVI': base_health * 0.8 + np.random.normal(0, 0.05),
                        'Yield (kg)': base_health * 50000 + np.random.normal(0, 5000)
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Health score comparison
            fig_comp = px.line(comparison_df, x='Date', y='Health Score', color='Field',
                              title='Health Score Comparison Across Fields')
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Performance metrics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                # Average metrics by field
                avg_metrics = comparison_df.groupby('Field').agg({
                    'Health Score': 'mean',
                    'NDVI': 'mean',
                    'Yield (kg)': 'mean'
                }).round(3)
                
                st.subheader("Average Performance Metrics")
                st.dataframe(avg_metrics, use_container_width=True)
        
            with col2:
                # Field ranking
                field_scores = comparison_df.groupby('Field')['Health Score'].mean().sort_values(ascending=False)
                
                st.subheader("🏆 Field Rankings")
                for i, (field, score) in enumerate(field_scores.items(), 1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
                    st.write(f"{medal} **{i}.** {field} - {score:.3f}")
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        if selected_fields:
            # Calculate correlations
            numeric_cols = ['Health Score', 'NDVI', 'Yield (kg)']
            correlation_matrix = comparison_df[numeric_cols].corr()
            
            # Create heatmap
            fig_corr = px.imshow(correlation_matrix,
                               labels=dict(color="Correlation"),
                               x=numeric_cols,
                               y=numeric_cols,
                               color_continuous_scale='RdBu',
                               aspect="auto",
                               title="Correlation Matrix")
            
            # Add correlation values as text
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    fig_corr.add_annotation(
                        x=j, y=i,
                        text=str(round(correlation_matrix.iloc[i, j], 2)),
                        showarrow=False,
                        font=dict(color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black")
                    )
            
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.subheader("📋 Data Export & Reports")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Quick Export")
            
            export_type = st.selectbox(
                "Data Type",
                ["Historical Analysis", "Field Comparison", "Disease Reports", "Yield Predictions"]
            )
            
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "Excel", "PDF Report", "JSON"]
            )
            
            date_range = st.selectbox(
                "Date Range",
                ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "All Time"]
            )
            
            if st.button("📥 Generate Export", type="primary"):
                # Generate sample export data
                export_data = history_df.copy()
                export_filename = f"{export_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if export_format == "CSV":
                    csv_data = export_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"{export_filename}.csv",
                        mime="text/csv"
                    )
                    st.success("✅ CSV export ready!")
                
                elif export_format == "JSON":
                    json_data = export_data.to_json(orient='records', date_format='iso')
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"{export_filename}.json",
                        mime="application/json"
                    )
                    st.success("✅ JSON export ready!")
                
                else:
                    st.info(f"📋 {export_format} export will be available in the next update")
        
        with col2:
            st.subheader("📈 Automated Reports")
            
            # Report scheduling
            report_frequency = st.selectbox(
                "Report Frequency",
                ["Daily", "Weekly", "Monthly", "Quarterly"]
            )
            
            report_recipients = st.text_area(
                "Email Recipients",
                placeholder="user1@farm.com, user2@farm.com",
                help="Comma-separated email addresses"
            )
            
            include_charts = st.checkbox("Include Charts", value=True)
            include_recommendations = st.checkbox("Include Recommendations", value=True)
            include_alerts = st.checkbox("Include Alerts", value=True)
            
            if st.button("⏰ Schedule Reports"):
                if report_recipients:
                    st.success(f"✅ {report_frequency} reports scheduled for: {report_recipients}")
                else:
                    st.error("❌ Please provide at least one email recipient")
        
        # Data summary
        st.subheader("📊 Data Summary")
        
        summary_stats = {
            "Total Records": len(history_df),
            "Date Range": f"{history_df['date'].min().strftime('%Y-%m-%d')} to {history_df['date'].max().strftime('%Y-%m-%d')}",
            "Average Health Score": f"{history_df['health_score'].mean():.3f}",
            "Disease Events": history_df['disease_detected'].sum(),
            "Health Score Std Dev": f"{history_df['health_score'].std():.3f}",
            "Data Completeness": "100%"
        }
        
        for key, value in summary_stats.items():
            st.write(f"**{key}:** {value}")

def show_alerts_reports():
    """Display alerts and reports interface"""
    st.header("🚨 Alerts & Reports")
    
    tab1, tab2, tab3 = st.tabs(["🔔 Active Alerts", "📊 Custom Reports", "⚙️ Alert Settings"])
    
    with tab1:
        st.subheader("Current Alerts")
        
        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                '<div class="alert-card"><h4>🔴 Critical</h4><h2>2</h2></div>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                '<div style="background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); padding: 1rem; border-radius: 10px; border-left: 5px solid #FF9800;"><h4>🟡 Warning</h4><h2>5</h2></div>',
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                '<div class="info-card"><h4>🔵 Info</h4><h2>3</h2></div>',
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                '<div class="success-card"><h4>✅ Resolved</h4><h2>12</h2></div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Alert list
        alerts = [
            {
                "id": 1,
                "level": "Critical",
                "title": "Disease Outbreak Detected",
                "description": "Severe leaf blight detected in West Field (Area: 15.2 ha)",
                "field": "West Field",
                "time": "2 hours ago",
                "status": "Active",
                "priority": 1
            },
            {
                "id": 2,
                "level": "Critical",
                "title": "Irrigation System Failure",
                "description": "Water pressure drop detected in North Field irrigation zone 3",
                "field": "North Field",
                "time": "4 hours ago",
                "status": "Active",
                "priority": 1
            },
            {
                "id": 3,
                "level": "Warning",
                "title": "Low Soil Moisture",
                "description": "Soil moisture below optimal levels (18% vs 25% target)",
                "field": "South Field",
                "time": "6 hours ago",
                "status": "Monitoring",
                "priority": 2
            },
            {
                "id": 4,
                "level": "Warning",
                "title": "Temperature Stress",
                "description": "Prolonged high temperatures (>35°C) detected",
                "field": "East Field",
                "time": "8 hours ago",
                "status": "Monitoring",
                "priority": 2
            },
            {
                "id": 5,
                "level": "Info",
                "title": "Harvest Window Opening",
                "description": "Optimal harvest conditions predicted for next week",
                "field": "Central Field",
                "time": "1 day ago",
                "status": "Scheduled",
                "priority": 3
            }
        ]
        
        # Filter and sort alerts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            level_filter = st.selectbox("Filter by Level", ["All", "Critical", "Warning", "Info"])
        
        with col2:
            status_filter = st.selectbox("Filter by Status", ["All", "Active", "Monitoring", "Scheduled", "Resolved"])
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Priority", "Time", "Field", "Level"])
        
        # Apply filters
        filtered_alerts = alerts.copy()
        if level_filter != "All":
            filtered_alerts = [a for a in filtered_alerts if a["level"] == level_filter]
        if status_filter != "All":
            filtered_alerts = [a for a in filtered_alerts if a["status"] == status_filter]
        
        # Display alerts
        for alert in filtered_alerts:
            with st.expander(f"{alert['level']} - {alert['title']} ({alert['time']})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Field:** {alert['field']}")
                    st.write(f"**Description:** {alert['description']}")
                    st.write(f"**Status:** {alert['status']}")
                    st.write(f"**Time:** {alert['time']}")
                
                with col2:
                    level_colors = {
                        "Critical": "🔴",
                        "Warning": "🟡",
                        "Info": "🔵"
                    }
                    st.write(f"**Priority:** {level_colors[alert['level']]} {alert['level']}")
                    
                    if st.button(f"📋 View Details", key=f"details_{alert['id']}"):
                        st.info("Detailed alert information would open here")
                    
                    if alert['status'] == 'Active':
                        if st.button(f"✅ Mark Resolved", key=f"resolve_{alert['id']}"):
                            st.success("Alert marked as resolved!")
    
    with tab2:
        st.subheader("📊 Custom Report Builder")
        
        # Report configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Report Configuration")
            
            report_name = st.text_input("Report Name", value="Monthly Farm Analysis")
            report_type = st.selectbox(
                "Report Type",
                ["Executive Summary", "Technical Analysis", "Field Performance", "Disease Report", "Yield Analysis"]
            )
            
            # Date range
            col1a, col1b = st.columns(2)
            with col1a:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
            with col1b:
                end_date = st.date_input("End Date", value=datetime.now())
            
            # Field selection
            fields = create_sample_fields()
            selected_fields = st.multiselect(
                "Select Fields",
                [f"{f['name']} - {f['crop_type']}" for f in fields],
                default=[f"{f['name']} - {f['crop_type']}" for f in fields[:3]]
            )
        
        with col2:
            st.subheader("Report Sections")
            
            include_summary = st.checkbox("Executive Summary", value=True)
            include_health = st.checkbox("Health Analysis", value=True)
            include_yield = st.checkbox("Yield Predictions", value=True)
            include_disease = st.checkbox("Disease Reports", value=True)
            include_recommendations = st.checkbox("Recommendations", value=True)
            include_charts = st.checkbox("Charts & Visualizations", value=True)
            include_raw_data = st.checkbox("Raw Data Tables", value=False)
            
            # Output format
            output_format = st.selectbox("Output Format", ["PDF", "HTML", "Word Document", "PowerPoint"])
        
        # Report preview
        if st.button("📋 Generate Report Preview", type="primary"):
            st.subheader("📄 Report Preview")
            
            # Mock report content
            st.markdown("## Executive Summary")
            st.write("This report covers the period from {} to {} for {} selected fields.".format(
                start_date.strftime('%B %d, %Y'),
                end_date.strftime('%B %d, %Y'),
                len(selected_fields)
            ))
            
            if include_health:
                st.markdown("## Health Analysis")
                # Generate sample health chart
                sample_data = pd.DataFrame({
                    'Field': selected_fields[:3],
                    'Health Score': [0.85, 0.72, 0.91],
                    'Status': ['Healthy', 'Monitoring', 'Excellent']
                })
                
                fig = px.bar(sample_data, x='Field', y='Health Score', 
                           title='Field Health Scores', color='Health Score',
                           color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
            
            if include_recommendations:
                st.markdown("## Key Recommendations")
                recommendations = [
                    "Increase irrigation frequency in South Field",
                    "Apply preventive fungicide treatment to West Field",
                    "Schedule soil testing for North Field",
                    "Monitor weather conditions for optimal harvest timing"
                ]
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            
            # Generate actual report button
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📧 Email Report"):
                    st.success("Report emailed successfully!")
            with col2:
                if st.button("💾 Save Report"):
                    st.success("Report saved to reports library!")
            with col3:
                if st.button("📥 Download Report"):
                    st.success("Report download started!")
    
    with tab3:
        st.subheader("⚙️ Alert Configuration")
        
        # Alert thresholds
        st.subheader("🎯 Alert Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Health Score Thresholds**")
            critical_health = st.slider("Critical Alert", 0.0, 1.0, 0.3, 0.01)
            warning_health = st.slider("Warning Alert", 0.0, 1.0, 0.5, 0.01)
            
            st.write("**NDVI Thresholds**")
            critical_ndvi = st.slider("Critical NDVI", 0.0, 1.0, 0.2, 0.01)
            warning_ndvi = st.slider("Warning NDVI", 0.0, 1.0, 0.4, 0.01)
        
        with col2:
            st.write("**Environmental Thresholds**")
            max_temp = st.number_input("Maximum Temperature (°C)", value=35.0)
            min_temp = st.number_input("Minimum Temperature (°C)", value=5.0)
            min_humidity = st.number_input("Minimum Humidity (%)", value=30.0)
            max_wind = st.number_input("Maximum Wind Speed (km/h)", value=50.0)
            
            st.write("**Soil Conditions**")
            min_soil_moisture = st.number_input("Minimum Soil Moisture (%)", value=20.0)
            max_soil_moisture = st.number_input("Maximum Soil Moisture (%)", value=80.0)
        
        # Notification settings
        st.subheader("📧 Notification Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Email Notifications**")
            email_critical = st.checkbox("Critical Alerts", value=True)
            email_warning = st.checkbox("Warning Alerts", value=True)
            email_info = st.checkbox("Info Alerts", value=False)
            
            email_addresses = st.text_area("Email Recipients", 
                                         value="farm.manager@example.com\nagronomist@example.com")
        
        with col2:
            st.write("**SMS Notifications**")
            sms_critical = st.checkbox("Critical SMS", value=True)
            sms_warning = st.checkbox("Warning SMS", value=False)
            
            phone_numbers = st.text_area("Phone Numbers",
                                       value="+1234567890\n+0987654321")
            
            st.write("**Push Notifications**")
            push_enabled = st.checkbox("Enable Push Notifications", value=True)
        
        # Alert frequency
        st.subheader("⏰ Alert Frequency")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alert_frequency = st.selectbox(
                "Check Frequency",
                ["Every 15 minutes", "Every 30 minutes", "Every hour", "Every 4 hours", "Daily"]
            )
        
        with col2:
            quiet_hours = st.checkbox("Enable Quiet Hours", value=True)
            if quiet_hours:
                quiet_start = st.time_input("Quiet Hours Start", value=datetime.strptime("22:00", "%H:%M").time())
                quiet_end = st.time_input("Quiet Hours End", value=datetime.strptime("06:00", "%H:%M").time())
        
        # Save settings
        if st.button("💾 Save Alert Settings", type="primary"):
            st.success("✅ Alert settings saved successfully!")
            
            # Show confirmation of settings
            with st.expander("📋 Settings Summary"):
                st.json({
                    "health_thresholds": {
                        "critical": critical_health,
                        "warning": warning_health
                    },
                    "ndvi_thresholds": {
                        "critical": critical_ndvi,
                        "warning": warning_ndvi
                    },
                    "notifications": {
                        "email_critical": email_critical,
                        "email_warning": email_warning,
                        "sms_critical": sms_critical,
                        "push_enabled": push_enabled
                    },
                    "frequency": alert_frequency,
                    "quiet_hours": quiet_hours
                })

def show_settings():
    """Display settings and configuration"""
    st.header("⚙️ Settings & Configuration")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔗 API Settings", "🎨 Display Preferences", "👥 User Management", "🔒 Security"])
    
    with tab1:
        st.subheader("API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Backend Connection**")
            api_url = st.text_input("API Base URL", value="http://localhost:8000")
            api_timeout = st.number_input("Request Timeout (seconds)", value=30, min_value=1, max_value=300)
            
            # Test connection
            if st.button("🔍 Test Connection"):
                try:
                    response = requests.get(f"{api_url}/dashboard/summary", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Connection successful!")
                    else:
                        st.error(f"❌ Connection failed: HTTP {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
        
        with col2:
            st.write("**Model Configuration**")
            model_path = st.text_input("YOLOv8 Model Path", value="yolov8n-seg.pt")
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
            iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01)
            
            # Model info
            st.info("💡 Upload custom trained models through the file manager")
        
        # Database settings
        st.subheader("📊 Database Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            db_host = st.text_input("Database Host", value="localhost")
            db_port = st.number_input("Database Port", value=5432)
            db_name = st.text_input("Database Name", value="agri_monitoring")
        
        with col2:
            db_user = st.text_input("Database User", value="postgres")
            db_password = st.text_input("Database Password", type="password")
            
            if st.button("🔍 Test Database Connection"):
                st.info("Database connection test would be performed here")
        
        # Data retention
        st.subheader("🗄️ Data Retention")
        
        col1, col2 = st.columns(2)
        
        with col1:
            image_retention = st.selectbox("Image Retention Period", 
                                         ["30 days", "90 days", "6 months", "1 year", "Permanent"])
            analysis_retention = st.selectbox("Analysis Data Retention",
                                            ["6 months", "1 year", "2 years", "5 years", "Permanent"])
        
        with col2:
            auto_cleanup = st.checkbox("Enable Auto Cleanup", value=True)
            compress_old_data = st.checkbox("Compress Old Data", value=True)
    
    with tab2:
        st.subheader("Display Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Theme Settings**")
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            color_scheme = st.selectbox("Color Scheme", ["Green", "Blue", "Custom"])
            
            if color_scheme == "Custom":
                primary_color = st.color_picker("Primary Color", "#4CAF50")
                secondary_color = st.color_picker("Secondary Color", "#2196F3")
            
            st.write("**Language**")
            language = st.selectbox("Display Language", ["English", "Spanish", "French", "German", "Chinese"])
        
        with col2:
            st.write("**Dashboard Layout**")
            default_page = st.selectbox("Default Page", 
                                      ["Dashboard", "Field Management", "Analysis & Monitoring"])
            
            show_tips = st.checkbox("Show Helpful Tips", value=True)
            compact_view = st.checkbox("Compact View", value=False)
            
            st.write("**Chart Preferences**")
            chart_style = st.selectbox("Chart Style", ["Modern", "Classic", "Minimal"])
            animation_speed = st.selectbox("Animation Speed", ["Slow", "Normal", "Fast", "None"])
        
        # Units and formats
        st.subheader("📏 Units & Formats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature_unit = st.selectbox("Temperature", ["Celsius", "Fahrenheit"])
            area_unit = st.selectbox("Area", ["Hectares", "Acres", "Square Meters"])
            distance_unit = st.selectbox("Distance", ["Meters", "Feet", "Kilometers", "Miles"])
        
        with col2:
            date_format = st.selectbox("Date Format", ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"])
            time_format = st.selectbox("Time Format", ["24 Hour", "12 Hour AM/PM"])
            number_format = st.selectbox("Number Format", ["1,234.56", "1.234,56", "1 234,56"])
    
    with tab3:
        st.subheader("User Management")
        
        # Current user info
        st.write("**Current User**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Username", value="farm_manager", disabled=True)
            st.text_input("Email", value="manager@farm.com")
            st.selectbox("Role", ["Admin", "Manager", "Analyst", "Viewer"], index=0)
        
        with col2:
            st.text_input("Full Name", value="John Smith")
            st.text_input("Phone", value="+1234567890")
            if st.button("📝 Update Profile"):
                st.success("✅ Profile updated successfully!")
        
        # User list
        st.subheader("👥 Team Members")
        
        users = [
            {"id": 1, "name": "John Smith", "email": "manager@farm.com", "role": "Admin", "status": "Active", "last_login": "2 hours ago"},
            {"id": 2, "name": "Sarah Johnson", "email": "sarah@farm.com", "role": "Analyst", "status": "Active", "last_login": "1 day ago"},
            {"id": 3, "name": "Mike Wilson", "email": "mike@farm.com", "role": "Manager", "status": "Active", "last_login": "3 days ago"},
            {"id": 4, "name": "Lisa Chen", "email": "lisa@farm.com", "role": "Viewer", "status": "Inactive", "last_login": "2 weeks ago"}
        ]
        
        df_users = pd.DataFrame(users)
        
        # Add action buttons
        for i, user in enumerate(users):
            with st.expander(f"{user['name']} - {user['role']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Email:** {user['email']}")
                    st.write(f"**Status:** {user['status']}")
                
                with col2:
                    st.write(f"**Last Login:** {user['last_login']}")
                    new_role = st.selectbox("Change Role", 
                                          ["Admin", "Manager", "Analyst", "Viewer"], 
                                          index=["Admin", "Manager", "Analyst", "Viewer"].index(user['role']),
                                          key=f"role_{user['id']}")
                
                with col3:
                    if st.button(f"🔒 Reset Password", key=f"reset_{user['id']}"):
                        st.success("Password reset email sent!")
                    
                    if user['status'] == 'Active':
                        if st.button(f"❌ Deactivate", key=f"deactivate_{user['id']}"):
                            st.warning("User deactivated")
                    else:
                        if st.button(f"✅ Activate", key=f"activate_{user['id']}"):
                            st.success("User activated")
        
        # Add new user
        st.subheader("➕ Add New User")
        
        with st.form("add_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input("Full Name*")
                new_email = st.text_input("Email*")
                new_role = st.selectbox("Role*", ["Viewer", "Analyst", "Manager", "Admin"])
            
            with col2:
                new_phone = st.text_input("Phone Number")
                send_invite = st.checkbox("Send Welcome Email", value=True)
                temp_password = st.text_input("Temporary Password", type="password")
            
            if st.form_submit_button("👤 Add User", type="primary"):
                if new_name and new_email and new_role:
                    st.success(f"✅ User '{new_name}' added successfully!")
                    if send_invite:
                        st.info("📧 Welcome email sent")
                else:
                    st.error("❌ Please fill in all required fields")
    
    with tab4:
        st.subheader("Security Settings")
        
        # Authentication settings
        st.write("**Authentication**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_2fa = st.checkbox("Enable Two-Factor Authentication", value=True)
            session_timeout = st.selectbox("Session Timeout", 
                                         ["15 minutes", "30 minutes", "1 hour", "4 hours", "8 hours", "Never"])
            
            password_policy = st.selectbox("Password Policy", 
                                         ["Basic", "Medium", "Strong", "Enterprise"])
            
            if password_policy == "Strong":
                st.info("🔒 Strong policy: Min 12 chars, uppercase, lowercase, numbers, symbols")
        
        with col2:
            login_attempts = st.number_input("Max Login Attempts", value=5, min_value=3, max_value=10)
            lockout_duration = st.selectbox("Account Lockout Duration",
                                          ["5 minutes", "15 minutes", "30 minutes", "1 hour", "24 hours"])
            
            enable_captcha = st.checkbox("Enable CAPTCHA", value=True)
            enable_audit_log = st.checkbox("Enable Audit Logging", value=True)
        
        # API Security
        st.subheader("🔐 API Security")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**API Access**")
            api_key_rotation = st.selectbox("API Key Rotation", 
                                          ["Never", "Monthly", "Quarterly", "Annually"])
            
            rate_limiting = st.checkbox("Enable Rate Limiting", value=True)
            if rate_limiting:
                requests_per_minute = st.number_input("Requests per Minute", value=100, min_value=10)
        
        with col2:
            st.write("**Access Control**")
            ip_whitelist = st.text_area("IP Whitelist (one per line)", 
                                      placeholder="192.168.1.0/24\n10.0.0.0/8")
            
            cors_origins = st.text_area("CORS Origins (one per line)",
                                      placeholder="https://farm.com\nhttps://mobile.farm.com")
        
        # Data encryption
        st.subheader("🛡️ Data Protection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            encryption_at_rest = st.checkbox("Encrypt Data at Rest", value=True)
            encryption_in_transit = st.checkbox("Encrypt Data in Transit", value=True)
            
            backup_encryption = st.checkbox("Encrypt Backups", value=True)
            secure_deletion = st.checkbox("Secure Data Deletion", value=True)
        
        with col2:
            data_anonymization = st.checkbox("Enable Data Anonymization", value=False)
            gdpr_compliance = st.checkbox("GDPR Compliance Mode", value=True)
            
            retention_enforcement = st.checkbox("Enforce Data Retention", value=True)
            audit_trail = st.checkbox("Maintain Audit Trail", value=True)
        
        # Security monitoring
        st.subheader("👁️ Security Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            intrusion_detection = st.checkbox("Intrusion Detection", value=True)
            anomaly_detection = st.checkbox("Anomaly Detection", value=True)
            
            failed_login_alerts = st.checkbox("Failed Login Alerts", value=True)
            suspicious_activity_alerts = st.checkbox("Suspicious Activity Alerts", value=True)
        
        with col2:
            security_scan_frequency = st.selectbox("Security Scan Frequency",
                                                  ["Daily", "Weekly", "Monthly"])
            
            vulnerability_alerts = st.checkbox("Vulnerability Alerts", value=True)
            security_reports = st.checkbox("Weekly Security Reports", value=True)
        
        # Save all settings
        if st.button("💾 Save All Settings", type="primary"):
            st.success("✅ All settings saved successfully!")
            
            # Show summary
            with st.expander("📋 Settings Summary"):
                st.write("**Security Settings Applied:**")
                st.write(f"- Two-Factor Authentication: {'Enabled' if enable_2fa else 'Disabled'}")
                st.write(f"- Session Timeout: {session_timeout}")
                st.write(f"- Password Policy: {password_policy}")
                st.write(f"- Rate Limiting: {'Enabled' if rate_limiting else 'Disabled'}")
                st.write(f"- Data Encryption: {'Enabled' if encryption_at_rest else 'Disabled'}")
                st.write(f"- Audit Logging: {'Enabled' if enable_audit_log else 'Disabled'}")

# Additional utility functions
def display_weather_widget():
    """Display weather information widget"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌤️ Current Weather")
    
    # Mock weather data
    weather_data = {
        "temperature": "24°C",
        "humidity": "65%",
        "wind_speed": "12 km/h",
        "conditions": "Partly Cloudy",
        "precipitation": "5%"
    }
    
    for key, value in weather_data.items():
        st.sidebar.write(f"**{key.replace('_', ' ').title()}:** {value}")

def display_quick_actions():
    """Display quick action buttons in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Actions")
    
    if st.sidebar.button("🔍 New Analysis", use_container_width=True):
        st.session_state.current_page = "Analysis & Monitoring"
        st.rerun()
    
    if st.sidebar.button("📊 Generate Report", use_container_width=True):
        st.session_state.current_page = "Alerts & Reports"
        st.rerun()
    
    if st.sidebar.button("➕ Add Field", use_container_width=True):
        st.session_state.current_page = "Field Management"
        st.rerun()
    
    if st.sidebar.button("🚨 View Alerts", use_container_width=True):
        st.session_state.current_page = "Alerts & Reports"
        st.rerun()

def display_system_status():
    """Display system status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🖥️ System Status")
    
    # Mock system status
    status_items = [
        ("API Server", "🟢", "Online"),
        ("Database", "🟢", "Connected"),
        ("ML Models", "🟢", "Ready"),
        ("Weather Service", "🟡", "Limited"),
        ("Satellite Data", "🟢", "Updated")
    ]
    
    for service, indicator, status in status_items:
        st.sidebar.write(f"{indicator} **{service}:** {status}")

# Run the application
if __name__ == "__main__":
    # Run main application
    main()
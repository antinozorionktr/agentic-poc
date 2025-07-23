import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import base64
import io
import json
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import time

# Page configuration
st.set_page_config(
    page_title="Agricultural Monitoring Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #2E8B57, #32CD32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .health-good { color: #28a745; font-weight: bold; }
    .health-warning { color: #ffc107; font-weight: bold; }
    .health-critical { color: #dc3545; font-weight: bold; }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2E8B57 0%, #32CD32 100%);
    }
</style>
""", unsafe_allow_html=True)

def call_api(endpoint: str, method: str = "GET", files=None, data=None, params=None):
    """Make API calls to the backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Please ensure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def display_health_status(health_score):
    """Display color-coded health status"""
    if health_score >= 80:
        return f'<span class="health-good">Excellent ({health_score:.1f}%)</span>'
    elif health_score >= 60:
        return f'<span class="health-warning">Good ({health_score:.1f}%)</span>'
    elif health_score >= 40:
        return f'<span class="health-warning">Fair ({health_score:.1f}%)</span>'
    else:
        return f'<span class="health-critical">Poor ({health_score:.1f}%)</span>'

def create_segmentation_visualization(image, segments):
    """Create visualization of segmentation results"""
    if not segments:
        return image
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    overlay = img_array.copy()
    
    # Color palette for different crops
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    for i, segment in enumerate(segments):
        try:
            # Decode mask data
            mask_data = base64.b64decode(segment['mask_data'])
            mask_img = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            
            if mask_img is not None:
                # Resize mask to match image size
                mask_resized = cv2.resize(mask_img, (img_array.shape[1], img_array.shape[0]))
                
                # Apply color overlay
                color = colors[i % len(colors)]
                colored_mask = np.zeros_like(img_array)
                colored_mask[mask_resized > 127] = color
                
                # Blend with original image
                overlay = cv2.addWeighted(overlay, 0.8, colored_mask, 0.3, 0)
                
        except Exception as e:
            st.warning(f"Could not display segment {i}: {str(e)}")
    
    return Image.fromarray(overlay)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Agricultural Monitoring Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?w=300&h=200&fit=crop", 
                caption="Smart Agriculture", use_column_width=True)
        
        st.markdown("### 🚀 Navigation")
        page = st.selectbox(
            "Select Analysis Type",
            ["🏠 Dashboard", "📸 Crop Analysis", "📊 Yield Prediction", "🛰️ Satellite Data", "📈 Analytics", "📋 History"]
        )
        
        st.markdown("### ⚙️ Settings")
        api_status = call_api("/health")
        if api_status:
            st.success("✅ API Connected")
            if api_status.get('model_loaded'):
                st.success("✅ AI Model Loaded")
            else:
                st.warning("⚠️ AI Model Not Loaded")
        else:
            st.error("❌ API Disconnected")
    
    # Main content based on page selection
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📸 Crop Analysis":
        show_crop_analysis()
    elif page == "📊 Yield Prediction":
        show_yield_prediction()
    elif page == "🛰️ Satellite Data":
        show_satellite_data()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "📋 History":
        show_history()

def show_dashboard():
    """Main dashboard with overview metrics"""
    st.markdown("## 📊 Farm Overview Dashboard")
    
    # Get recent analyses for dashboard metrics
    history_data = call_api("/history/analyses", params={"limit": 5})
    
    if history_data:
        # Calculate summary metrics
        total_analyses = len(history_data)
        avg_health = np.mean([
            np.mean([crop['health_score'] for crop in json.loads(item['crop_data'])])
            for item in history_data if item['crop_data']
        ]) if history_data else 0
        
        disease_count = sum([
            sum([1 for crop in json.loads(item['crop_data']) if crop['disease_detected']])
            for item in history_data if item['crop_data']
        ])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📈 Recent Analyses",
                value=total_analyses,
                delta=f"+{total_analyses} this week"
            )
        
        with col2:
            st.metric(
                label="🌱 Average Health Score",
                value=f"{avg_health:.1f}%",
                delta="+2.3%" if avg_health > 70 else "-1.2%"
            )
        
        with col3:
            st.metric(
                label="⚠️ Disease Alerts",
                value=disease_count,
                delta=f"-{max(0, disease_count-1)} from last week" if disease_count > 0 else "No diseases"
            )
        
        with col4:
            st.metric(
                label="📅 Last Update",
                value="Today",
                delta="Real-time monitoring"
            )
    
    # Farm map
    st.markdown("### 🗺️ Farm Locations")
    
    # Create a sample farm map
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
    
    # Add sample farm locations
    farm_locations = [
        {"name": "North Field", "lat": 40.7200, "lon": -74.0100, "health": 85, "crop": "Wheat"},
        {"name": "South Field", "lat": 40.7050, "lon": -74.0200, "health": 72, "crop": "Corn"},
        {"name": "East Field", "lat": 40.7150, "lon": -73.9900, "health": 91, "crop": "Soybean"},
    ]
    
    for farm in farm_locations:
        color = "green" if farm["health"] > 80 else "orange" if farm["health"] > 60 else "red"
        folium.Marker(
            [farm["lat"], farm["lon"]],
            popup=f"<b>{farm['name']}</b><br>Crop: {farm['crop']}<br>Health: {farm['health']}%",
            tooltip=farm["name"],
            icon=folium.Icon(color=color, icon="leaf")
        ).add_to(m)
    
    map_data = st_folium(m, width=700, height=400)
    
    # Recent activity timeline
    st.markdown("### 📅 Recent Activity")
    
    if history_data:
        timeline_df = pd.DataFrame([
            {
                "Date": item["upload_date"][:10],
                "File": item["filename"],
                "Crops Detected": len(json.loads(item["crop_data"])) if item["crop_data"] else 0,
                "Status": json.loads(item["health_metrics"])["overall_status"] if item["health_metrics"] else "unknown"
            }
            for item in history_data
        ])
        
        st.dataframe(timeline_df, use_container_width=True)
    else:
        st.info("No recent analyses found. Upload an image to get started!")

def show_crop_analysis():
    """Crop analysis page"""
    st.markdown("## 📸 Crop Segmentation & Health Analysis")
    
    # Add tabs for single vs batch analysis
    tab1, tab2 = st.tabs(["Single Image", "Multiple Images (Batch)"])
    
    with tab1:
        show_single_crop_analysis()
    
    with tab2:
        show_batch_crop_analysis()

def show_single_crop_analysis():
    """Single image crop analysis"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload drone footage or field images for analysis",
            key="single_crop_upload"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analysis button
            if st.button("🔍 Analyze Crops", type="primary", key="single_analyze"):
                with st.spinner("Analyzing crops... This may take a moment."):
                    try:
                        # Reset file pointer and prepare for API call
                        uploaded_file.seek(0)
                        
                        # Prepare files properly for requests
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type or "image/jpeg")}
                        result = call_api("/analyze/crop-segmentation", method="POST", files=files)
                        
                        if result:
                            st.session_state['analysis_result'] = result
                            st.session_state['original_image'] = image
                            st.success("✅ Analysis completed!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    
    with col2:
        st.markdown("### Analysis Results")
        
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            image = st.session_state['original_image']
            
            # Display segmentation visualization
            if result['segments']:
                segmented_image = create_segmentation_visualization(image, result['segments'])
                st.image(segmented_image, caption="Segmented Crops", use_column_width=True)
            
            # Health summary
            health_summary = result['health_summary']
            st.markdown("#### 🏥 Health Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                st.metric("Total Segments", health_summary['total_segments'])
                st.metric("Healthy Segments", health_summary['healthy_segments'])
            with summary_col2:
                st.metric("Diseased Segments", health_summary['diseased_segments'])
                avg_health = health_summary['average_health_score']
                st.markdown(f"**Average Health:** {display_health_status(avg_health)}", unsafe_allow_html=True)
            
            # Detailed analysis
            st.markdown("#### 📋 Detailed Analysis")
            analysis_data = []
            for i, analysis in enumerate(result['analysis']):
                analysis_data.append({
                    "Segment": i + 1,
                    "Crop Type": analysis['crop_type'].title(),
                    "Area (ha)": f"{analysis['area_hectares']:.4f}",
                    "Health Score": f"{analysis['health_score']:.1f}%",
                    "Disease Status": "⚠️ Detected" if analysis['disease_detected'] else "✅ Healthy",
                    "Confidence": f"{analysis['confidence']:.2f}"
                })
            
            df = pd.DataFrame(analysis_data)
            st.dataframe(df, use_container_width=True)
            
            # Charts
            if len(result['analysis']) > 1:
                st.markdown("#### 📊 Health Distribution")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Health score distribution
                    health_scores = [a['health_score'] for a in result['analysis']]
                    crop_types = [a['crop_type'] for a in result['analysis']]
                    
                    fig_health = px.bar(
                        x=crop_types,
                        y=health_scores,
                        title="Health Scores by Crop Type",
                        labels={'x': 'Crop Type', 'y': 'Health Score (%)'},
                        color=health_scores,
                        color_continuous_scale="RdYlGn"
                    )
                    st.plotly_chart(fig_health, use_container_width=True)
                
                with chart_col2:
                    # Area distribution
                    areas = [a['area_hectares'] for a in result['analysis']]
                    
                    fig_area = px.pie(
                        values=areas,
                        names=crop_types,
                        title="Area Distribution by Crop Type"
                    )
                    st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("👆 Upload an image and click 'Analyze Crops' to see results here.")

def show_batch_crop_analysis():
    """Batch crop analysis for multiple images"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload Multiple Images")
        uploaded_files = st.file_uploader(
            "Choose images...",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple drone footage or field images for batch analysis",
            key="batch_crop_upload"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} images selected:**")
            for i, file in enumerate(uploaded_files[:3]):  # Show first 3 images
                col_img1, col_img2 = st.columns([1, 3])
                with col_img1:
                    image = Image.open(file)
                    st.image(image, caption=f"{file.name}", use_column_width=True)
                with col_img2:
                    st.write(f"**File:** {file.name}")
                    st.write(f"**Size:** {len(file.getvalue()) / 1024:.1f} KB")
            
            if len(uploaded_files) > 3:
                st.write(f"... and {len(uploaded_files) - 3} more images")
            
            # Batch analysis button
            if st.button("🔍 Analyze All Images", type="primary", key="batch_analyze"):
                with st.spinner(f"Analyzing {len(uploaded_files)} images... This may take a few minutes."):
                    try:
                        # Prepare files for batch API call
                        files = []
                        for file in uploaded_files:
                            file.seek(0)
                            files.append(("files", (file.name, file, file.type or "image/jpeg")))
                        
                        # Call batch analysis API
                        import requests
                        response = requests.post(f"{API_BASE_URL}/analyze/batch-crop-segmentation", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['batch_analysis_result'] = result
                            st.success(f"✅ Batch analysis completed! Processed {result['images_processed']} images.")
                            st.rerun()
                        else:
                            st.error(f"API Error: {response.status_code} - {response.text}")
                            
                    except Exception as e:
                        st.error(f"Error during batch analysis: {str(e)}")
    
    with col2:
        st.markdown("### Batch Analysis Results")
        
        if 'batch_analysis_result' in st.session_state:
            result = st.session_state['batch_analysis_result']
            
            # Combined summary metrics
            st.markdown("#### 📊 Combined Summary")
            summary = result['combined_health_summary']
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Images Processed", f"{summary['images_processed']}/{summary['total_images']}")
            with metric_col2:
                st.metric("Total Segments", summary['total_segments'])
            with metric_col3:
                st.metric("Total Area", f"{summary['total_area_hectares']:.2f} ha")
            with metric_col4:
                avg_health = summary['average_health_score']
                st.markdown(f"**Avg Health:** {display_health_status(avg_health)}", unsafe_allow_html=True)
            
            # Crop distribution
            if summary['crop_distribution']:
                st.markdown("#### 🌾 Crop Distribution Across All Images")
                
                crop_data = []
                for crop_type, data in summary['crop_distribution'].items():
                    crop_data.append({
                        "Crop Type": crop_type.title(),
                        "Count": data['count'],
                        "Total Area (ha)": f"{data['area']:.4f}",
                        "Avg Health": f"{data['avg_health']:.1f}%"
                    })
                
                df_crops = pd.DataFrame(crop_data)
                st.dataframe(df_crops, use_container_width=True)
                
                # Visualization
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Area by crop type
                    crop_types = [data['Crop Type'] for data in crop_data]
                    areas = [float(data['Total Area (ha)'].replace(' ha', '')) for data in crop_data]
                    
                    fig_area = px.pie(
                        values=areas,
                        names=crop_types,
                        title="Total Area by Crop Type"
                    )
                    st.plotly_chart(fig_area, use_container_width=True)
                
                with chart_col2:
                    # Health by crop type
                    health_scores = [float(data['Avg Health'].replace('%', '')) for data in crop_data]
                    
                    fig_health = px.bar(
                        x=crop_types,
                        y=health_scores,
                        title="Average Health by Crop Type",
                        color=health_scores,
                        color_continuous_scale="RdYlGn"
                    )
                    st.plotly_chart(fig_health, use_container_width=True)
            
            # Individual image results
            st.markdown("#### 📋 Individual Image Results")
            
            individual_data = []
            for img_result in result['individual_results']:
                status_icon = "✅" if img_result['status'] == 'success' else "❌"
                individual_data.append({
                    "Image": f"{status_icon} {img_result['filename']}",
                    "Segments": img_result['segments_count'],
                    "Area (ha)": f"{img_result['area']:.4f}",
                    "Avg Health": f"{img_result.get('average_health', 0):.1f}%" if img_result.get('average_health') else "N/A",
                    "Diseases": img_result.get('disease_count', 0),
                    "Crop Types": ", ".join(img_result.get('crop_types', [])) if img_result.get('crop_types') else "None"
                })
            
            df_individual = pd.DataFrame(individual_data)
            st.dataframe(df_individual, use_container_width=True)
            
        else:
            st.info("👆 Upload multiple images and click 'Analyze All Images' to see batch results here.")
            st.markdown(f"**Average Health:** {display_health_status(avg_health)}", unsafe_allow_html=True)

            # Detailed analysis
            st.markdown("#### 📋 Detailed Analysis")
            analysis_data = []
            for i, analysis in enumerate(result['analysis']):
                analysis_data.append({
                    "Segment": i + 1,
                    "Crop Type": analysis['crop_type'].title(),
                    "Area (ha)": f"{analysis['area_hectares']:.4f}",
                    "Health Score": f"{analysis['health_score']:.1f}%",
                    "Disease Status": "⚠️ Detected" if analysis['disease_detected'] else "✅ Healthy",
                    "Confidence": f"{analysis['confidence']:.2f}"
                })
            
            df = pd.DataFrame(analysis_data)
            st.dataframe(df, use_container_width=True)
            
            # Charts
            if len(result['analysis']) > 1:
                st.markdown("#### 📊 Health Distribution")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Health score distribution
                    health_scores = [a['health_score'] for a in result['analysis']]
                    crop_types = [a['crop_type'] for a in result['analysis']]
                    
                    fig_health = px.bar(
                        x=crop_types,
                        y=health_scores,
                        title="Health Scores by Crop Type",
                        labels={'x': 'Crop Type', 'y': 'Health Score (%)'},
                        color=health_scores,
                        color_continuous_scale="RdYlGn"
                    )
                    st.plotly_chart(fig_health, use_container_width=True)
                
                with chart_col2:
                    # Area distribution
                    areas = [a['area_hectares'] for a in result['analysis']]
                    
                    fig_area = px.pie(
                        values=areas,
                        names=crop_types,
                        title="Area Distribution by Crop Type"
                    )
                    st.plotly_chart(fig_area, use_container_width=True)
        # else:
        #     st.info("👆 Upload an image and click 'Analyze Crops' to see results here.")

def show_yield_prediction():
    """Yield prediction page"""
    st.markdown("## 📊 Crop Yield Prediction")
    
    # Add tabs for single vs batch prediction
    tab1, tab2 = st.tabs(["Single Image", "Multiple Images (Batch)"])
    
    with tab1:
        show_single_yield_prediction()
    
    with tab2:
        show_batch_yield_prediction()

def show_single_yield_prediction():
    """Single image yield prediction"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload Field Image")
        uploaded_file = st.file_uploader(
            "Upload image for yield analysis...",
            type=['png', 'jpg', 'jpeg'],
            key="single_yield_upload"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Field Image", use_column_width=True)
            
            if st.button("🎯 Predict Yield", type="primary", key="single_yield_predict"):
                with st.spinner("Analyzing yield potential..."):
                    try:
                        # Reset file pointer and prepare for API call
                        uploaded_file.seek(0)
                        
                        # Prepare files properly for requests
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type or "image/jpeg")}
                        result = call_api("/predict/yield", method="POST", files=files)
                        
                        if result:
                            st.session_state['yield_result'] = result
                            st.success("✅ Yield prediction completed!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error during yield prediction: {str(e)}")
    
    with col2:
        st.markdown("### Yield Predictions")
        
        if 'yield_result' in st.session_state:
            result = st.session_state['yield_result']
            
            # Total yield summary
            st.markdown("#### 🎯 Total Estimated Yield")
            total_yield = result['total_estimated_yield']
            st.metric(
                label="Estimated Yield",
                value=f"{total_yield} tons",
                delta=f"±{total_yield * 0.1:.1f} tons (confidence interval)"
            )
            
            # Individual predictions
            st.markdown("#### 📋 Crop-wise Predictions")
            
            predictions_data = []
            for pred in result['individual_predictions']:
                predictions_data.append({
                    "Crop Type": pred['crop_type'].title(),
                    "Area (ha)": f"{pred['area_hectares']:.4f}",
                    "Predicted Yield (tons)": f"{pred['predicted_yield']:.2f}",
                    "Yield/ha (tons)": f"{pred['predicted_yield']/pred['area_hectares']:.2f}" if pred['area_hectares'] > 0 else "0",
                    "Confidence": f"{pred['confidence']:.2%}"
                })
            
            df_predictions = pd.DataFrame(predictions_data)
            st.dataframe(df_predictions, use_container_width=True)
            
            # Yield visualization
            if len(result['individual_predictions']) > 1:
                st.markdown("#### 📊 Yield Analysis Charts")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Yield by crop type
                    crop_names = [p['crop_type'] for p in result['individual_predictions']]
                    yields = [p['predicted_yield'] for p in result['individual_predictions']]
                    
                    fig_yield = px.bar(
                        x=crop_names,
                        y=yields,
                        title="Predicted Yield by Crop Type",
                        labels={'x': 'Crop Type', 'y': 'Yield (tons)'},
                        color=yields,
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_yield, use_container_width=True)
                
                with chart_col2:
                    # Confidence levels
                    confidences = [p['confidence'] for p in result['individual_predictions']]
                    
                    fig_conf = px.scatter(
                        x=yields,
                        y=confidences,
                        size=[p['area_hectares'] for p in result['individual_predictions']],
                        hover_name=crop_names,
                        title="Yield vs Confidence",
                        labels={'x': 'Predicted Yield (tons)', 'y': 'Confidence'}
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
            
            # Factors analysis
            if result['individual_predictions']:
                st.markdown("#### 🔍 Yield Factors Analysis")
                
                sample_factors = result['individual_predictions'][0]['factors']
                factor_df = pd.DataFrame([
                    {"Factor": "Health Factor", "Value": sample_factors.get('health_factor', 0)},
                    {"Factor": "Vegetation Factor", "Value": sample_factors.get('vegetation_factor', 0)},
                    {"Factor": "Disease Penalty", "Value": sample_factors.get('disease_penalty', 1)},
                ])
                
                fig_factors = px.bar(
                    factor_df,
                    x="Factor",
                    y="Value",
                    title="Yield Influencing Factors",
                    color="Value",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig_factors, use_container_width=True)
        else:
            st.info("👆 Upload a field image and click 'Predict Yield' to see predictions here.")

def show_batch_yield_prediction():
    """Batch yield prediction for multiple images"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload Multiple Field Images")
        uploaded_files = st.file_uploader(
            "Choose field images...",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple field images for batch yield prediction",
            key="batch_yield_upload"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} images selected:**")
            for i, file in enumerate(uploaded_files[:3]):  # Show first 3 images
                col_img1, col_img2 = st.columns([1, 3])
                with col_img1:
                    image = Image.open(file)
                    st.image(image, caption=f"{file.name}", use_column_width=True)
                with col_img2:
                    st.write(f"**File:** {file.name}")
                    st.write(f"**Size:** {len(file.getvalue()) / 1024:.1f} KB")
            
            if len(uploaded_files) > 3:
                st.write(f"... and {len(uploaded_files) - 3} more images")
            
            # Batch yield prediction button
            if st.button("🎯 Predict Yield for All Images", type="primary", key="batch_yield_predict"):
                with st.spinner(f"Predicting yield for {len(uploaded_files)} images... This may take a few minutes."):
                    try:
                        # Prepare files for batch API call
                        files = []
                        for file in uploaded_files:
                            file.seek(0)
                            files.append(("files", (file.name, file, file.type or "image/jpeg")))
                        
                        # Call batch yield prediction API
                        import requests
                        response = requests.post(f"{API_BASE_URL}/predict/batch-yield", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['batch_yield_result'] = result
                            st.success(f"✅ Batch yield prediction completed! Processed {result['images_processed']} images.")
                            st.rerun()
                        else:
                            st.error(f"API Error: {response.status_code} - {response.text}")
                            
                    except Exception as e:
                        st.error(f"Error during batch yield prediction: {str(e)}")
    
    with col2:
        st.markdown("### Batch Yield Predictions")
        
        if 'batch_yield_result' in st.session_state:
            result = st.session_state['batch_yield_result']
            
            # Combined yield summary
            st.markdown("#### 🎯 Combined Yield Summary")
            total_yield = result['total_estimated_yield']
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Images Processed", result['images_processed'])
            with metric_col2:
                st.metric("Total Estimated Yield", f"{total_yield} tons")
            with metric_col3:
                st.metric("Average per Image", f"{total_yield/result['images_processed']:.2f} tons" if result['images_processed'] > 0 else "0 tons")
            
            # Combined predictions table
            if result['combined_predictions']:
                st.markdown("#### 📋 All Crop Predictions")
                
                combined_data = []
                for i, pred in enumerate(result['combined_predictions']):
                    combined_data.append({
                        "Segment": i + 1,
                        "Crop Type": pred['crop_type'].title(),
                        "Area (ha)": f"{pred['area_hectares']:.4f}",
                        "Predicted Yield (tons)": f"{pred['predicted_yield']:.2f}",
                        "Yield/ha (tons)": f"{pred['predicted_yield']/pred['area_hectares']:.2f}" if pred['area_hectares'] > 0 else "0",
                        "Confidence": f"{pred['confidence']:.2%}"
                    })
                
                df_combined = pd.DataFrame(combined_data)
                st.dataframe(df_combined, use_container_width=True)
                
                # Combined visualization
                st.markdown("#### 📊 Combined Yield Analysis")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Yield by crop type (aggregated)
                    crop_yields = {}
                    for pred in result['combined_predictions']:
                        crop_type = pred['crop_type']
                        if crop_type not in crop_yields:
                            crop_yields[crop_type] = 0
                        crop_yields[crop_type] += pred['predicted_yield']
                    
                    fig_yield = px.bar(
                        x=list(crop_yields.keys()),
                        y=list(crop_yields.values()),
                        title="Total Yield by Crop Type",
                        labels={'x': 'Crop Type', 'y': 'Total Yield (tons)'},
                        color=list(crop_yields.values()),
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_yield, use_container_width=True)
                
                with chart_col2:
                    # Yield distribution pie chart
                    fig_pie = px.pie(
                        values=list(crop_yields.values()),
                        names=list(crop_yields.keys()),
                        title="Yield Distribution by Crop Type"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Individual image results
            st.markdown("#### 📋 Individual Image Results")
            
            individual_data = []
            for img_result in result['individual_results']:
                individual_data.append({
                    "Image": img_result['filename'],
                    "Crops Analyzed": img_result['crops_analyzed'],
                    "Total Yield (tons)": f"{img_result['total_yield']:.2f}",
                    "Predictions": len(img_result['predictions'])
                })
            
            df_individual = pd.DataFrame(individual_data)
            st.dataframe(df_individual, use_container_width=True)
            
            # Individual image details (expandable)
            st.markdown("#### 📖 Detailed Results by Image")
            for img_result in result['individual_results']:
                with st.expander(f"📁 {img_result['filename']} - {img_result['total_yield']:.2f} tons"):
                    if img_result['predictions']:
                        pred_data = []
                        for pred in img_result['predictions']:
                            pred_data.append({
                                "Crop Type": pred['crop_type'].title(),
                                "Area (ha)": f"{pred['area_hectares']:.4f}",
                                "Predicted Yield (tons)": f"{pred['predicted_yield']:.2f}",
                                "Confidence": f"{pred['confidence']:.2%}"
                            })
                        
                        df_pred = pd.DataFrame(pred_data)
                        st.dataframe(df_pred, use_container_width=True)
                    else:
                        st.write("No crops detected in this image.")
            
        else:
            st.info("👆 Upload multiple field images and click 'Predict Yield for All Images' to see batch predictions here.")

def show_satellite_data():
    """Satellite data processing page"""
    st.markdown("## 🛰️ Satellite Data Processing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Satellite Data Parameters")
        
        # Date selection
        acquisition_date = st.date_input(
            "Acquisition Date",
            value=datetime.now() - timedelta(days=7),
            max_value=datetime.now()
        )
        
        # Satellite type
        satellite_type = st.selectbox(
            "Satellite Type",
            ["Sentinel-2", "Landsat-8", "MODIS", "Sentinel-1"]
        )
        
        # Coordinates
        st.markdown("#### 📍 Coordinates")
        lat = st.number_input("Latitude", value=40.7128, step=0.0001, format="%.4f")
        lon = st.number_input("Longitude", value=-74.0060, step=0.0001, format="%.4f")
        
        coordinates = {"lat": lat, "lon": lon}
        
        if st.button("🔄 Process Satellite Data", type="primary"):
            with st.spinner("Processing satellite imagery..."):
                data = {
                    "acquisition_date": acquisition_date.isoformat(),
                    "satellite_type": satellite_type,
                    "coordinates": coordinates
                }
                
                result = call_api("/satellite/process", method="POST", data=data)
                
                if result:
                    st.session_state['satellite_result'] = result
                    st.success("✅ Satellite data processed!")
                    st.rerun()
    
    with col2:
        st.markdown("### Processing Results")
        
        if 'satellite_result' in st.session_state:
            result = st.session_state['satellite_result']
            
            # Processing status
            st.success(f"✅ Status: {result['processing_status'].title()}")
            
            # Metadata
            st.markdown("#### 📊 Metadata")
            metadata_col1, metadata_col2 = st.columns(2)
            
            with metadata_col1:
                st.metric("Satellite", result['satellite_type'])
                st.metric("Acquisition Date", result['acquisition_date'][:10])
            
            with metadata_col2:
                coords = result['coordinates']
                st.metric("Latitude", f"{coords['lat']:.4f}")
                st.metric("Longitude", f"{coords['lon']:.4f}")
            
            # NDVI Statistics
            st.markdown("#### 🌱 NDVI Statistics")
            ndvi_stats = result['ndvi_stats']
            
            ndvi_col1, ndvi_col2, ndvi_col3, ndvi_col4 = st.columns(4)
            
            with ndvi_col1:
                st.metric("Mean NDVI", f"{ndvi_stats['mean']:.3f}")
            with ndvi_col2:
                st.metric("Std Deviation", f"{ndvi_stats['std']:.3f}")
            with ndvi_col3:
                st.metric("Minimum", f"{ndvi_stats['min']:.3f}")
            with ndvi_col4:
                st.metric("Maximum", f"{ndvi_stats['max']:.3f}")
            
            # NDVI Interpretation
            mean_ndvi = ndvi_stats['mean']
            if mean_ndvi > 0.6:
                ndvi_status = "🌱 Excellent vegetation health"
                status_color = "green"
            elif mean_ndvi > 0.4:
                ndvi_status = "🟡 Good vegetation health"
                status_color = "orange"
            elif mean_ndvi > 0.2:
                ndvi_status = "🟠 Moderate vegetation"
                status_color = "red"
            else:
                ndvi_status = "🔴 Poor vegetation health"
                status_color = "darkred"
            
            st.markdown(f"**Vegetation Assessment:** {ndvi_status}")
            
            # Simulated NDVI visualization
            st.markdown("#### 📈 NDVI Distribution")
            
            # Generate sample NDVI data for visualization
            np.random.seed(42)
            sample_data = np.random.normal(mean_ndvi, ndvi_stats['std'], 1000)
            sample_data = np.clip(sample_data, -1, 1)
            
            fig_ndvi = px.histogram(
                x=sample_data,
                nbins=50,
                title="NDVI Value Distribution",
                labels={'x': 'NDVI Value', 'y': 'Frequency'},
                color_discrete_sequence=['green']
            )
            fig_ndvi.add_vline(x=mean_ndvi, line_dash="dash", line_color="red", 
                              annotation_text=f"Mean: {mean_ndvi:.3f}")
            st.plotly_chart(fig_ndvi, use_container_width=True)
            
        else:
            st.info("👆 Set parameters and click 'Process Satellite Data' to see results here.")
            
            # Show sample satellite imagery
            st.markdown("#### 🖼️ Sample Satellite Imagery")
            st.image("https://images.unsplash.com/photo-1446776877081-d282a0f896e2?w=600&h=400&fit=crop", 
                    caption="Sample Sentinel-2 imagery", use_column_width=True)

def show_analytics():
    """Analytics and trends page"""
    st.markdown("## 📈 Analytics & Trends")
    
    # Get historical data
    history_data = call_api("/history/analyses", params={"limit": 20})
    
    if not history_data:
        st.info("No historical data available for analytics. Perform some analyses first!")
        return
    
    # Process data for analytics
    analytics_data = []
    for item in history_data:
        if item['crop_data'] and item['health_metrics']:
            crop_data = json.loads(item['crop_data'])
            health_metrics = json.loads(item['health_metrics'])
            
            for crop in crop_data:
                analytics_data.append({
                    "date": item['upload_date'][:10],
                    "crop_type": crop['crop_type'],
                    "health_score": crop['health_score'],
                    "area_hectares": crop['area_hectares'],
                    "disease_detected": crop['disease_detected'],
                    "filename": item['filename']
                })
    
    if not analytics_data:
        st.warning("No valid data found for analytics.")
        return
    
    df = pd.DataFrame(analytics_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Overview metrics
    st.markdown("### 📊 Overview Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        total_area = df['area_hectares'].sum()
        st.metric("Total Area Analyzed", f"{total_area:.2f} ha")
    
    with metric_col2:
        avg_health = df['health_score'].mean()
        st.metric("Average Health Score", f"{avg_health:.1f}%")
    
    with metric_col3:
        disease_rate = (df['disease_detected'].sum() / len(df)) * 100
        st.metric("Disease Detection Rate", f"{disease_rate:.1f}%")
    
    with metric_col4:
        unique_crops = df['crop_type'].nunique()
        st.metric("Crop Types Analyzed", unique_crops)
    
    # Time series analysis
    st.markdown("### 📈 Trends Over Time")
    
    # Group by date for time series
    daily_stats = df.groupby('date').agg({
        'health_score': 'mean',
        'disease_detected': 'sum',
        'area_hectares': 'sum'
    }).reset_index()
    
    # Health score trend
    fig_health_trend = px.line(
        daily_stats,
        x='date',
        y='health_score',
        title="Average Health Score Over Time",
        labels={'health_score': 'Health Score (%)', 'date': 'Date'}
    )
    fig_health_trend.add_hline(y=70, line_dash="dash", line_color="orange", 
                              annotation_text="Target Health Score")
    st.plotly_chart(fig_health_trend, use_container_width=True)
    
    # Crop type analysis
    st.markdown("### 🌾 Crop Type Analysis")
    
    crop_col1, crop_col2 = st.columns(2)
    
    with crop_col1:
        # Health by crop type
        crop_health = df.groupby('crop_type')['health_score'].mean().reset_index()
        
        fig_crop_health = px.bar(
            crop_health,
            x='crop_type',
            y='health_score',
            title="Average Health Score by Crop Type",
            color='health_score',
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_crop_health, use_container_width=True)
    
    with crop_col2:
        # Disease rate by crop type
        crop_disease = df.groupby('crop_type').agg({
            'disease_detected': 'sum',
            'crop_type': 'count'
        }).rename(columns={'crop_type': 'total_count'}).reset_index()
        crop_disease['disease_rate'] = (crop_disease['disease_detected'] / crop_disease['total_count']) * 100
        
        fig_crop_disease = px.bar(
            crop_disease,
            x='crop_type',
            y='disease_rate',
            title="Disease Detection Rate by Crop Type (%)",
            color='disease_rate',
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_crop_disease, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    # Health vs Area correlation
    fig_correlation = px.scatter(
        df,
        x='area_hectares',
        y='health_score',
        color='crop_type',
        size='area_hectares',
        title="Health Score vs Area Correlation",
        labels={'area_hectares': 'Area (hectares)', 'health_score': 'Health Score (%)'}
    )
    st.plotly_chart(fig_correlation, use_container_width=True)
    
    # Statistical summary
    st.markdown("### 📋 Statistical Summary")
    summary_stats = df.groupby('crop_type').agg({
        'health_score': ['mean', 'std', 'min', 'max'],
        'area_hectares': ['sum', 'mean'],
        'disease_detected': 'sum'
    }).round(2)
    
    st.dataframe(summary_stats, use_container_width=True)

def show_history():
    """Historical analyses page"""
    st.markdown("## 📋 Analysis History")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        limit = st.selectbox("Show last", [10, 20, 50, 100], index=1)
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    with col3:
        if st.button("📥 Export Data"):
            st.info("Export functionality would be implemented here")
    
    # Get historical data
    history_data = call_api("/history/analyses", params={"limit": limit})
    
    if not history_data:
        st.info("No historical analyses found.")
        return
    
    # Display data
    for i, item in enumerate(history_data):
        with st.expander(f"📁 {item['filename']} - {item['upload_date'][:19]}"):
            
            if item['crop_data'] and item['health_metrics']:
                crop_data = json.loads(item['crop_data'])
                health_metrics = json.loads(item['health_metrics'])
                
                # Summary metrics
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Crops Detected", len(crop_data))
                
                with summary_col2:
                    avg_health = health_metrics.get('average_health_score', 0)
                    st.markdown(f"**Average Health:** {display_health_status(avg_health)}", 
                               unsafe_allow_html=True)
                
                with summary_col3:
                    status = health_metrics.get('overall_status', 'unknown')
                    status_icon = "✅" if status == "healthy" else "⚠️" if status == "attention_needed" else "🔴"
                    st.write(f"**Status:** {status_icon} {status.replace('_', ' ').title()}")
                
                # Detailed crop data
                if crop_data:
                    crop_df = pd.DataFrame([
                        {
                            "Crop Type": crop['crop_type'].title(),
                            "Health Score": f"{crop['health_score']:.1f}%",
                            "Area (ha)": f"{crop['area_hectares']:.4f}",
                            "Disease": "Yes" if crop['disease_detected'] else "No",
                            "Confidence": f"{crop['confidence']:.2f}"
                        }
                        for crop in crop_data
                    ])
                    
                    st.dataframe(crop_df, use_container_width=True)
            else:
                st.warning("No analysis data available for this entry.")

if __name__ == "__main__":
    main()
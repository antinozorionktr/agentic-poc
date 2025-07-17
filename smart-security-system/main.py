import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from main import SecuritySystem
import time
import threading
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Smart Security System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .status-safe {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-alert {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'security_system' not in st.session_state:
    st.session_state.security_system = SecuritySystem()
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False

def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Smart Security System</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Detection settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.4, 0.1)
        
        # Alert settings
        st.subheader("Alert Settings")
        alert_sensitivity = st.selectbox("Alert Sensitivity", ["Low", "Medium", "High"], index=1)
        enable_notifications = st.checkbox("Enable Notifications", True)
        
        # Update system settings
        st.session_state.security_system.update_settings({
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
            'alert_sensitivity': alert_sensitivity,
            'enable_notifications': enable_notifications
        })
        
        st.divider()
        
        # System status
        st.subheader("📊 System Status")
        total_detections = len(st.session_state.detection_history)
        total_alerts = len(st.session_state.alerts)
        
        st.metric("Total Detections", total_detections)
        st.metric("Active Alerts", total_alerts)
        
        if st.button("🗑️ Clear History"):
            st.session_state.alerts = []
            st.session_state.detection_history = []
            st.rerun()

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Monitoring", "📁 File Upload", "📊 Analytics", "⚠️ Alerts"])
    
    with tab1:
        live_monitoring_tab()
    
    with tab2:
        file_upload_tab()
    
    with tab3:
        analytics_tab()
    
    with tab4:
        alerts_tab()

def live_monitoring_tab():
    st.header("Live Camera Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Controls")
        
        camera_source = st.selectbox("Camera Source", ["Webcam (0)", "External Camera (1)", "IP Camera"])
        
        if camera_source == "IP Camera":
            ip_url = st.text_input("IP Camera URL", "rtsp://admin:password@192.168.1.100:554/stream")
            camera_index = ip_url
        else:
            camera_index = 0 if "Webcam" in camera_source else 1
        
        if st.button("🎥 Start Monitoring"):
            st.session_state.is_monitoring = True
            
        if st.button("⏹️ Stop Monitoring"):
            st.session_state.is_monitoring = False
    
    with col1:
        if st.session_state.is_monitoring:
            st.subheader("Live Feed")
            
            # Placeholder for video stream
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                cap = cv2.VideoCapture(camera_index)
                
                if not cap.isOpened():
                    st.error("Could not open camera. Please check your camera connection.")
                    return
                
                frame_count = 0
                while st.session_state.is_monitoring:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame from camera.")
                        break
                    
                    # Process frame through security system
                    processed_frame, detections = st.session_state.security_system.process_frame(frame)
                    
                    # Update detection history
                    if detections:
                        timestamp = datetime.now()
                        for detection in detections:
                            st.session_state.detection_history.append({
                                'timestamp': timestamp,
                                'type': detection['class'],
                                'confidence': detection['confidence'],
                                'bbox': detection['bbox']
                            })
                    
                    # Convert frame for display
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(processed_frame_rgb, caption="Live Security Feed", use_column_width=True)
                    
                    # Update status
                    if detections:
                        status_placeholder.markdown(
                            '<div class="status-box status-alert">⚠️ PERSON DETECTED</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        status_placeholder.markdown(
                            '<div class="status-box status-safe">✅ ALL CLEAR</div>',
                            unsafe_allow_html=True
                        )
                    
                    frame_count += 1
                    time.sleep(0.1)  # Control frame rate
                
                cap.release()
                
            except Exception as e:
                st.error(f"Error during live monitoring: {str(e)}")
        else:
            st.info("Click 'Start Monitoring' to begin live camera feed.")

def file_upload_tab():
    st.header("Upload Image or Video for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="Upload an image or video file for security analysis"
    )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            if file_extension in ['jpg', 'jpeg', 'png']:
                process_image(tmp_file_path, uploaded_file.name)
            elif file_extension in ['mp4', 'avi', 'mov']:
                process_video(tmp_file_path, uploaded_file.name)
        finally:
            os.unlink(tmp_file_path)

def process_image(image_path, filename):
    st.subheader(f"Analysis Results for: {filename}")
    
    # Load and display original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_column_width=True)
    
    with col2:
        st.subheader("Detection Results")
        
        # Process image
        with st.spinner("Analyzing image..."):
            processed_image, detections = st.session_state.security_system.process_frame(image)
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        st.image(processed_image_rgb, use_column_width=True)
        
        # Display detection summary
        if detections:
            st.success(f"Found {len(detections)} person(s)")
            
            for i, detection in enumerate(detections):
                with st.expander(f"Person {i+1}"):
                    st.write(f"**Confidence:** {detection['confidence']:.2%}")
                    st.write(f"**Bounding Box:** {detection['bbox']}")
                    st.write(f"**Classification:** {detection.get('behavior', 'Normal')}")
        else:
            st.info("No persons detected in the image")

def process_video(video_path, filename):
    st.subheader(f"Video Analysis: {filename}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    st.info(f"Video Info: {total_frames} frames, {fps} FPS")
    
    # Video processing controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        process_all = st.button("🎬 Process Entire Video")
    with col2:
        sample_frames = st.button("📸 Sample Frames")
    with col3:
        frame_number = st.number_input("Jump to Frame", 0, total_frames-1, 0)
    
    if process_all:
        process_entire_video(cap, total_frames)
    elif sample_frames:
        process_sample_frames(cap, total_frames)
    
    # Frame-by-frame analysis
    if st.button("Analyze Current Frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            processed_frame, detections = st.session_state.security_system.process_frame(frame)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original Frame")
            with col2:
                st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Processed Frame")
            
            if detections:
                st.success(f"Frame {frame_number}: {len(detections)} person(s) detected")
    
    cap.release()

def process_entire_video(cap, total_frames):
    st.subheader("Processing Entire Video...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    detection_timeline = []
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 10th frame to speed up analysis
        if frame_number % 10 == 0:
            processed_frame, detections = st.session_state.security_system.process_frame(frame)
            
            if detections:
                detection_timeline.append({
                    'frame': frame_number,
                    'timestamp': frame_number / 30,  # Assuming 30 FPS
                    'detections': len(detections)
                })
        
        frame_number += 1
        progress = frame_number / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_number}/{total_frames}")
    
    # Display results
    st.success("Video processing complete!")
    
    if detection_timeline:
        df = pd.DataFrame(detection_timeline)
        st.line_chart(df.set_index('timestamp')['detections'])
        st.dataframe(df)
    else:
        st.info("No persons detected in the video")

def process_sample_frames(cap, total_frames):
    st.subheader("Sample Frame Analysis")
    
    # Sample 5 frames evenly distributed
    sample_frames = [int(i * total_frames / 5) for i in range(5)]
    
    cols = st.columns(5)
    
    for i, frame_num in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            processed_frame, detections = st.session_state.security_system.process_frame(frame)
            
            with cols[i]:
                st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                        caption=f"Frame {frame_num}")
                if detections:
                    st.success(f"{len(detections)} detected")
                else:
                    st.info("No detection")

def analytics_tab():
    st.header("📊 Security Analytics")
    
    if not st.session_state.detection_history:
        st.info("No detection data available. Start monitoring or upload files to see analytics.")
        return
    
    # Create DataFrame from detection history
    df = pd.DataFrame(st.session_state.detection_history)
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Detections", len(df))
    with col2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    with col3:
        unique_days = df['date'].nunique()
        st.metric("Active Days", unique_days)
    with col4:
        peak_hour = df['hour'].mode().iloc[0] if not df.empty else 0
        st.metric("Peak Hour", f"{peak_hour}:00")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detections by Hour")
        hourly_counts = df.groupby('hour').size()
        st.bar_chart(hourly_counts)
    
    with col2:
        st.subheader("Daily Detection Trends")
        daily_counts = df.groupby('date').size()
        st.line_chart(daily_counts)
    
    # Detailed data table
    st.subheader("Recent Detections")
    recent_detections = df.tail(20).copy()
    recent_detections['timestamp'] = recent_detections['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(recent_detections, use_container_width=True)

def alerts_tab():
    st.header("⚠️ Security Alerts")
    
    # Alert summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Alerts", len(st.session_state.alerts))
    with col2:
        active_alerts = sum(1 for alert in st.session_state.alerts if alert.get('status') == 'active')
        st.metric("Active Alerts", active_alerts)
    with col3:
        if st.session_state.alerts:
            latest_alert = max(st.session_state.alerts, key=lambda x: x['timestamp'])
            time_diff = datetime.now() - latest_alert['timestamp']
            st.metric("Last Alert", f"{time_diff.seconds // 60}m ago")
        else:
            st.metric("Last Alert", "Never")
    
    # Alert configuration
    with st.expander("🔧 Alert Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            alert_types = st.multiselect(
                "Alert Types",
                ["Person Detection", "Loitering", "Unusual Movement", "After Hours"],
                default=["Person Detection"]
            )
        
        with col2:
            notification_methods = st.multiselect(
                "Notification Methods",
                ["Email", "SMS", "Push Notification", "Sound Alert"],
                default=["Sound Alert"]
            )
    
    # Recent alerts
    if st.session_state.alerts:
        st.subheader("Recent Alerts")
        
        for alert in reversed(st.session_state.alerts[-10:]):  # Show last 10 alerts
            timestamp = alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            alert_type = alert.get('type', 'Unknown')
            severity = alert.get('severity', 'Medium')
            
            # Color code by severity
            if severity == 'High':
                alert_color = '🔴'
            elif severity == 'Medium':
                alert_color = '🟡'
            else:
                alert_color = '🟢'
            
            with st.expander(f"{alert_color} {alert_type} - {timestamp}"):
                st.write(f"**Severity:** {severity}")
                st.write(f"**Message:** {alert.get('message', 'No details available')}")
                st.write(f"**Location:** {alert.get('location', 'Unknown')}")
                
                if alert.get('status') == 'active':
                    if st.button(f"Mark as Resolved", key=f"resolve_{alert['timestamp']}"):
                        alert['status'] = 'resolved'
                        st.rerun()
    else:
        st.info("No alerts generated yet. The system will display alerts when suspicious activity is detected.")

if __name__ == "__main__":
    main()
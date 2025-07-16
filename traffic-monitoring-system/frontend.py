# simple_frontend.py - Simplified Frontend that works without threading issues

import streamlit as st
import cv2
import numpy as np
import websocket
import json
import base64
import time
import requests
from datetime import datetime

st.set_page_config(page_title="Traffic Monitor", layout="wide")

# Initialize session state
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'total_frames' not in st.session_state:
    st.session_state.total_frames = 0
if 'vehicle_stats' not in st.session_state:
    st.session_state.vehicle_stats = {}

st.title("🚦 Traffic Monitoring System")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'])
    
    # Backend status
    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        if response.status_code == 200:
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Error")
    except:
        st.error("❌ Backend Offline")
        st.info("Start backend with: python backend.py")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Video Feed")
    video_placeholder = st.empty()
    progress_bar = st.progress(0)

with col2:
    st.subheader("Statistics")
    metric1 = st.empty()
    metric2 = st.empty()
    metric3 = st.empty()
    metric4 = st.empty()
    metric5 = st.empty()  # For license plates

# Process button
if st.button("🎬 Process Video", type="primary"):
    if uploaded_file is not None:
        # Save video temporarily
        video_path = "temp_upload.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if cap.isOpened():
            # Get video info
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.session_state.total_frames = total_frames
            
            # Connect to WebSocket
            try:
                ws = websocket.create_connection("ws://localhost:8000/ws")
                
                # Process frames
                frame_count = 0
                
                # Create a container for the stop button
                stop_container = st.empty()
                stop_button = stop_container.button("⏹️ Stop Processing")
                
                while True:
                    if stop_button:
                        break
                        
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Skip frames for performance (process every 3rd frame)
                    if frame_count % 3 != 0:
                        continue
                    
                    # Resize frame
                    height, width = frame.shape[:2]
                    if width > 960:
                        scale = 960 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Encode and send
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    encoded = base64.b64encode(buffer).decode('utf-8')
                    
                    ws.send(json.dumps({
                        'type': 'frame',
                        'frame': encoded,
                        'total_frames': total_frames
                    }))
                    
                    # Receive response
                    try:
                        response = json.loads(ws.recv())
                        
                        if response['type'] == 'processed':
                            # Decode processed frame
                            processed_data = base64.b64decode(response['frame'])
                            processed_array = np.frombuffer(processed_data, np.uint8)
                            processed_frame = cv2.imdecode(processed_array, cv2.IMREAD_COLOR)
                            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            
                            # Display - Fixed deprecation warning
                            video_placeholder.image(processed_frame_rgb, use_container_width=True)
                            
                            # Update stats
                            stats = response['stats']
                            st.session_state.vehicle_stats = stats
                            
                            # Update progress
                            progress = frame_count / total_frames
                            progress_bar.progress(progress, text=f"Frame {frame_count}/{total_frames}")
                            
                            # Update metrics
                            metric1.metric("Current Vehicles", stats.get('current_vehicles', 0))
                            metric2.metric("Total Detected", sum(stats.get('vehicle_count', {}).values()))
                            metric3.metric("Avg Speed", f"{stats.get('average_speed', 0):.1f} km/h")
                            metric4.metric("License Plates", stats.get('plates_count', 0))
                            metric5.metric("Progress", f"{stats.get('progress_percentage', 0):.1f}%")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        break
                    
                    # Small delay
                    time.sleep(0.03)
                
                # Cleanup
                ws.close()
                cap.release()
                stop_container.empty()
                
                st.success("✅ Processing complete!")
                
                # Show final stats
                if st.session_state.vehicle_stats:
                    st.subheader("Final Statistics")
                    stats = st.session_state.vehicle_stats
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Vehicles", sum(stats.get('vehicle_count', {}).values()))
                    with col2:
                        st.metric("Average Speed", f"{stats.get('average_speed', 0):.1f} km/h")
                    with col3:
                        st.metric("License Plates", stats.get('plates_count', 0))
                    with col4:
                        st.metric("Frames Processed", frame_count)
                    
                    # Vehicle breakdown
                    if stats.get('vehicle_count'):
                        st.subheader("Vehicle Breakdown")
                        for vtype, count in stats['vehicle_count'].items():
                            st.write(f"- **{vtype}**: {count}")
                    
                    # License plates detected
                    if stats.get('detected_plates'):
                        st.subheader("Detected License Plates")
                        plates_df = []
                        for track_id, plate_info in stats['detected_plates'].items():
                            plates_df.append({
                                'Vehicle ID': track_id,
                                'License Plate': plate_info['text'],
                                'Vehicle Type': plate_info['vehicle_class'],
                                'Confidence': f"{plate_info['confidence']:.2f}"
                            })
                        
                        if plates_df:
                            import pandas as pd
                            df = pd.DataFrame(plates_df)
                            st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"WebSocket error: {str(e)}")
                st.info("Make sure the backend is running on http://localhost:8000")
        else:
            st.error("Failed to open video file")
    else:
        st.warning("Please upload a video file first")

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    1. **Start the backend**: Run `python backend.py` in terminal
    2. **Install dependencies**: `pip install easyocr` for license plate reading
    3. **Upload a video**: Use the file uploader in the sidebar
    4. **Click Process Video**: The system will analyze vehicles and license plates
    5. **View results**: See live detection, statistics, and detected license plates
    
    **Features**:
    - Vehicle detection and tracking
    - License plate detection and OCR
    - Speed estimation
    - Real-time statistics
    
    **Note**: The system processes every 3rd frame for better performance.
    License plate detection requires good image quality and clear plates.
    """)
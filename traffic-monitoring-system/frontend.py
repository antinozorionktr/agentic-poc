import streamlit as st
import cv2
import numpy as np
import websocket
import json
import base64
import time
import requests
from datetime import datetime
import pandas as pd
import altair as alt

st.set_page_config(page_title="Traffic Monitor", layout="wide", page_icon="🚦")

# Initialize session state
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'total_frames' not in st.session_state:
    st.session_state.total_frames = 0
if 'vehicle_stats' not in st.session_state:
    st.session_state.vehicle_stats = {}
if 'plate_history' not in st.session_state:
    st.session_state.plate_history = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

st.title("🚦 Traffic Monitoring System")
st.markdown("Real-time vehicle detection, tracking, and license plate recognition")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Monitoring", "🚗 License Plates", "📊 Analytics", "⚙️ Settings"])

# Tab 1: Live Monitoring
with tab1:
    # Sidebar for this tab
    with st.sidebar:
        st.header("Video Configuration")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov', 'mkv'])
        
        # Processing options
        st.subheader("Processing Options")
        skip_frames = st.slider("Process every N frames", 1, 5, 2, 
                              help="Lower values = better accuracy but slower processing")
        
        # Backend status
        st.subheader("System Status")
        try:
            response = requests.get("http://localhost:8000/", timeout=2)
            if response.status_code == 200:
                st.success("✅ Backend Connected")
                backend_status = True
            else:
                st.error("❌ Backend Error")
                backend_status = False
        except:
            st.error("❌ Backend Offline")
            st.info("Start backend with: `python backend.py`")
            backend_status = False
    
    # Main monitoring area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video Feed")
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        progress_text = st.empty()
    
    with col2:
        st.subheader("Real-Time Statistics")
        
        # Create metric placeholders
        metric_container = st.container()
        with metric_container:
            m1, m2 = st.columns(2)
            with m1:
                metric1 = st.empty()
                metric3 = st.empty()
                metric5 = st.empty()
            with m2:
                metric2 = st.empty()
                metric4 = st.empty()
                metric6 = st.empty()
        
        # Vehicle breakdown
        st.subheader("Vehicle Types")
        vehicle_chart_placeholder = st.empty()
        
        # Speed distribution
        st.subheader("Speed Distribution")
        speed_chart_placeholder = st.empty()
    
    # Process button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        process_btn = st.button("🎬 Start Processing", type="primary", disabled=not backend_status)
    with col_btn2:
        reset_btn = st.button("🔄 Reset Stats", disabled=not backend_status)
    
    if reset_btn and backend_status:
        try:
            response = requests.post("http://localhost:8000/reset")
            if response.status_code == 200:
                st.success("Statistics reset successfully!")
                st.session_state.vehicle_stats = {}
                st.session_state.plate_history = []
                st.session_state.processing_complete = False
        except:
            st.error("Failed to reset statistics")
    
    # Process video
    if process_btn and uploaded_file is not None and backend_status:
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
                
                # Create stop button container
                stop_container = st.empty()
                stop_button = stop_container.button("⏹️ Stop Processing")
                
                while True:
                    if stop_button:
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Skip frames based on user setting
                    if frame_count % skip_frames != 0:
                        continue
                    
                    # Resize frame for better performance
                    height, width = frame.shape[:2]
                    max_width = 1280
                    if width > max_width:
                        scale = max_width / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Encode and send
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
                            
                            # Display
                            video_placeholder.image(processed_frame_rgb, use_container_width=True)
                            
                            # Update stats
                            stats = response['stats']
                            st.session_state.vehicle_stats = stats
                            
                            # Store plate history
                            if stats.get('detected_plates'):
                                st.session_state.plate_history = list(stats['detected_plates'].values())
                            
                            # Update progress
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            progress_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
                            
                            # Update metrics
                            metric1.metric("🚗 Current Vehicles", stats.get('current_vehicles', 0))
                            metric2.metric("📊 Total Detected", sum(stats.get('vehicle_count', {}).values()))
                            metric3.metric("🏃 Avg Speed", f"{stats.get('average_speed', 0):.1f} km/h")
                            metric4.metric("🎯 License Plates", stats.get('plates_count', 0))
                            metric5.metric("📹 Frames", f"{stats.get('frame_count', 0)}")
                            metric6.metric("⏱️ Progress", f"{stats.get('progress_percentage', 0):.1f}%")
                            
                            # Update vehicle breakdown chart
                            if stats.get('vehicle_count'):
                                vehicle_df = pd.DataFrame(
                                    list(stats['vehicle_count'].items()),
                                    columns=['Vehicle Type', 'Count']
                                )
                                vehicle_chart = alt.Chart(vehicle_df).mark_bar().encode(
                                    x='Count:Q',
                                    y=alt.Y('Vehicle Type:N', sort='-x'),
                                    color=alt.Color('Vehicle Type:N', legend=None)
                                ).properties(height=150)
                                vehicle_chart_placeholder.altair_chart(vehicle_chart, use_container_width=True)
                            
                            # Update speed distribution
                            if stats.get('plate_average_speeds'):
                                speeds = [info['average'] for info in stats['plate_average_speeds'].values() if info['average'] > 0]
                                if speeds:
                                    speed_df = pd.DataFrame({'Speed (km/h)': speeds})
                                    speed_hist = alt.Chart(speed_df).mark_bar().encode(
                                        x=alt.X('Speed (km/h):Q', bin=alt.Bin(maxbins=20)),
                                        y='count()'
                                    ).properties(height=150)
                                    speed_chart_placeholder.altair_chart(speed_hist, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        break
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)
                
                # Cleanup
                ws.close()
                cap.release()
                stop_container.empty()
                
                st.session_state.processing_complete = True
                st.success("✅ Processing complete!")
                
                # Show final summary
                if st.session_state.vehicle_stats:
                    st.subheader("📊 Processing Summary")
                    stats = st.session_state.vehicle_stats
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Vehicles", sum(stats.get('vehicle_count', {}).values()))
                    with col2:
                        st.metric("Unique Plates", stats.get('plates_count', 0))
                    with col3:
                        st.metric("Avg Speed", f"{stats.get('average_speed', 0):.1f} km/h")
                    with col4:
                        st.metric("Frames Processed", frame_count)
                
            except Exception as e:
                st.error(f"WebSocket connection error: {str(e)}")
                st.info("Please ensure the backend is running on http://localhost:8000")
        else:
            st.error("Failed to open video file")
    elif process_btn and not uploaded_file:
        st.warning("Please upload a video file first")

# Tab 2: License Plates
with tab2:
    st.header("🚗 Detected License Plates")
    
    if st.session_state.plate_history:
        # Create DataFrame from plate history
        plate_df = pd.DataFrame(st.session_state.plate_history)
        
        # Add some calculated columns
        if 'average_speed' in plate_df.columns:
            plate_df['Speed Category'] = pd.cut(
                plate_df['average_speed'], 
                bins=[0, 30, 60, 90, float('inf')],
                labels=['Slow', 'Normal', 'Fast', 'Very Fast']
            )
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Plates Detected", len(plate_df))
        with col2:
            if 'average_speed' in plate_df.columns:
                avg_speed = plate_df['average_speed'].mean()
                st.metric("Average Speed", f"{avg_speed:.1f} km/h")
        with col3:
            if 'confidence' in plate_df.columns:
                avg_conf = plate_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
        with col4:
            if 'detection_count' in plate_df.columns:
                total_detections = plate_df['detection_count'].sum()
                st.metric("Total Detections", total_detections)
        
        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'vehicle_class' in plate_df.columns:
                vehicle_types = ['All'] + list(plate_df['vehicle_class'].unique())
                selected_vehicle = st.selectbox("Vehicle Type", vehicle_types)
        
        with col2:
            if 'average_speed' in plate_df.columns:
                speed_range = st.slider(
                    "Speed Range (km/h)",
                    float(plate_df['average_speed'].min()),
                    float(plate_df['average_speed'].max()),
                    (float(plate_df['average_speed'].min()), float(plate_df['average_speed'].max()))
                )
        
        with col3:
            if 'confidence' in plate_df.columns:
                min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5)
        
        # Apply filters
        filtered_df = plate_df.copy()
        
        if 'vehicle_class' in plate_df.columns and selected_vehicle != 'All':
            filtered_df = filtered_df[filtered_df['vehicle_class'] == selected_vehicle]
        
        if 'average_speed' in plate_df.columns:
            filtered_df = filtered_df[
                (filtered_df['average_speed'] >= speed_range[0]) & 
                (filtered_df['average_speed'] <= speed_range[1])
            ]
        
        if 'confidence' in plate_df.columns:
            filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
        
        # Display options
        st.subheader("License Plate Details")
        
        # Sort options
        sort_columns = ['text', 'average_speed', 'confidence', 'detection_count']
        available_sort_columns = [col for col in sort_columns if col in filtered_df.columns]
        
        if available_sort_columns:
            col1, col2 = st.columns([3, 1])
            with col1:
                sort_by = st.selectbox("Sort by", available_sort_columns)
            with col2:
                sort_order = st.radio("Order", ["Descending", "Ascending"])
            
            ascending = sort_order == "Ascending"
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
        
        # Display the data
        if not filtered_df.empty:
            # Format the DataFrame for display
            display_df = filtered_df.copy()
            
            # Rename columns for better display
            column_mapping = {
                'text': 'License Plate',
                'vehicle_class': 'Vehicle Type',
                'average_speed': 'Avg Speed (km/h)',
                'max_speed': 'Max Speed (km/h)',
                'current_speed': 'Last Speed (km/h)',
                'confidence': 'Confidence',
                'detection_count': 'Times Detected'
            }
            
            display_df = display_df.rename(columns=column_mapping)
            
            # Format numeric columns
            if 'Avg Speed (km/h)' in display_df.columns:
                display_df['Avg Speed (km/h)'] = display_df['Avg Speed (km/h)'].round(1)
            if 'Max Speed (km/h)' in display_df.columns:
                display_df['Max Speed (km/h)'] = display_df['Max Speed (km/h)'].round(1)
            if 'Last Speed (km/h)' in display_df.columns:
                display_df['Last Speed (km/h)'] = display_df['Last Speed (km/h)'].round(1)
            if 'Confidence' in display_df.columns:
                display_df['Confidence'] = (display_df['Confidence'] * 100).round(1).astype(str) + '%'
            
            # Select columns to display
            display_columns = ['License Plate', 'Vehicle Type', 'Avg Speed (km/h)', 
                             'Max Speed (km/h)', 'Confidence', 'Times Detected']
            display_columns = [col for col in display_columns if col in display_df.columns]
            
            # Display the table
            st.dataframe(
                display_df[display_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "License Plate": st.column_config.TextColumn("License Plate", width="medium"),
                    "Vehicle Type": st.column_config.TextColumn("Vehicle Type", width="small"),
                    "Avg Speed (km/h)": st.column_config.NumberColumn("Avg Speed", format="%.1f km/h"),
                    "Max Speed (km/h)": st.column_config.NumberColumn("Max Speed", format="%.1f km/h"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Times Detected": st.column_config.NumberColumn("Detections", width="small")
                }
            )
            
            # Export options
            st.subheader("Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"license_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create a summary report
                if st.button("📄 Generate Report"):
                    report = f"""
# License Plate Detection Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total unique plates detected: {len(plate_df)}
- Filtered plates shown: {len(filtered_df)}
- Average speed: {filtered_df['average_speed'].mean():.1f} km/h
- Average confidence: {filtered_df['confidence'].mean():.2%}

## Top Speeders
{filtered_df.nlargest(5, 'average_speed')[['text', 'average_speed', 'vehicle_class']].to_string()}

## Most Frequently Detected
{filtered_df.nlargest(5, 'detection_count')[['text', 'detection_count', 'vehicle_class']].to_string()}
"""
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("No license plates match the selected filters.")
    else:
        st.info("No license plates detected yet. Process a video first to see results.")
        
        # Show instructions
        with st.expander("How to detect license plates"):
            st.markdown("""
            1. Go to the **Live Monitoring** tab
            2. Upload a video file
            3. Click **Start Processing**
            4. The system will automatically detect and read license plates
            5. Return to this tab to see all detected plates with their average speeds
            
            **Tips for better detection:**
            - Use high-quality video with clear license plates
            - Ensure good lighting conditions
            - License plates should be clearly visible and not too far from the camera
            - The system works best with stationary or slow-moving vehicles
            """)

# Tab 3: Analytics
with tab3:
    st.header("📊 Traffic Analytics")
    
    if st.session_state.vehicle_stats and st.session_state.plate_history:
        stats = st.session_state.vehicle_stats
        
        # Overall metrics
        st.subheader("Overall Traffic Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_vehicles = sum(stats.get('vehicle_count', {}).values())
            st.metric("Total Vehicles", total_vehicles)
        
        with col2:
            unique_plates = stats.get('plates_count', 0)
            st.metric("Unique License Plates", unique_plates)
        
        with col3:
            avg_speed = stats.get('average_speed', 0)
            st.metric("Average Speed", f"{avg_speed:.1f} km/h")
        
        with col4:
            if total_vehicles > 0:
                detection_rate = (unique_plates / total_vehicles) * 100
                st.metric("Plate Detection Rate", f"{detection_rate:.1f}%")
        
        # Vehicle type distribution
        st.subheader("Vehicle Type Distribution")
        if stats.get('vehicle_count'):
            vehicle_df = pd.DataFrame(
                list(stats['vehicle_count'].items()),
                columns=['Vehicle Type', 'Count']
            )
            
            # Pie chart
            pie_chart = alt.Chart(vehicle_df).mark_arc().encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Vehicle Type", type="nominal"),
                tooltip=['Vehicle Type', 'Count']
            ).properties(height=300)
            
            st.altair_chart(pie_chart, use_container_width=True)
        
        # Speed analysis
        st.subheader("Speed Analysis")
        
        plate_df = pd.DataFrame(st.session_state.plate_history)
        if 'average_speed' in plate_df.columns and not plate_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Speed distribution histogram
                speed_hist = alt.Chart(plate_df).mark_bar().encode(
                    x=alt.X('average_speed:Q', bin=alt.Bin(maxbins=30), title='Speed (km/h)'),
                    y=alt.Y('count()', title='Number of Vehicles'),
                    tooltip=['count()']
                ).properties(
                    title='Speed Distribution',
                    height=300
                )
                st.altair_chart(speed_hist, use_container_width=True)
            
            with col2:
                # Speed by vehicle type
                if 'vehicle_class' in plate_df.columns:
                    speed_by_type = plate_df.groupby('vehicle_class')['average_speed'].agg(['mean', 'max', 'count']).reset_index()
                    speed_by_type.columns = ['Vehicle Type', 'Avg Speed', 'Max Speed', 'Count']
                    
                    speed_chart = alt.Chart(speed_by_type).mark_bar().encode(
                        x=alt.X('Avg Speed:Q', title='Average Speed (km/h)'),
                        y=alt.Y('Vehicle Type:N', sort='-x'),
                        color=alt.Color('Vehicle Type:N', legend=None),
                        tooltip=['Vehicle Type', 'Avg Speed', 'Max Speed', 'Count']
                    ).properties(
                        title='Average Speed by Vehicle Type',
                        height=300
                    )
                    st.altair_chart(speed_chart, use_container_width=True)
        
        # Time-based analysis (if we had timestamps)
        st.subheader("Detection Statistics")
        
        if 'confidence' in plate_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence distribution
                conf_hist = alt.Chart(plate_df).mark_bar().encode(
                    x=alt.X('confidence:Q', bin=alt.Bin(maxbins=20), title='Confidence Score'),
                    y=alt.Y('count()', title='Number of Detections'),
                    tooltip=['count()']
                ).properties(
                    title='License Plate Detection Confidence',
                    height=250
                )
                st.altair_chart(conf_hist, use_container_width=True)
            
            with col2:
                # Detection count distribution
                if 'detection_count' in plate_df.columns:
                    detection_dist = alt.Chart(plate_df).mark_bar().encode(
                        x=alt.X('detection_count:Q', title='Times Detected'),
                        y=alt.Y('count()', title='Number of Vehicles'),
                        tooltip=['count()']
                    ).properties(
                        title='Detection Frequency',
                        height=250
                    )
                    st.altair_chart(detection_dist, use_container_width=True)
        
        # Top violators (high speed)
        st.subheader("Speed Violations")
        
        if 'average_speed' in plate_df.columns:
            speed_limit = st.slider("Set Speed Limit (km/h)", 30, 120, 60)
            
            violators = plate_df[plate_df['average_speed'] > speed_limit].sort_values('average_speed', ascending=False)
            
            if not violators.empty:
                st.warning(f"⚠️ {len(violators)} vehicles exceeded the speed limit of {speed_limit} km/h")
                
                # Show top violators
                display_cols = ['text', 'vehicle_class', 'average_speed', 'max_speed', 'confidence']
                display_cols = [col for col in display_cols if col in violators.columns]
                
                st.dataframe(
                    violators[display_cols].head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "text": "License Plate",
                        "vehicle_class": "Vehicle Type",
                        "average_speed": st.column_config.NumberColumn("Avg Speed (km/h)", format="%.1f"),
                        "max_speed": st.column_config.NumberColumn("Max Speed (km/h)", format="%.1f"),
                        "confidence": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1)
                    }
                )
            else:
                st.success(f"✅ No vehicles exceeded the speed limit of {speed_limit} km/h")
        
    else:
        st.info("No data available for analytics. Process a video first to see analytics.")

# Tab 4: Settings
with tab4:
    st.header("⚙️ System Settings")
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Vehicle Detection Model**")
        vehicle_model = st.selectbox(
            "Select model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
            help="Larger models are more accurate but slower"
        )
        
        st.markdown("**Processing Settings**")
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            0.0, 1.0, 0.5,
            help="Higher values reduce false positives"
        )
    
    with col2:
        st.markdown("**License Plate Model**")
        st.info("Using: license_plate_detector.pt")
        
        st.markdown("**Speed Calculation**")
        pixel_to_meter = st.number_input(
            "Pixel to Meter Ratio",
            0.01, 0.5, 0.05,
            help="Calibrate based on your camera setup"
        )
    
    st.subheader("Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Default Export Format",
            ["CSV", "JSON", "Excel"]
        )
    
    with col2:
        include_timestamps = st.checkbox("Include Timestamps", value=True)
        include_images = st.checkbox("Include Vehicle Images", value=False)
    
    st.subheader("About")
    
    with st.expander("System Information"):
        st.markdown("""
        ### Traffic Monitoring System v2.0
        
        **Features:**
        - Real-time vehicle detection and tracking using YOLOv8
        - License plate detection and OCR using EasyOCR
        - Speed estimation using object tracking
        - Comprehensive analytics and reporting
        
        **Improvements in v2.0:**
        - Enhanced SORT tracker with better ID management
        - Improved license plate detection with multiple preprocessing techniques
        - Duplicate vehicle counting prevention
        - Better speed calculation with smoothing
        - Advanced filtering and analytics
        
        **Requirements:**
        - Python 3.8+
        - YOLOv8 (ultralytics)
        - EasyOCR
        - OpenCV
        - FastAPI
        - Streamlit
        
        **Tips for Best Results:**
        1. Use high-quality video (minimum 720p)
        2. Ensure good lighting conditions
        3. Camera should have a clear view of license plates
        4. Optimal camera angle: 20-45 degrees
        5. Process every 2-3 frames for balance between speed and accuracy
        
        **Known Limitations:**
        - License plate detection works best for standard rectangular plates
        - Speed estimation accuracy depends on camera calibration
        - OCR accuracy varies with image quality and plate condition
        """)
    
    with st.expander("Troubleshooting"):
        st.markdown("""
        **Backend not connecting?**
        1. Ensure backend.py is running: `python backend.py`
        2. Check that port 8000 is not in use
        3. Verify all dependencies are installed
        
        **Poor license plate detection?**
        1. Check video quality (minimum 720p recommended)
        2. Ensure plates are clearly visible
        3. Adjust confidence thresholds
        4. Try processing fewer frames (every 1-2 frames)
        
        **Duplicate vehicle counting?**
        - This version includes duplicate prevention
        - Vehicles are counted only once when first detected
        - ID tracking ensures consistent counting
        
        **Speed calculations seem incorrect?**
        - Calibrate the pixel-to-meter ratio for your camera
        - Ensure consistent frame rate
        - Speed is averaged over multiple frames for accuracy
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Traffic Monitoring System v2.0 | Enhanced with better tracking and license plate detection</p>
    </div>
    """,
    unsafe_allow_html=True
)
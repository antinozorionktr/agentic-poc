import cv2
import numpy as np
import requests
import json
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import tempfile
import os
import base64
import streamlit as st

# Try to import WebRTC components (optional)
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    import threading
    import queue
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning("⚠️ WebRTC not available. Install with: `pip install streamlit-webrtc aiortc`")

# Configure Streamlit page
st.set_page_config(
    page_title="AI Fitness Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .exercise-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 0.5rem 0;
    }
    .form-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .good-form { color: #28a745; }
    .average-form { color: #ffc107; }
    .poor-form { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

# RTC Configuration for WebRTC
if WEBRTC_AVAILABLE:
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.result_queue = queue.Queue(maxsize=10)
            self.latest_result = None
            self.frame_count = 0
            
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Process every 5th frame to reduce load
            self.frame_count += 1
            if self.frame_count % 5 == 0:
                self.process_frame(img)
            
            # Draw pose overlay if we have results
            if self.latest_result:
                img = self.draw_pose_overlay(img, self.latest_result)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        def process_frame(self, frame):
            """Process frame and send to backend"""
            try:
                # Encode frame as base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frame_data = f"data:image/jpeg;base64,{frame_base64}"
                
                # Send to API (simplified for demo)
                payload = {
                    "frame_data": frame_data,
                    "timestamp": time.time()
                }
                
                # This would ideally use WebSocket, but for simplicity using HTTP
                try:
                    response = requests.post(f"{API_BASE_URL}/analyze_video_frame", 
                                           json=payload, timeout=0.5)
                    if response.status_code == 200:
                        result = response.json()
                        if not self.result_queue.full():
                            self.result_queue.put(result)
                        self.latest_result = result
                except:
                    pass  # Skip if API is slow
                    
            except Exception as e:
                pass  # Skip frame processing errors
        
        def draw_pose_overlay(self, frame, result):
            """Draw pose analysis overlay on frame"""
            if not result.get('success'):
                return frame
            
            h, w = frame.shape[:2]
            
            # Draw exercise info
            exercise = result.get('exercise_detected', 'none')
            reps = result.get('rep_count', 0)
            form_score = result.get('form_score', 0)
            
            # Create overlay
            overlay = frame.copy()
            
            # Draw info box
            cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add text
            cv2.putText(frame, f"Exercise: {exercise.title()}", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Reps: {reps}", (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Form Score: {form_score:.0f}%", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw keypoints if available
            keypoints = result.get('keypoints', [])
            if keypoints:
                for point in keypoints:
                    if len(point) >= 4 and point[3] > 0.5:  # Check visibility
                        x, y = int(point[0] * w), int(point[1] * h)
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            
            return frame
        
        def get_latest_result(self):
            """Get latest analysis result"""
            try:
                return self.result_queue.get_nowait()
            except queue.Empty:
                return self.latest_result

class FitnessApp:
    def __init__(self):
        self.session_state_init()
    
    def session_state_init(self):
        """Initialize session state variables"""
        if 'workout_started' not in st.session_state:
            st.session_state.workout_started = False
        if 'total_reps' not in st.session_state:
            st.session_state.total_reps = {'pushup': 0, 'squat': 0, 'pullup': 0, 'plank': 0, 'yoga_poses': 0}
        if 'workout_history' not in st.session_state:
            st.session_state.workout_history = []
        if 'current_exercise' not in st.session_state:
            st.session_state.current_exercise = "none"
        if 'form_scores' not in st.session_state:
            st.session_state.form_scores = []
        if 'yoga_poses_detected' not in st.session_state:
            st.session_state.yoga_poses_detected = []
        if 'current_yoga_pose' not in st.session_state:
            st.session_state.current_yoga_pose = None
    
    def call_api(self, endpoint, method="GET", data=None, files=None):
        """Make API calls to backend"""
        try:
            url = f"{API_BASE_URL}/{endpoint}"
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                if files:
                    # For file uploads, don't set Content-Type header
                    response = requests.post(url, files=files)
                else:
                    response = requests.post(url, json=data)
            
            if response.status_code == 200:
                return response.json(), True
            else:
                return {"error": response.text}, False
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to backend. Make sure the API server is running on port 8000."}, False
        except Exception as e:
            return {"error": str(e)}, False
    
    def analyze_pose_from_camera(self):
        """Analyze pose from camera feed"""
        camera_input = st.camera_input("Take a picture for pose analysis")
        
        if camera_input is not None:
            # Display the captured image
            image = Image.open(camera_input)
            st.image(image, caption="Captured Image", width=300)
            
            # Convert to bytes for API
            camera_input.seek(0)  # Reset file pointer
            image_bytes = camera_input.read()
            
            # Send to API
            files = {'file': ('camera_image.jpg', image_bytes, 'image/jpeg')}
            result, success = self.call_api("analyze_pose", "POST", files=files)
            
            if success and result.get('success'):
                return result
            else:
                st.error(f"Pose analysis failed: {result.get('error', 'Unknown error')}")
                return None
        return None
    
    def analyze_uploaded_video(self):
        """Analyze pose from uploaded video file"""
        st.subheader("🎬 Video File Analysis")
        
        uploaded_video = st.file_uploader(
            "Upload a workout video",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload a video file showing your workout for complete analysis"
        )
        
        if uploaded_video is not None:
            # Display video info
            file_details = {
                "filename": uploaded_video.name,
                "filetype": uploaded_video.type,
                "filesize": uploaded_video.size
            }
            st.write("📁 **File Details:**")
            st.json(file_details)
            
            # Video preview
            st.video(uploaded_video)
            
            # Analysis options
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_type = st.selectbox(
                    "Analysis Type",
                    ["Full Video", "Video Segment"],
                    help="Choose to analyze the entire video or a specific segment"
                )
            
            with col2:
                if analysis_type == "Video Segment":
                    start_time = st.number_input("Start Time (seconds)", min_value=0.0, value=0.0, step=0.5)
                    end_time = st.number_input("End Time (seconds)", min_value=0.0, value=30.0, step=0.5)
                else:
                    start_time = 0.0
                    end_time = None
            
            # Analysis button
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Analyzing video... This may take a few minutes."):
                    # Reset file pointer
                    uploaded_video.seek(0)
                    
                    if analysis_type == "Full Video":
                        # Full video analysis
                        files = {'file': (uploaded_video.name, uploaded_video.read(), uploaded_video.type)}
                        result, success = self.call_api("analyze_video", "POST", files=files)
                    else:
                        # Segment analysis
                        files = {'file': (uploaded_video.name, uploaded_video.read(), uploaded_video.type)}
                        # Note: For segment analysis, we'd need to modify the API call to include start/end times
                        # For now, using full video analysis
                        result, success = self.call_api("analyze_video", "POST", files=files)
                    
                    if success and result.get('success'):
                        self.display_video_results(result)
                    else:
                        st.error(f"Video analysis failed: {result.get('error', 'Unknown error')}")
        
        return None
    
    def display_video_results(self, result):
        """Display comprehensive video analysis results"""
        st.success("✅ Video analysis completed!")
        
        # Video Information
        video_info = result.get('video_info', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{video_info.get('duration', 0):.1f}s")
        with col2:
            st.metric("FPS", f"{video_info.get('fps', 0):.1f}")
        with col3:
            st.metric("Total Frames", video_info.get('total_frames', 0))
        with col4:
            st.metric("Processed", video_info.get('processed_frames', 0))
        
        st.markdown("---")
        
        # Overall Statistics
        st.subheader("📊 Workout Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Total repetitions
            total_reps = result.get('total_reps', {})
            st.write("**Total Repetitions:**")
            for exercise, count in total_reps.items():
                if count > 0:
                    st.metric(exercise.title(), count)
        
        with col2:
            # Average form score
            avg_form = result.get('avg_form_score', 0)
            form_color = "🟢" if avg_form >= 80 else "🟡" if avg_form >= 60 else "🔴"
            st.metric("Average Form Score", f"{form_color} {avg_form:.1f}%")
            
            # Exercise summary
            exercise_summary = result.get('exercise_summary', {})
            if exercise_summary:
                st.write("**Exercise Breakdown:**")
                for exercise, data in exercise_summary.items():
                    if exercise != 'unknown' and data['frames'] > 0:
                        st.write(f"• {exercise.title()}: {data['duration']:.1f}s (Form: {data['avg_form_score']:.1f}%)")
        
        # Detailed Charts
        st.markdown("---")
        st.subheader("📈 Detailed Analysis")
        
        # Frame-by-frame results
        frame_results = result.get('frame_results', [])
        if frame_results:
            # Create DataFrame for plotting
            df = pd.DataFrame(frame_results)
            
            # Exercise detection over time
            fig_exercise = px.scatter(
                df, 
                x='timestamp', 
                y='exercise_detected',
                color='form_score',
                title="Exercise Detection Over Time",
                labels={'timestamp': 'Time (seconds)', 'exercise_detected': 'Exercise Type'},
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_exercise, use_container_width=True)
            
            # Form score over time
            df_with_scores = df[df['form_score'] > 0]
            if not df_with_scores.empty:
                fig_form = px.line(
                    df_with_scores,
                    x='timestamp',
                    y='form_score',
                    title="Form Score Over Time",
                    labels={'timestamp': 'Time (seconds)', 'form_score': 'Form Score (%)'}
                )
                fig_form.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Good Form")
                fig_form.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Average Form")
                st.plotly_chart(fig_form, use_container_width=True)
            
            # Joint angles over time (if available)
            if 'angles' in df.columns:
                angles_data = []
                for _, row in df.iterrows():
                    if isinstance(row['angles'], dict):
                        for angle_type, angle_value in row['angles'].items():
                            if angle_value > 0:
                                angles_data.append({
                                    'timestamp': row['timestamp'],
                                    'angle_type': angle_type.replace('_', ' ').title(),
                                    'angle_value': angle_value
                                })
                
                if angles_data:
                    angles_df = pd.DataFrame(angles_data)
                    fig_angles = px.line(
                        angles_df,
                        x='timestamp',
                        y='angle_value',
                        color='angle_type',
                        title="Joint Angles Over Time",
                        labels={'timestamp': 'Time (seconds)', 'angle_value': 'Angle (degrees)'}
                    )
                    st.plotly_chart(fig_angles, use_container_width=True)
        
        # Performance insights
        st.markdown("---")
        st.subheader("💡 Performance Insights")
        
        insights = []
        
        # Form analysis
        if avg_form >= 80:
            insights.append("🟢 **Excellent form!** You maintained good technique throughout the workout.")
        elif avg_form >= 60:
            insights.append("🟡 **Good form overall** with room for improvement in some areas.")
        else:
            insights.append("🔴 **Focus on form.** Consider slowing down and ensuring proper technique.")
        
        # Rep analysis
        total_exercise_reps = sum([count for exercise, count in total_reps.items() if exercise != 'unknown'])
        if total_exercise_reps > 50:
            insights.append("💪 **Great workout volume!** You completed a substantial number of repetitions.")
        elif total_exercise_reps > 20:
            insights.append("👍 **Good workout intensity** with moderate repetition count.")
        
        # Exercise variety
        exercises_performed = len([ex for ex, count in total_reps.items() if count > 0 and ex != 'unknown'])
        if exercises_performed > 1:
            insights.append("🎯 **Good exercise variety** - you performed multiple exercise types.")
        
        for insight in insights:
            st.markdown(insight)
        
        # Download results option
        st.markdown("---")
        if st.button("📥 Download Analysis Report"):
            # Create downloadable report
            report_data = {
                "video_info": video_info,
                "total_reps": total_reps,
                "avg_form_score": avg_form,
                "exercise_summary": exercise_summary,
                "frame_results": frame_results,
                "insights": insights,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            report_json = json.dumps(report_data, indent=2)
            st.download_button(
                label="Download JSON Report",
                data=report_json,
                file_name=f"workout_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def display_pose_results(self, result):
        """Display pose analysis results"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            exercise = result['exercise_detected']
            if exercise == 'yoga' and result.get('yoga_pose'):
                display_text = f"Yoga: {result['yoga_pose'].title()}"
            else:
                display_text = exercise.title()
                
            st.markdown(f"""
            <div class="metric-card">
                <h3>Exercise Detected</h3>
                <h2>{display_text}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Rep Count</h3>
                <h2>{result['rep_count']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            form_score = result['form_score']
            form_class = "good-form" if form_score >= 80 else "average-form" if form_score >= 60 else "poor-form"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Form Score</h3>
                <h2 class="{form_class}">{form_score:.0f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_reps = sum(result['total_reps'].values())
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Activity</h3>
                <h2>{total_reps}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Update session state
        st.session_state.total_reps = result['total_reps']
        st.session_state.current_exercise = result['exercise_detected']
        if result['form_score'] > 0:
            st.session_state.form_scores.append(result['form_score'])
        
        # Yoga-specific information
        if result.get('yoga_pose') and result.get('yoga_confidence', 0) > 0:
            st.subheader("🧘 Yoga Pose Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Pose", result['yoga_pose'].replace('_', ' ').title())
                st.metric("Confidence", f"{result['yoga_confidence']:.2f}")
            
            with col2:
                # Show recent yoga poses
                yoga_history = result.get('yoga_pose_history', [])
                if yoga_history:
                    st.write("**Recent Poses:**")
                    for pose_data in yoga_history[-3:]:
                        pose_name = pose_data['pose'].replace('_', ' ').title()
                        st.write(f"• {pose_name} ({pose_data['confidence']:.2f})")
        
        # Display detailed angles for traditional exercises
        if result.get('angles') and result['exercise_detected'] in ['pushup', 'squat']:
            st.subheader("📐 Joint Angles")
            angle_col1, angle_col2 = st.columns(2)
            
            with angle_col1:
                if result['angles']['pushup_angle'] > 0:
                    st.metric("Elbow Angle (Pushup)", f"{result['angles']['pushup_angle']:.1f}°")
            
            with angle_col2:
                if result['angles']['squat_angle'] > 0:
                    st.metric("Knee Angle (Squat)", f"{result['angles']['squat_angle']:.1f}°")
    
    def live_video_analysis(self):
        """Live video analysis with WebRTC"""
        if not WEBRTC_AVAILABLE:
            st.error("WebRTC is not available. Please install: `pip install streamlit-webrtc aiortc`")
            st.info("Try 'Webcam Analysis' mode instead for live video analysis.")
            return
            
        st.subheader("🎥 Live Video Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("📹 **Live Camera Feed**")
            
            # WebRTC streamer
            ctx = webrtc_streamer(
                key="fitness-tracker",
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if ctx.video_processor:
                # Display live results
                placeholder = st.empty()
                
                while ctx.state.playing:
                    result = ctx.video_processor.get_latest_result()
                    if result:
                        with placeholder.container():
                            self.display_live_results(result)
                    time.sleep(0.1)
        
        with col2:
            st.write("⚡ **Live Stats**")
            
            # Real-time controls
            if st.button("🔄 Reset Counters", key="live_reset"):
                # Reset through API
                self.call_api("reset_counters", "POST")
                st.success("Counters reset!")
            
            # Live workout timer
            if 'workout_start_time' not in st.session_state:
                st.session_state.workout_start_time = time.time()
            
            elapsed_time = time.time() - st.session_state.workout_start_time
            st.metric("Workout Time", f"{elapsed_time/60:.1f} min")
            
            # Quick exercise selector
            st.subheader("🎯 Focus Exercise")
            focus_exercise = st.selectbox(
                "Target exercise",
                ["Auto-detect", "Pushup", "Squat", "Pullup"],
                key="live_focus"
            )
            
            # Live feedback settings
            st.subheader("⚙️ Settings")
            
            feedback_enabled = st.checkbox("Audio Feedback", value=True)
            show_keypoints = st.checkbox("Show Keypoints", value=True)
            sensitivity = st.slider("Detection Sensitivity", 0.3, 0.9, 0.5)
    
    def display_live_results(self, result):
        """Display live analysis results"""
        if not result or not result.get('success'):
            st.warning("⏳ Waiting for pose detection...")
            return
        
        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            exercise = result.get('exercise_detected', 'none')
            if exercise == 'yoga' and result.get('yoga_pose'):
                display_text = f"🧘 {result['yoga_pose'].replace('_', ' ').title()}"
            else:
                display_text = exercise.title()
            st.metric("Exercise", display_text)
        
        with col2:
            reps = result.get('rep_count', 0)
            if exercise == 'yoga':
                st.metric("Unique Poses", reps)
            else:
                st.metric("Current Reps", reps)
        
        with col3:
            form_score = result.get('form_score', 0)
            color = "🟢" if form_score >= 80 else "🟡" if form_score >= 60 else "🔴"
            st.metric("Form Score", f"{color} {form_score:.0f}%")
        
        with col4:
            total_reps = sum(result.get('total_reps', {}).values())
            st.metric("Total Activity", total_reps)
        
        # Yoga-specific live feedback
        if exercise == 'yoga' and result.get('yoga_pose'):
            yoga_confidence = result.get('yoga_confidence', 0)
            st.progress(yoga_confidence)
            st.caption(f"Pose Confidence: {yoga_confidence:.2f}")
            
            # Show recent pose transitions
            yoga_history = result.get('yoga_pose_history', [])
            if len(yoga_history) > 1:
                st.write("**Recent Transitions:**")
                for pose_data in yoga_history[-3:]:
                    pose_name = pose_data['pose'].replace('_', ' ').title()
                    st.caption(f"→ {pose_name}")
        
        # Live progress bar for traditional exercises
        elif exercise != 'none' and exercise != 'unknown' and exercise != 'yoga':
            angles = result.get('angles', {})
            if exercise == 'pushup' and 'pushup_angle' in angles:
                angle = angles['pushup_angle']
                progress = max(0, min(100, (180 - angle) / 90 * 100))
                st.progress(progress / 100)
                st.caption(f"Pushup Progress: {angle:.1f}° elbow angle")
            elif exercise == 'squat' and 'squat_angle' in angles:
                angle = angles['squat_angle']
                progress = max(0, min(100, (180 - angle) / 60 * 100))
                st.progress(progress / 100)
                st.caption(f"Squat Progress: {angle:.1f}° knee angle")
    
    def webcam_analysis(self):
        """Simple webcam analysis using OpenCV"""
        st.subheader("📷 Webcam Analysis")
        
        # Controls
        col1, col2 = st.columns(2)
        with col1:
            run_webcam = st.checkbox("Start Webcam Analysis")
        with col2:
            analysis_interval = st.slider("Analysis Interval (frames)", 5, 30, 10)
        
        if run_webcam:
            # Create placeholders
            frame_placeholder = st.empty()
            results_placeholder = st.empty()
            
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera permissions.")
                return
            
            frame_count = 0
            
            try:
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam")
                        break
                    
                    # Process every nth frame
                    frame_count += 1
                    if frame_count % analysis_interval == 0:
                        # Encode frame
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        frame_data = f"data:image/jpeg;base64,{frame_base64}"
                        
                        # Analyze frame
                        payload = {
                            "frame_data": frame_data,
                            "timestamp": time.time()
                        }
                        
                        result, success = self.call_api("analyze_video_frame", "POST", data=payload)
                        
                        if success and result.get('success'):
                            # Display results
                            with results_placeholder.container():
                                self.display_live_results(result)
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Small delay
                    time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                st.error(f"Webcam error: {str(e)}")
            finally:
                cap.release()
    
    def display_workout_stats(self):
        """Display current workout statistics"""
        stats, success = self.call_api("stats")
        
        if success:
            st.subheader("📊 Workout Statistics")
            
            # Create rep count chart
            rep_data = stats['rep_counters']
            exercises = list(rep_data.keys())
            counts = list(rep_data.values())
            
            fig = px.bar(
                x=exercises,
                y=counts,
                labels={'x': 'Exercise Type', 'y': 'Repetitions'},
                title="Repetitions by Exercise Type",
                color=counts,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Form score trend
            if st.session_state.form_scores:
                st.subheader("📈 Form Score Trend")
                form_fig = px.line(
                    x=range(len(st.session_state.form_scores)),
                    y=st.session_state.form_scores,
                    labels={'x': 'Analysis Count', 'y': 'Form Score (%)'},
                    title="Form Score Over Time"
                )
                form_fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Good Form")
                form_fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Average Form")
                st.plotly_chart(form_fig, use_container_width=True)
    
    def display_exercise_guide(self):
        """Display exercise guides and tips"""
        st.subheader("🏋️ Exercise Guides")
        
        # Tabs for different exercise types
        tab1, tab2, tab3 = st.tabs(["💪 Strength Training", "🧘 Yoga Poses", "📋 All Supported"])
        
        with tab1:
            exercise_guides = {
                "Pushup": {
                    "description": "A classic upper body exercise targeting chest, shoulders, and triceps.",
                    "tips": [
                        "Keep your body in a straight line from head to heels",
                        "Lower your chest to just above the ground",
                        "Keep your core engaged throughout the movement",
                        "Control both the down and up phases"
                    ],
                    "common_mistakes": [
                        "Sagging hips or raising hips too high",
                        "Not going through full range of motion",
                        "Flaring elbows too wide"
                    ]
                },
                "Squat": {
                    "description": "A fundamental lower body exercise targeting quads, glutes, and hamstrings.",
                    "tips": [
                        "Keep your feet shoulder-width apart",
                        "Lower until thighs are parallel to ground",
                        "Keep your chest up and core engaged",
                        "Drive through your heels to stand up"
                    ],
                    "common_mistakes": [
                        "Knees caving inward",
                        "Not going deep enough",
                        "Leaning too far forward"
                    ]
                }
            }
            
            for exercise, guide in exercise_guides.items():
                with st.expander(f"{exercise} Guide"):
                    st.write(f"**Description:** {guide['description']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Tips for Good Form:**")
                        for tip in guide['tips']:
                            st.write(f"• {tip}")
                    
                    with col2:
                        st.write("**Common Mistakes to Avoid:**")
                        for mistake in guide['common_mistakes']:
                            st.write(f"• {mistake}")
        
        with tab2:
            st.write("🧘 **Yoga Pose Detection**")
            st.info("Our AI can detect 107 different yoga poses! Here are some popular ones:")
            
            # Display some popular yoga poses
            popular_yoga_poses = [
                "downward facing dog (adho mukha svanasana)",
                "warrior i (virabhadrasana i)",
                "warrior ii (virabhadrasana ii)", 
                "tree pose (vriksasana)",
                "child's pose (balasana)",
                "mountain pose (tadasana)",
                "cobra pose (bhujangasana)",
                "triangle pose (utthita trikonasana)",
                "plank pose (phalakasana)",
                "lotus pose (padmasana)"
            ]
            
            col1, col2 = st.columns(2)
            for i, pose in enumerate(popular_yoga_poses):
                if i % 2 == 0:
                    col1.write(f"• {pose.title()}")
                else:
                    col2.write(f"• {pose.title()}")
            
            # Yoga tips
            st.write("**Tips for Yoga Practice:**")
            yoga_tips = [
                "Focus on your breathing throughout each pose",
                "Hold poses for 5-8 breaths for best results",
                "Listen to your body and don't force positions",
                "Use props (blocks, straps) when needed",
                "Practice regularly, even if just for 10 minutes"
            ]
            
            for tip in yoga_tips:
                st.write(f"• {tip}")
        
        with tab3:
            # Get yoga poses from API
            yoga_data, success = self.call_api("yoga_poses")
            if success:
                st.write(f"**Total Supported Exercises:**")
                st.metric("Yoga Poses", yoga_data.get('total_poses', 107))
                st.metric("Strength Exercises", "4 (Pushup, Squat, Pullup, Plank)")
                
                if st.checkbox("Show all yoga poses"):
                    poses = yoga_data.get('poses', [])
                    # Display in 3 columns
                    cols = st.columns(3)
                    for i, pose in enumerate(poses):
                        col_idx = i % 3
                        cols[col_idx].write(f"• {pose.replace('_', ' ').title()}")
            else:
                st.error("Could not load yoga poses list")
    
    def workout_controls(self):
        """Workout control buttons"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reset Counters", type="secondary"):
                result, success = self.call_api("reset_counters", "POST")
                if success:
                    st.success("Counters reset successfully!")
                    st.session_state.total_reps = {'pushup': 0, 'squat': 0, 'pullup': 0, 'plank': 0, 'yoga_poses': 0}
                    st.session_state.form_scores = []
                    st.session_state.yoga_poses_detected = []
                    st.rerun()
                else:
                    st.error("Failed to reset counters")
        
        with col2:
            if st.button("📊 Refresh Stats", type="secondary"):
                st.rerun()
        
        with col3:
            if st.button("💾 Save Session", type="primary"):
                # Save current session data
                session_data = {
                    "timestamp": datetime.now().isoformat(),
                    "total_reps": st.session_state.total_reps,
                    "avg_form_score": np.mean(st.session_state.form_scores) if st.session_state.form_scores else 0
                }
                st.session_state.workout_history.append(session_data)
                st.success("Session saved!")
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">💪 AI Fitness Tracker</h1>', unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("🎯 Workout Options")
            
            # API Health Check
            health, success = self.call_api("health")
            if success:
                st.success("✅ Backend Connected")
                if not health.get('mediapipe_loaded', False):
                    st.warning("⚠️ MediaPipe not loaded")
                
                # Show active sessions
                active_sessions = health.get('active_sessions', 0)
                if active_sessions > 0:
                    st.info(f"🔴 {active_sessions} live session(s)")
            else:
                st.error("❌ Backend Disconnected")
                st.error("Make sure to run: `python main.py`")
            
            st.markdown("---")
            
def analyze_uploaded_image(self):
    """Analyze pose from uploaded image"""
    uploaded_file = st.file_uploader(
        "Upload an image for pose analysis",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image showing your full body in exercise position"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        # Reset file pointer to beginning
        uploaded_file.seek(0)

        # Send to API with proper file handling
        files = {'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
        result, success = self.call_api("analyze_pose", "POST", files=files)

        if success and result.get('success'):
            return result
        else:
            st.error(f"Pose analysis failed: {result.get('error', 'Unknown error')}")
            return None

    # The rest of the UI should still be shown, even if no file is uploaded
    st.markdown("---")

    # Exercise selection
    st.subheader("Target Exercise")
    target_exercise = st.selectbox(
        "Select target exercise",
        ["Auto-detect", "Pushup", "Squat", "Pullup", "Plank", "Yoga Poses"]
    )

    # Workout intensity
    st.subheader("Intensity Level")
    intensity = st.slider("Set intensity level", 1, 10, 5)

    st.markdown("---")

    # Quick stats
    if st.session_state.get("total_reps"):
        st.subheader("Quick Stats")
        for exercise, count in st.session_state.total_reps.items():
            if count > 0:
                st.metric(exercise.title(), count)

    # Analysis options
    analysis_modes = [
        "Live Video (WebRTC)",
        "Webcam Analysis",
        "Upload Video",
        "Camera Snapshot",
        "Upload Image",
        "View Stats Only"
    ]
    if not WEBRTC_AVAILABLE:
        analysis_modes[0] = "Live Video (WebRTC) - Not Available"

    analysis_mode = st.selectbox("Choose Analysis Mode", analysis_modes)

    # Workout controls
    st.markdown("---")
    self.workout_controls()

    # Exercise guides
    st.markdown("---")
    self.display_exercise_guide()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>🤖 Powered by YOLOv8 + MediaPipe + FastAPI + Streamlit</p>
            <p>Keep pushing your limits! 💪</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = FitnessApp()
    app.run()
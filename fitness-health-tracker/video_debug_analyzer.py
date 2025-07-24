# video_debug_analyzer.py
import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import requests
import time
from PIL import Image
import io

class VideoDebugAnalyzer:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.mp_drawing = None
        self.mp_pose = None
        try:
            import mediapipe as mp
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_pose = mp.solutions.pose
        except:
            st.warning("MediaPipe not available for visualization")
    
    def draw_pose_landmarks(self, image, keypoints):
        """Draw pose landmarks on image"""
        if not self.mp_drawing or not self.mp_pose:
            return image
        
        h, w = image.shape[:2]
        
        # Create landmark list for MediaPipe
        landmarks = []
        for kp in keypoints:
            if len(kp) >= 4:
                landmark = type('obj', (object,), {
                    'x': kp[0],
                    'y': kp[1],
                    'z': kp[2],
                    'visibility': kp[3]
                })
                landmarks.append(landmark)
        
        # Create pose landmarks object
        pose_landmarks = type('obj', (object,), {'landmark': landmarks})
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            image,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        return image
    
    def draw_exercise_info(self, image, analysis_result):
        """Draw exercise information overlay on image"""
        h, w = image.shape[:2]
        
        # Create semi-transparent overlay
        overlay = image.copy()
        
        # Draw background rectangle
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw text information
        exercise = analysis_result.get('exercise_detected', 'none')
        reps = analysis_result.get('rep_count', 0)
        form_score = analysis_result.get('form_score', 0)
        
        # Exercise type
        cv2.putText(image, f"Exercise: {exercise.upper()}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Rep count
        cv2.putText(image, f"Reps: {reps}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Form score with color coding
        form_color = (0, 255, 0) if form_score >= 80 else (0, 255, 255) if form_score >= 60 else (0, 0, 255)
        cv2.putText(image, f"Form Score: {form_score:.1f}%", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, form_color, 2)
        
        # Yoga pose info if detected
        yoga_pose = analysis_result.get('yoga_pose')
        if yoga_pose:
            yoga_confidence = analysis_result.get('yoga_confidence', 0)
            cv2.putText(image, f"Yoga: {yoga_pose.replace('_', ' ').title()}", (20, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(image, f"Confidence: {yoga_confidence:.2f}", (300, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw angles if available
        angles = analysis_result.get('angles', {})
        if exercise == 'pushup' and angles.get('pushup_angle', 0) > 0:
            angle_text = f"Elbow Angle: {angles['pushup_angle']:.1f}°"
            cv2.putText(image, angle_text, (w - 300, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif exercise == 'squat' and angles.get('squat_angle', 0) > 0:
            angle_text = f"Knee Angle: {angles['squat_angle']:.1f}°"
            cv2.putText(image, angle_text, (w - 300, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw frame number and timestamp
        frame_num = analysis_result.get('frame_number', 0)
        timestamp = analysis_result.get('timestamp', 0)
        cv2.putText(image, f"Frame: {frame_num} | Time: {timestamp:.2f}s", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image
    
    def draw_angle_visualization(self, image, landmarks, exercise_type):
        """Draw angle visualization for exercises"""
        h, w = image.shape[:2]
        
        try:
            if exercise_type == 'pushup' and landmarks:
                # Get pushup keypoints
                left_shoulder = landmarks[11]
                left_elbow = landmarks[13]
                left_wrist = landmarks[15]
                
                # Convert normalized coordinates to pixel coordinates
                pts = []
                for point in [left_shoulder, left_elbow, left_wrist]:
                    x = int(point.x * w)
                    y = int(point.y * h)
                    pts.append((x, y))
                
                # Draw lines
                cv2.line(image, pts[0], pts[1], (0, 255, 0), 3)
                cv2.line(image, pts[1], pts[2], (0, 255, 0), 3)
                
                # Draw points
                for pt in pts:
                    cv2.circle(image, pt, 8, (0, 0, 255), -1)
                
                # Calculate and display angle
                angle = self.calculate_angle(
                    [left_shoulder.x, left_shoulder.y],
                    [left_elbow.x, left_elbow.y],
                    [left_wrist.x, left_wrist.y]
                )
                
                # Draw angle arc
                cv2.ellipse(image, pts[1], (50, 50), 0, 0, int(angle), (255, 255, 0), 2)
                cv2.putText(image, f"{angle:.1f}°", (pts[1][0] + 60, pts[1][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
            elif exercise_type == 'squat' and landmarks:
                # Get squat keypoints
                left_hip = landmarks[23]
                left_knee = landmarks[25]
                left_ankle = landmarks[27]
                
                # Convert to pixel coordinates
                pts = []
                for point in [left_hip, left_knee, left_ankle]:
                    x = int(point.x * w)
                    y = int(point.y * h)
                    pts.append((x, y))
                
                # Draw lines
                cv2.line(image, pts[0], pts[1], (0, 255, 0), 3)
                cv2.line(image, pts[1], pts[2], (0, 255, 0), 3)
                
                # Draw points
                for pt in pts:
                    cv2.circle(image, pt, 8, (0, 0, 255), -1)
                
                # Calculate angle
                angle = self.calculate_angle(
                    [left_hip.x, left_hip.y],
                    [left_knee.x, left_knee.y],
                    [left_ankle.x, left_ankle.y]
                )
                
                # Draw angle arc
                cv2.ellipse(image, pts[1], (50, 50), 0, 0, int(angle), (255, 255, 0), 2)
                cv2.putText(image, f"{angle:.1f}°", (pts[1][0] + 60, pts[1][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        except:
            pass  # Skip if landmarks are not available
        
        return image
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def process_video_with_visualization(self, video_path, progress_placeholder, frame_placeholder, stats_placeholder):
        """Process video and show live visualization"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Could not open video file")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer for output
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Initialize tracking variables
        frame_results = []
        frame_count = 0
        exercise_counts = {}
        form_scores = []
        last_update_time = time.time()
        
        # Frame skip for performance (process every nth frame)
        frame_skip = max(1, int(fps / 5))  # Process ~5 frames per second
        
        st.info(f"📹 Processing {total_frames} frames at {fps:.1f} FPS (analyzing every {frame_skip} frames)")
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = (frame_count + 1) / total_frames
            progress_placeholder.progress(progress, text=f"Processing frame {frame_count + 1}/{total_frames}")
            
            # Only process every nth frame for performance
            if frame_count % frame_skip == 0:
                # Encode frame for API
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frame_data = f"data:image/jpeg;base64,{frame_base64}"
                
                # Analyze frame
                try:
                    response = requests.post(
                        f"{self.api_base_url}/analyze_video_frame",
                        json={"frame_data": frame_data, "timestamp": frame_count / fps},
                        timeout=1.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get('success'):
                            # Store results
                            result['frame_number'] = frame_count
                            frame_results.append(result)
                            
                            # Update counters
                            exercise = result.get('exercise_detected', 'none')
                            if exercise not in exercise_counts:
                                exercise_counts[exercise] = 0
                            exercise_counts[exercise] += 1
                            
                            if result.get('form_score', 0) > 0:
                                form_scores.append(result['form_score'])
                            
                            # Draw visualizations on frame
                            annotated_frame = frame.copy()
                            
                            # Draw pose landmarks if available
                            if result.get('keypoints'):
                                # Draw skeleton
                                h, w = annotated_frame.shape[:2]
                                keypoints = result['keypoints']
                                
                                # Draw connections (simplified skeleton)
                                connections = [
                                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                                    (11, 23), (12, 24), (23, 24),  # Torso
                                    (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
                                ]
                                
                                for connection in connections:
                                    if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                                        pt1 = keypoints[connection[0]]
                                        pt2 = keypoints[connection[1]]
                                        if pt1[3] > 0.5 and pt2[3] > 0.5:  # Check visibility
                                            x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
                                            x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
                                            cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Draw keypoints
                                for i, kp in enumerate(keypoints):
                                    if len(kp) >= 4 and kp[3] > 0.5:
                                        x, y = int(kp[0] * w), int(kp[1] * h)
                                        cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)
                            
                            # Draw exercise info overlay
                            annotated_frame = self.draw_exercise_info(annotated_frame, result)
                            
                            # Write annotated frame to output video
                            out.write(annotated_frame)
                            
                            # Display frame (rate-limited for smooth display)
                            current_time = time.time()
                            if current_time - last_update_time > 0.1:  # Update display every 100ms
                                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                
                                # Resize for display if too large
                                max_display_width = 800
                                if frame_width > max_display_width:
                                    scale = max_display_width / frame_width
                                    new_width = int(frame_width * scale)
                                    new_height = int(frame_height * scale)
                                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                                
                                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                                last_update_time = current_time
                                
                                # Update live stats
                                with stats_placeholder.container():
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Current Exercise", exercise.title())
                                    with col2:
                                        reps = result.get('total_reps', {})
                                        total_reps = sum(reps.values())
                                        st.metric("Total Reps", total_reps)
                                    with col3:
                                        current_form = result.get('form_score', 0)
                                        color_emoji = "🟢" if current_form >= 80 else "🟡" if current_form >= 60 else "🔴"
                                        st.metric("Form Score", f"{color_emoji} {current_form:.1f}%")
                                    with col4:
                                        st.metric("Progress", f"{progress*100:.1f}%")
                                    
                                    # Show current angles if available
                                    angles = result.get('angles', {})
                                    if angles:
                                        st.write("**Joint Angles:**")
                                        angle_text = []
                                        if angles.get('pushup_angle', 0) > 0:
                                            angle_text.append(f"Elbow: {angles['pushup_angle']:.1f}°")
                                        if angles.get('squat_angle', 0) > 0:
                                            angle_text.append(f"Knee: {angles['squat_angle']:.1f}°")
                                        if angle_text:
                                            st.write(" | ".join(angle_text))
                                    
                                    # Show yoga pose if detected
                                    if result.get('yoga_pose'):
                                        st.write(f"**Yoga Pose:** {result['yoga_pose'].replace('_', ' ').title()}")
                                        st.write(f"**Confidence:** {result.get('yoga_confidence', 0):.2f}")
                        else:
                            # No detection - write original frame
                            out.write(frame)
                            
                except requests.exceptions.Timeout:
                    # Skip frame if API is slow
                    out.write(frame)
                    st.warning(f"⚠️ Skipped frame {frame_count} (API timeout)")
                except Exception as e:
                    # On error, write original frame
                    out.write(frame)
                    if frame_count % 100 == 0:  # Only show error occasionally
                        st.error(f"Error at frame {frame_count}: {str(e)}")
            else:
                # For skipped frames, just write original
                out.write(frame)
                
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        # Final update
        progress_placeholder.progress(1.0, text="Processing complete!")
        
        # Calculate final statistics
        avg_form_score = np.mean(form_scores) if form_scores else 0
        
        # Show final frame
        if frame_results:
            st.success(f"✅ Processed {len(frame_results)} frames with exercise detection")
        
        return {
            'output_video_path': output_path,
            'frame_results': frame_results,
            'exercise_counts': exercise_counts,
            'avg_form_score': avg_form_score,
            'total_frames': total_frames,
            'fps': fps,
            'frames_analyzed': len(frame_results)
        }

def analyze_video_with_debug(api_base_url="http://localhost:8000"):
    """Enhanced video analysis with visual debugging"""
    # File uploader (removed subheader as it's already in the main app)
    uploaded_video = st.file_uploader(
        "Upload a workout video for detailed analysis",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="Upload a video to see frame-by-frame pose detection and analysis"
    )
    
    if uploaded_video is not None:
        # Save uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            temp_video_path = tmp_file.name
        
        # Display original video
        st.video(uploaded_video)
        
        # Analysis options
        col1, col2 = st.columns(2)
        with col1:
            show_live = st.checkbox("Show Live Processing", value=True)
        with col2:
            save_output = st.checkbox("Save Annotated Video", value=True)
        
        if st.button("🔍 Start Video Analysis", type="primary"):
            # Create placeholders for live updates
            progress_placeholder = st.empty()
            
            if show_live:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("📹 **Live Processing View**")
                    frame_placeholder = st.empty()
                with col2:
                    st.write("📊 **Live Statistics**")
                    stats_placeholder = st.empty()
            else:
                frame_placeholder = None
                stats_placeholder = st.empty()
            
            # Initialize analyzer
            analyzer = VideoDebugAnalyzer(api_base_url)
            
            # Process video
            with st.spinner("Processing video... This may take a few minutes."):
                results = analyzer.process_video_with_visualization(
                    temp_video_path,
                    progress_placeholder,
                    frame_placeholder if show_live else st.empty(),
                    stats_placeholder
                )
            
            # Clear progress bar
            progress_placeholder.empty()
            
            if results:
                st.success("✅ Video analysis complete!")
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Overall statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", results['total_frames'])
                with col2:
                    st.metric("FPS", f"{results['fps']:.1f}")
                with col3:
                    st.metric("Avg Form Score", f"{results['avg_form_score']:.1f}%")
                with col4:
                    st.metric("Frames Analyzed", len(results['frame_results']))
                
                # Exercise breakdown
                st.subheader("🏋️ Exercise Breakdown")
                exercise_df = pd.DataFrame(
                    [(k, v) for k, v in results['exercise_counts'].items()],
                    columns=['Exercise', 'Frame Count']
                )
                exercise_df = exercise_df[exercise_df['Exercise'] != 'none']
                
                if not exercise_df.empty:
                    fig = px.pie(exercise_df, values='Frame Count', names='Exercise',
                                title="Exercise Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Frame-by-frame analysis
                if results['frame_results']:
                    st.subheader("📈 Detailed Frame Analysis")
                    
                    # Create DataFrame
                    df = pd.DataFrame(results['frame_results'])
                    
                    # Exercise timeline
                    fig = px.scatter(df, x='timestamp', y='exercise_detected',
                                   color='form_score',
                                   title="Exercise Detection Timeline",
                                   color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Form score over time
                    df_with_scores = df[df['form_score'] > 0]
                    if not df_with_scores.empty:
                        fig = px.line(df_with_scores, x='timestamp', y='form_score',
                                    title="Form Score Over Time")
                        fig.add_hline(y=80, line_dash="dash", line_color="green",
                                    annotation_text="Good Form")
                        fig.add_hline(y=60, line_dash="dash", line_color="orange",
                                    annotation_text="Average Form")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download annotated video
                if save_output and os.path.exists(results['output_video_path']):
                    st.subheader("📥 Download Annotated Video")
                    with open(results['output_video_path'], 'rb') as f:
                        video_bytes = f.read()
                    st.download_button(
                        label="Download Video with Annotations",
                        data=video_bytes,
                        file_name=f"annotated_{uploaded_video.name}",
                        mime="video/mp4"
                    )
                
                # Frame-by-frame review
                if st.checkbox("Show Frame-by-Frame Review"):
                    st.subheader("🔍 Frame-by-Frame Review")
                    
                    frame_num = st.slider("Select Frame", 0, len(results['frame_results'])-1, 0)
                    frame_data = results['frame_results'][frame_num]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            'frame': frame_data['frame_number'],
                            'timestamp': f"{frame_data['timestamp']:.2f}s",
                            'exercise': frame_data['exercise_detected'],
                            'form_score': frame_data['form_score'],
                            'rep_count': frame_data.get('rep_count', 0)
                        })
                    
                    with col2:
                        if frame_data.get('angles'):
                            st.write("**Joint Angles:**")
                            for angle_name, angle_value in frame_data['angles'].items():
                                if angle_value > 0:
                                    st.write(f"• {angle_name}: {angle_value:.1f}°")
                        
                        if frame_data.get('yoga_pose'):
                            st.write(f"**Yoga Pose:** {frame_data['yoga_pose']}")
                            st.write(f"**Confidence:** {frame_data.get('yoga_confidence', 0):.2f}")
            
            # Cleanup
            os.unlink(temp_video_path)
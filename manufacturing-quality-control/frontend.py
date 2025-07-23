import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import os
from typing import Dict, List
import time

# Import backend functionality
from main import QualityControlSystem, DefectInfo

# Page configuration
st.set_page_config(
    page_title="Manufacturing Quality Control System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric {
        background-color: #2c303b; /* darker gray */
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
        color: #ffffff; /* white text */
    }

    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #14532d; /* dark green */
        border: 1px solid #22c55e; /* bright green border */
        color: #bbf7d0; /* light green text */
    }

    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #78350f; /* dark amber */
        border: 1px solid #f59e0b; /* amber border */
        color: #fde68a; /* amber text */
    }

    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #7f1d1d; /* dark red */
        border: 1px solid #ef4444; /* bright red border */
        color: #fecaca; /* light red text */
    }
</style>
""", unsafe_allow_html=True)

class QualityControlApp:
    """Streamlit application for manufacturing quality control"""
    
    def __init__(self):
        self.initialize_session_state()
        self.qc_system = None
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = []
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = None
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
            
    def load_model(self):
        """Load the quality control model"""
        if not st.session_state.model_loaded:
            with st.spinner("Loading quality control model..."):
                self.qc_system = QualityControlSystem(
                    confidence_threshold=st.session_state.get('confidence_threshold', 0.5)
                )
                st.session_state.model_loaded = True
                
    def render_sidebar(self):
        """Render sidebar with settings and info"""
        st.sidebar.title("⚙️ Settings")
        
        # Model settings
        st.sidebar.subheader("Model Configuration")
        confidence = st.sidebar.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Lower values detect more defects but may include false positives"
        )
        
        if confidence != st.session_state.get('confidence_threshold', 0.5):
            st.session_state.confidence_threshold = confidence
            st.session_state.model_loaded = False
            
        # Quality thresholds
        st.sidebar.subheader("Quality Grading Thresholds")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Excellent", "0 defects")
            st.metric("Good", "≤ 2 defects")
        with col2:
            st.metric("Acceptable", "≤ 5 defects")
            st.metric("Reject", "> 5 defects")
            
        # Statistics
        st.sidebar.subheader("📊 Session Statistics")
        if st.session_state.history:
            total_processed = len(st.session_state.history)
            passed = sum(1 for h in st.session_state.history if h['grade'] in ['excellent', 'good', 'acceptable'])
            failed = total_processed - passed
            avg_score = np.mean([h['score'] for h in st.session_state.history])
            
            st.sidebar.metric("Total Processed", total_processed)
            st.sidebar.metric("Pass Rate", f"{(passed/total_processed)*100:.1f}%")
            st.sidebar.metric("Average Score", f"{avg_score:.1f}")
        else:
            st.sidebar.info("No images processed yet")
            
        # Export options
        st.sidebar.subheader("📤 Export")
        if st.sidebar.button("Export Session Report"):
            self.export_report()
            
    def render_header(self):
        """Render application header"""
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.title("🏭 Manufacturing Quality Control System")
            st.markdown("**AI-Powered Defect Detection and Quality Analysis**")
            
    def render_main_tabs(self):
        """Render main application tabs"""
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Image Analysis", "📦 Batch Processing", "📈 Analytics", "📋 History"])
        
        with tab1:
            self.render_single_image_tab()
            
        with tab2:
            self.render_batch_processing_tab()
            
        with tab3:
            self.render_analytics_tab()
            
        with tab4:
            self.render_history_tab()
            
    def render_single_image_tab(self):
        """Render single image analysis tab"""
        st.header("Single Image Quality Inspection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Product Image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image of the product to inspect"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Convert to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                if st.button("🔍 Analyze Quality", type="primary"):
                    self.analyze_single_image(image_cv, uploaded_file.name)
                    
        with col2:
            if 'current_analysis' in st.session_state:
                self.display_analysis_results(st.session_state.current_analysis)
                
    def analyze_single_image(self, image: np.ndarray, filename: str):
        """Analyze a single image for defects"""
        self.load_model()
        
        with st.spinner("Analyzing image for defects..."):
            # Detect anomalies
            defects, annotated_image = self.qc_system.detect_anomalies(image)
            
            # Calculate quality score
            quality_metrics = self.qc_system.calculate_quality_score(defects)
            
            # Store results
            analysis_results = {
                'filename': filename,
                'timestamp': datetime.now(),
                'defects': defects,
                'annotated_image': annotated_image,
                'quality_metrics': quality_metrics,
                'score': quality_metrics['score'],
                'grade': quality_metrics['grade']
            }
            
            st.session_state.current_analysis = analysis_results
            st.session_state.history.append({
                'filename': filename,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'score': quality_metrics['score'],
                'grade': quality_metrics['grade'],
                'defect_count': quality_metrics['defect_count']
            })
            
    def display_analysis_results(self, results: Dict):
        """Display analysis results"""
        st.subheader("Analysis Results")
        
        # Display annotated image
        annotated_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Defect Detection Results", use_column_width=True)
        
        # Quality metrics
        metrics = results['quality_metrics']
        
        # Grade display with color coding
        grade_colors = {
            'excellent': 'success-box',
            'good': 'success-box',
            'acceptable': 'warning-box',
            'reject': 'danger-box'
        }
        
        st.markdown(f"""
        <div class="{grade_colors.get(metrics['grade'], 'warning-box')}">
            <h3>Quality Grade: {metrics['grade'].upper()}</h3>
            <h2>Score: {metrics['score']}/100</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Defects", metrics['defect_count'])
        with col2:
            st.metric("Defect Types", len(metrics['defect_summary']))
        with col3:
            status = "PASS" if metrics['grade'] != 'reject' else "FAIL"
            st.metric("Status", status)
            
        # Defect breakdown
        if metrics['defect_summary']:
            st.subheader("Defect Analysis")
            defect_df = pd.DataFrame(
                list(metrics['defect_summary'].items()),
                columns=['Defect Type', 'Count']
            )
            
            fig = px.bar(defect_df, x='Defect Type', y='Count', 
                        title="Defect Distribution",
                        color='Count',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            
        # Recommendations
        if metrics['recommendations']:
            st.subheader("🔧 Recommendations")
            for rec in metrics['recommendations']:
                st.warning(f"• {rec}")
                
    def render_batch_processing_tab(self):
        """Render batch processing tab"""
        st.header("Batch Quality Inspection")
        
        uploaded_files = st.file_uploader(
            "Upload Multiple Product Images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} images uploaded")
            
            if st.button("🚀 Process Batch", type="primary"):
                self.process_batch(uploaded_files)
                
        if st.session_state.batch_results:
            self.display_batch_results(st.session_state.batch_results)
            
    def process_batch(self, uploaded_files):
        """Process a batch of uploaded files"""
        self.load_model()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save uploaded files temporarily
        temp_paths = []
        for i, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_paths.append(tmp_file.name)
                
            progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.text(f"Processing {i + 1}/{len(uploaded_files)} images...")
            
        # Process batch
        batch_results = self.qc_system.process_batch(temp_paths)
        
        # Update results with original filenames
        for i, result in enumerate(batch_results['individual_results']):
            result['original_name'] = uploaded_files[i].name
            
        st.session_state.batch_results = batch_results
        
        # Cleanup temp files
        for path in temp_paths:
            os.unlink(path)
            
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Batch processing completed! Processed {batch_results['processed']} images.")
        
    def display_batch_results(self, results: Dict):
        """Display batch processing results"""
        st.subheader("Batch Processing Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Processed", results['processed'])
        with col2:
            st.metric("Passed", results['passed'], 
                     f"{(results['passed']/results['processed']*100):.1f}%")
        with col3:
            st.metric("Failed", results['failed'],
                     f"{(results['failed']/results['processed']*100):.1f}%")
        with col4:
            st.metric("Avg Score", f"{results['average_score']:.1f}")
            
        # Processing time
        st.info(f"⏱️ Processing time: {results['processing_time']:.2f} seconds")
        
        # Defect distribution
        if results['defect_distribution']:
            st.subheader("Overall Defect Distribution")
            defect_df = pd.DataFrame(
                list(results['defect_distribution'].items()),
                columns=['Defect Type', 'Total Count']
            )
            
            fig = px.pie(defect_df, values='Total Count', names='Defect Type',
                        title="Defect Type Distribution Across Batch")
            st.plotly_chart(fig, use_container_width=True)
            
        # Individual results table
        st.subheader("Individual Image Results")
        
        results_data = []
        for result in results['individual_results']:
            results_data.append({
                'Image': result['original_name'],
                'Score': result['quality']['score'],
                'Grade': result['quality']['grade'],
                'Defects': result['defects'],
                'Status': 'PASS' if result['quality']['grade'] != 'reject' else 'FAIL'
            })
            
        results_df = pd.DataFrame(results_data)
        
        # Color-code the dataframe
        def color_status(val):
            if val == 'PASS':
                return 'background-color: #d4edda'
            else:
                return 'background-color: #f8d7da'
                
        styled_df = results_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
        
    def render_analytics_tab(self):
        """Render analytics tab"""
        st.header("Production Analytics Dashboard")
        
        if not st.session_state.history:
            st.info("No data available yet. Process some images to see analytics.")
            return
            
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Time range selector
        col1, col2 = st.columns([2, 1])
        with col1:
            time_range = st.selectbox(
                "Select Time Range",
                ["All Time", "Last Hour", "Last 24 Hours", "Last 7 Days"]
            )
            
        # Filter data based on time range
        now = datetime.now()
        if time_range == "Last Hour":
            filtered_df = history_df[history_df['timestamp'] > now - timedelta(hours=1)]
        elif time_range == "Last 24 Hours":
            filtered_df = history_df[history_df['timestamp'] > now - timedelta(days=1)]
        elif time_range == "Last 7 Days":
            filtered_df = history_df[history_df['timestamp'] > now - timedelta(days=7)]
        else:
            filtered_df = history_df
            
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(filtered_df)
        passed = len(filtered_df[filtered_df['grade'].isin(['excellent', 'good', 'acceptable'])])
        failed = total - passed
        avg_score = filtered_df['score'].mean() if total > 0 else 0
        
        with col1:
            st.metric("Total Inspected", total)
        with col2:
            st.metric("Pass Rate", f"{(passed/total*100):.1f}%" if total > 0 else "0%")
        with col3:
            st.metric("Failure Rate", f"{(failed/total*100):.1f}%" if total > 0 else "0%")
        with col4:
            st.metric("Avg Quality Score", f"{avg_score:.1f}")
            
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality score trend
            st.subheader("Quality Score Trend")
            if len(filtered_df) > 1:
                fig = px.line(filtered_df, x='timestamp', y='score',
                             title="Quality Score Over Time",
                             markers=True)
                fig.add_hline(y=70, line_dash="dash", line_color="red",
                             annotation_text="Minimum Acceptable Score")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need more data points to show trend")
                
        with col2:
            # Grade distribution
            st.subheader("Grade Distribution")
            grade_counts = filtered_df['grade'].value_counts()
            fig = px.pie(values=grade_counts.values, names=grade_counts.index,
                        title="Quality Grade Distribution",
                        color_discrete_map={
                            'excellent': '#28a745',
                            'good': '#90EE90',
                            'acceptable': '#ffc107',
                            'reject': '#dc3545'
                        })
            st.plotly_chart(fig, use_container_width=True)
            
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            # Defect frequency
            st.subheader("Defect Frequency Analysis")
            defect_bins = [0, 1, 3, 5, 10, 20]
            defect_labels = ['0', '1-2', '3-4', '5-9', '10+']
            filtered_df['defect_range'] = pd.cut(filtered_df['defect_count'], 
                                                bins=defect_bins, 
                                                labels=defect_labels,
                                                include_lowest=True)
            defect_freq = filtered_df['defect_range'].value_counts().sort_index()
            
            fig = px.bar(x=defect_freq.index, y=defect_freq.values,
                        title="Defect Count Distribution",
                        labels={'x': 'Number of Defects', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Production quality heatmap (by hour and day)
            st.subheader("Quality Heatmap")
            if len(filtered_df) > 10:
                filtered_df['hour'] = filtered_df['timestamp'].dt.hour
                filtered_df['day'] = filtered_df['timestamp'].dt.day_name()
                
                pivot_table = filtered_df.pivot_table(
                    values='score', 
                    index='hour', 
                    columns='day', 
                    aggfunc='mean'
                )
                
                fig = px.imshow(pivot_table,
                              title="Average Quality Score by Day and Hour",
                              labels=dict(x="Day of Week", y="Hour of Day", color="Avg Score"),
                              color_continuous_scale="RdYlGn")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need more data to generate heatmap")
                
        # Statistical insights
        st.subheader("📊 Statistical Insights")
        
        insights = []
        
        if total > 0:
            # Best performing period
            if len(filtered_df) > 1:
                best_score_idx = filtered_df['score'].idxmax()
                best_time = filtered_df.loc[best_score_idx, 'timestamp']
                best_score = filtered_df.loc[best_score_idx, 'score']
                insights.append(f"🏆 Best quality score: {best_score:.1f} at {best_time.strftime('%Y-%m-%d %H:%M')}")
                
            # Most common defect count
            most_common_defects = filtered_df['defect_count'].mode().values[0]
            insights.append(f"📌 Most common defect count: {most_common_defects}")
            
            # Quality trend
            if len(filtered_df) > 5:
                recent_avg = filtered_df.tail(5)['score'].mean()
                overall_avg = filtered_df['score'].mean()
                trend = "improving" if recent_avg > overall_avg else "declining"
                insights.append(f"📈 Quality trend: {trend} (recent avg: {recent_avg:.1f} vs overall: {overall_avg:.1f})")
                
        for insight in insights:
            st.info(insight)
            
    def render_history_tab(self):
        """Render history tab"""
        st.header("Inspection History")
        
        if not st.session_state.history:
            st.info("No inspection history available.")
            return
            
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.history)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            grade_filter = st.multiselect(
                "Filter by Grade",
                options=['excellent', 'good', 'acceptable', 'reject'],
                default=['excellent', 'good', 'acceptable', 'reject']
            )
            
        with col2:
            score_range = st.slider(
                "Filter by Score Range",
                min_value=0,
                max_value=100,
                value=(0, 100)
            )
            
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=['timestamp', 'score', 'defect_count'],
                index=0
            )
            sort_order = st.radio("Order", ['Descending', 'Ascending'])
            
        # Apply filters
        filtered_history = history_df[
            (history_df['grade'].isin(grade_filter)) &
            (history_df['score'] >= score_range[0]) &
            (history_df['score'] <= score_range[1])
        ]
        
        # Sort
        ascending = sort_order == 'Ascending'
        filtered_history = filtered_history.sort_values(by=sort_by, ascending=ascending)
        
        # Display statistics
        st.subheader("Filtered Results Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(filtered_history))
        with col2:
            st.metric("Average Score", f"{filtered_history['score'].mean():.1f}" if len(filtered_history) > 0 else "N/A")
        with col3:
            st.metric("Average Defects", f"{filtered_history['defect_count'].mean():.1f}" if len(filtered_history) > 0 else "N/A")
            
        # Display table
        st.subheader("Inspection Records")
        
        # Format the dataframe for display
        display_df = filtered_history.copy()
        display_df['Status'] = display_df['grade'].apply(
            lambda x: 'PASS' if x in ['excellent', 'good', 'acceptable'] else 'FAIL'
        )
        
        # Reorder columns
        display_df = display_df[['timestamp', 'filename', 'score', 'grade', 'defect_count', 'Status']]
        
        # Apply styling
        def style_dataframe(df):
            return df.style.apply(lambda x: ['background-color: #d4edda' if v == 'PASS' 
                                            else 'background-color: #f8d7da' if v == 'FAIL' 
                                            else '' for v in x], 
                                subset=['Status'])
            
        styled_df = style_dataframe(display_df)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Export functionality
        if st.button("📥 Download History as CSV"):
            csv = filtered_history.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"inspection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
    def export_report(self):
        """Export comprehensive session report"""
        if not st.session_state.history:
            st.sidebar.error("No data to export")
            return
            
        report = {
            "report_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_inspections": len(st.session_state.history),
                "pass_rate": sum(1 for h in st.session_state.history if h['grade'] != 'reject') / len(st.session_state.history) * 100,
                "average_score": np.mean([h['score'] for h in st.session_state.history]),
                "total_defects": sum(h['defect_count'] for h in st.session_state.history)
            },
            "grade_distribution": pd.DataFrame(st.session_state.history)['grade'].value_counts().to_dict(),
            "detailed_history": st.session_state.history
        }
        
        json_str = json.dumps(report, indent=4, default=str)
        
        st.sidebar.download_button(
            label="💾 Download Report (JSON)",
            data=json_str,
            file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    def run(self):
        """Run the Streamlit application"""
        self.render_sidebar()
        self.render_header()
        self.render_main_tabs()
        

# Main execution
if __name__ == "__main__":
    # Create and run the app
    app = QualityControlApp()
    app.run()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Manufacturing Quality Control System v1.0 | Powered by YOLOv8 & MLflow</p>
        </div>
        """,
        unsafe_allow_html=True
    )
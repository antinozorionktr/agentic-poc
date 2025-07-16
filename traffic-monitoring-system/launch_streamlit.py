#!/usr/bin/env python3
"""
Launch script for Enhanced Streamlit Traffic Monitoring App
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"🎯 CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU detected - using CPU (slower processing)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - installing...")
        return False

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'data/videos',
        'data/outputs', 
        'data/ua_detrac/images/train',
        'data/ua_detrac/images/val',
        'data/ua_detrac/labels/train',
        'data/ua_detrac/labels/val'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("📁 Created directory structure")

def download_yolo_model():
    """Download YOLO model if not exists"""
    try:
        from ultralytics import YOLO
        print("🤖 Checking YOLO model...")
        model = YOLO('yolov8n.pt')  # This will download if not exists
        print("✅ YOLO model ready")
        return True
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        return False

def launch_app():
    """Launch the Streamlit app with optimal settings"""
    print("\n🚀 Launching Enhanced Traffic Monitoring System...")
    print("🌐 App will open at: http://localhost:8501")
    print("📱 For mobile access, use your computer's IP address")
    print("⏹️  Press Ctrl+C to stop the app")
    print("\n" + "="*50)
    
    # Check if custom app file exists
    app_file = "streamlit_traffic_app.py"
    if not os.path.exists(app_file):
        print(f"❌ {app_file} not found!")
        print("💡 Please save the enhanced Streamlit app code as 'streamlit_traffic_app.py'")
        return False
    
    try:
        # Launch with optimal configuration
        cmd = [
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.maxUploadSize", "500",  # 500MB max upload
            "--server.maxMessageSize", "500",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return False

def main():
    print("🔍 System Check...")
    
    # Setup directories
    setup_directories()
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Download YOLO model
    if not download_yolo_model():
        print("⚠️  YOLO model check failed, but continuing...")
    
    # Performance recommendations
    if gpu_available:
        print("\n🎯 Performance Tips:")
        print("- GPU detected - you'll get fast processing!")
        print("- Try processing 100-300 frames for best experience")
    else:
        print("\n💡 Performance Tips:")
        print("- Using CPU - start with 50-100 frames")
        print("- Consider smaller video files (<100MB)")
        print("- Close other applications for better performance")
    
    print("\n📋 Features Available:")
    print("✅ Real-time video processing")
    print("✅ Live analytics dashboard") 
    print("✅ Multi-object tracking")
    print("✅ UA-DETRAC dataset testing")
    print("✅ Demo video downloads")
    print("✅ Interactive charts and metrics")
    
    # Launch app
    success = launch_app()
    
    if not success:
        print("\n🛠️  Troubleshooting:")
        print("1. Check if all dependencies are installed: pip install -r requirements.txt")
        print("2. Ensure streamlit_traffic_app.py exists in current directory")
        print("3. Try running manually: streamlit run streamlit_traffic_app.py")

if __name__ == "__main__":
    main()

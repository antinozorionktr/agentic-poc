## Fitness & Health Tracker (Pose Estimation)
 
**Tech Stack:** 

YOLOv8-Pose + MediaPipe + OpenCV + FastAPI 

**Data Sources:**
 
- COCO pose keypoints
- Yoga/exercise videos (YouTube)
- Custom workout datasets 

**Key Features:** 

Pose detection, form analysis, rep counting, progress tracking 

**Hardware:** 

Ubuntu server + RTX 3060/4060 (real-time pose processing)
 
## Hardware Recommendations
 
**Minimum:** 
RTX 3060 (12GB) - handles single stream inference 

**Recommended:** 

RTX 4070 (12GB) - multiple streams + training 

**Optimal:** 

RTX 4080 (16GB) - large models + batch processing

##

*Steps To Run*

- Clone the repository
    - git clone git@github.com:antinozorionktr/agentic-poc.git
- Navigate inside the directory
    - cd Agentic-POCs/fitness-health-tracker
- Create a Virtual Environment
    - python3.10 -m venv env
- Activate Virtual Environment
    - source env/bin/activate
- Install requirements
    - pip install -r requirements.txt
- Run Backend (Terminal 1)
    - python main.py
- Run Frontend (Terminal 2)
    - streamlit run frontend.py
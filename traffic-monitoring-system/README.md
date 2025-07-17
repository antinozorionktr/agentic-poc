## Traffic Monitoring System (Object Detection)
 
**Tech Stack:** 

- YOLOv8
- OpenCV
- ByteTrack
- FastAPI 

**Data Sources:**
 
- UA-DETRAC dataset
- Traffic camera footage (YouTube)
- Cityscapes dataset 

**Key Features:** 

- Vehicle Detection
- Counting
- Speed Estimation
- Traffic Flow Analysis 

**Hardware:** 

- Ubuntu server
- RTX 3060/4060 (8GB VRAM sufficient)

*Steps To Run*

- Clone the repository
    - git clone git@github.com:antinozorionktr/agentic-poc.git
- Navigate inside the directory
    - cd Agentic-POCs/traffic-monitoring-system
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
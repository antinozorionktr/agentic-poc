## 3. Agricultural Monitoring Platform (Segmentation)
 
**Tech Stack:** 

- YOLOv8-Seg
- GDAL
- FastAPI
- PostgreSQL 

**Data Sources:**
 
- Agriculture-Vision dataset
- Sentinel-2 satellite imagery
- Drone crop footage 

**Key Features:** 

- Crop Segmentation
- Health Assessment 
- Yield prediction
- Disease detection 

**Hardware:** 

- Ubuntu Server
- RTX 3080/4080 (large image processing)

##

*Steps To Run*

- Clone the repository
    - git clone git@github.com:antinozorionktr/agentic-poc.git
- Navigate inside the directory
    - cd Agentic-POCs/agricultural-monitoring-platform
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
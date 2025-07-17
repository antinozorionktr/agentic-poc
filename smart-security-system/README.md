## Smart Security System (Classification + Detection)
 
**Tech Stack:**

- YOLOv8
- OpenCV
- DeepSORT
- Streamlit 

**Data Sources:**
 
- COCO person class
- Open Images security footage
- Custom CCTV datasets 

**Key Features:** 

- Person detection
- Intruder classification
- Behavior analysis
- Alert system 

**Hardware:**

- Ubuntu server
- RTX 3070/4070 (real-time multi-camera processing)

##

*Steps To Run*

- Clone the repository
    - git clone git@github.com:antinozorionktr/agentic-poc.git
- Navigate inside the directory
    - cd Agentic-POCs/smart-security-system
- Create a Virtual Environment
    - python3.10 -m venv env
- Activate Virtual Environment
    - source env/bin/activate
- Install requirements
    - pip install -r requirements.txt
- Run Application
    - streamlit run main.py
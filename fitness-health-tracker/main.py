from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import tempfile
import os
from typing import List, Dict, Any
from pydantic import BaseModel
import json
from datetime import datetime
import math
import asyncio
import base64
from io import BytesIO
import threading
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
import cvzone

app = FastAPI(title="Fitness Tracker API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize YOLOv8 models
try:
    yolo_model = YOLO('yolov8n-pose.pt')  # For pose estimation
    yolo_detection_model = YOLO('yolov8n.pt')  # For person detection
except:
    yolo_model = None
    yolo_detection_model = None
    print("YOLOv8 models not found. MediaPipe will be used instead.")

# Yoga poses list
yoga_poses = [
    "adho mukha svanasana", "adho mukha vriksasana", "agnistambhasana", "ananda balasana",
    "anantasana", "anjaneyasana", "ardha bhekasana", "ardha chandrasana", "ardha matsyendrasana",
    "ardha pincha mayurasana", "ardha uttanasana", "ashtanga namaskara", "astavakrasana",
    "baddha konasana", "bakasana", "balasana", "bhairavasana", "bharadvajasana i", "bhekasana",
    "bhujangasana", "bhujapidasana", "bitilasana", "camatkarasana", "chakravakasana",
    "chaturanga dandasana", "dandasana", "dhanurasana", "durvasasana", "dwi pada viparita dandasana",
    "eka pada koundinyanasana i", "eka pada koundinyanasana ii", "eka pada rajakapotasana",
    "eka pada rajakapotasana ii", "ganda bherundasana", "garbha pindasana", "garudasana",
    "gomukhasana", "halasana", "hanumanasana", "janu sirsasana", "kapotasana", "krounchasana",
    "kurmasana", "lolasana", "makara adho mukha svanasana", "makarasana", "malasana",
    "marichyasana i", "marichyasana iii", "marjaryasana", "matsyasana", "mayurasana",
    "natarajasana", "padangusthasana", "padmasana", "parighasana", "paripurna navasana",
    "parivrtta janu sirsasana", "parivrtta parsvakonasana", "parivrtta trikonasana",
    "parsva bakasana", "parsvottanasana", "pasasana", "paschimottanasana", "phalakasana",
    "pincha mayurasana", "prasarita padottanasana", "purvottanasana", "salabhasana",
    "salamba bhujangasana", "salamba sarvangasana", "salamba sirsasana", "savasana",
    "setu bandha sarvangasana", "simhasana", "sukhasana", "supta baddha konasana",
    "supta matsyendrasana", "supta padangusthasana", "supta virasana", "tadasana",
    "tittibhasana", "tolasana", "tulasana", "upavistha konasana", "urdhva dhanurasana",
    "urdhva hastasana", "urdhva mukha svanasana", "urdhva prasarita eka padasana", "ustrasana",
    "utkatasana", "uttana shishosana", "uttanasana", "utthita ashwa sanchalanasana",
    "utthita hasta padangustasana", "utthita parsvakonasana", "utthita trikonasana",
    "vajrasana", "vasisthasana", "viparita karani", "virabhadrasana i", "virabhadrasana ii",
    "virabhadrasana iii", "virasana", "vriksasana", "vrischikasana", "yoganidrasana"
]

# YOLO class names for person detection
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", 
    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "mobile phone","microwave","oven","toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"
]

# Initialize Yoga Model
yoga_model = None

def load_yoga_model():
    """Load the yoga pose classification model"""
    global yoga_model
    try:
        yoga_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(64, 64, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
            
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
            
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
            
            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(107, activation='softmax')
        ])
        
        # Try to load weights if available
        if os.path.exists('yoga-model.h5'):
            yoga_model.load_weights('yoga-model.h5')
            print("✅ Yoga model weights loaded successfully!")
        else:
            print("⚠️ Yoga model weights not found. Using untrained model.")
            
    except Exception as e:
        print(f"❌ Failed to load yoga model: {e}")
        yoga_model = None

def preprocess_for_yoga(image):
    """Preprocess image for yoga pose classification"""
    try:
        target_size = (64, 64)
        img = cv2.resize(image, target_size)
        img = img / 255.0  
        img = np.expand_dims(img, axis=0) 
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_yoga_pose(cropped_image):
    """Predict yoga pose from cropped person image"""
    if yoga_model is None:
        return -1, "model_not_loaded"
    
    try:
        processed_img = preprocess_for_yoga(cropped_image)
        if processed_img is None:
            return -1, "preprocessing_failed"
        
        prediction = yoga_model.predict(processed_img, verbose=0)
        predicted_class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        if predicted_class_index < len(yoga_poses):
            return predicted_class_index, yoga_poses[predicted_class_index], confidence
        else:
            return -1, "unknown_pose", 0.0
            
    except Exception as e:
        print(f"Error predicting yoga pose: {e}")
        return -1, "prediction_error", 0.0

# Load the yoga model on startup
load_yoga_model()

class ExerciseStats(BaseModel):
    exercise_type: str
    reps: int
    duration: float
    form_score: float
    timestamp: datetime

class VideoAnalysisRequest(BaseModel):
    frame_data: str  # Base64 encoded frame
    timestamp: float

class LiveSessionManager:
    def __init__(self):
        self.active_sessions = {}
        self.session_counter = 0
    
    def create_session(self, websocket):
        session_id = f"session_{self.session_counter}"
        self.session_counter += 1
        self.active_sessions[session_id] = {
            'websocket': websocket,
            'fitness_tracker': FitnessTracker(),
            'last_analysis': time.time(),
            'frame_count': 0
        }
        return session_id
    
    def get_session(self, session_id):
        return self.active_sessions.get(session_id)
    
    def remove_session(self, session_id):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

# Global session manager
session_manager = LiveSessionManager()

class PoseKeypoints(BaseModel):
    keypoints: List[List[float]]
    confidence: float
    exercise_detected: str

class FitnessTracker:
    def __init__(self):
        self.exercise_data = []
        self.rep_counters = {
            'pushup': 0,
            'squat': 0,
            'pullup': 0,
            'plank': 0,
            'yoga_poses': 0
        }
        self.previous_angles = {}
        self.exercise_states = {}
        self.yoga_pose_history = []
        self.current_yoga_pose = None
        self.yoga_pose_duration = {}
        
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
    
    def detect_pushup(self, landmarks):
        """Detect pushup exercise and count reps"""
        try:
            # Get coordinates for pushup analysis
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate elbow angle
            elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Pushup detection logic
            if elbow_angle > 160:  # Arms extended (up position)
                self.exercise_states['pushup'] = 'up'
            elif elbow_angle < 90 and self.exercise_states.get('pushup') == 'up':  # Arms bent (down position)
                self.rep_counters['pushup'] += 1
                self.exercise_states['pushup'] = 'down'
                
            return elbow_angle, self.rep_counters['pushup']
        except:
            return 0, self.rep_counters['pushup']
    
    def detect_squat(self, landmarks):
        """Detect squat exercise and count reps"""
        try:
            # Get coordinates for squat analysis
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate knee angle
            knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # Squat detection logic
            if knee_angle > 160:  # Standing position
                self.exercise_states['squat'] = 'up'
            elif knee_angle < 120 and self.exercise_states.get('squat') == 'up':  # Squatting position
                self.rep_counters['squat'] += 1
                self.exercise_states['squat'] = 'down'
                
            return knee_angle, self.rep_counters['squat']
        except:
            return 0, self.rep_counters['squat']
    
    def track_yoga_pose(self, pose_name, confidence):
        """Track yoga pose duration and transitions"""
        current_time = time.time()
        
        if pose_name != self.current_yoga_pose:
            # Pose changed
            if self.current_yoga_pose:
                # Log previous pose duration
                duration = current_time - self.yoga_pose_start_time
                if self.current_yoga_pose not in self.yoga_pose_duration:
                    self.yoga_pose_duration[self.current_yoga_pose] = 0
                self.yoga_pose_duration[self.current_yoga_pose] += duration
            
            # Start tracking new pose
            self.current_yoga_pose = pose_name
            self.yoga_pose_start_time = current_time
            self.yoga_pose_history.append({
                'pose': pose_name,
                'timestamp': current_time,
                'confidence': confidence
            })
            
            # Count unique yoga poses
            unique_poses = len(set([h['pose'] for h in self.yoga_pose_history if h['pose'] != 'unknown_pose']))
            self.rep_counters['yoga_poses'] = unique_poses
    
    def analyze_form(self, landmarks, exercise_type):
        """Analyze exercise form and provide score"""
        form_score = 85.0  # Default good form score
        
        try:
            if exercise_type == 'pushup':
                # Check if back is straight
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                
                # Calculate body alignment
                body_angle = self.calculate_angle(
                    [left_shoulder.x, left_shoulder.y],
                    [left_hip.x, left_hip.y],
                    [left_ankle.x, left_ankle.y]
                )
                
                # Penalize if body is not straight
                if abs(body_angle - 180) > 20:
                    form_score -= 15
                    
            elif exercise_type == 'squat':
                # Check knee alignment and depth
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                
                # Check if knees are properly aligned
                knee_ankle_diff = abs(left_knee.x - left_ankle.x)
                if knee_ankle_diff > 0.1:  # Knees caving in
                    form_score -= 20
                    
        except:
            form_score = 70.0  # Lower score if analysis fails
            
        return max(0, min(100, form_score))

# Initialize fitness tracker
fitness_tracker = FitnessTracker()

@app.get("/")
async def root():
    return {"message": "Fitness Tracker API is running!"}

@app.post("/analyze_pose")
async def analyze_pose(file: UploadFile = File(...)):
    """Analyze pose from uploaded image/video frame"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded file
        contents = await file.read()
        
        # Ensure we have data
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Create numpy array from bytes
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            # Try alternative decoding methods
            try:
                # Save to temporary file and read again
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(contents)
                    tmp_file.flush()
                    frame = cv2.imread(tmp_file.name)
                    os.unlink(tmp_file.name)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not decode image: {str(e)}")
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format or corrupted file")
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract keypoints
            keypoints = []
            for landmark in landmarks:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            # Detect exercises
            pushup_angle, pushup_reps = fitness_tracker.detect_pushup(landmarks)
            squat_angle, squat_reps = fitness_tracker.detect_squat(landmarks)
            
            # Determine primary exercise (simplified logic)
            if pushup_angle > 0 and pushup_angle < 170:
                exercise_type = "pushup"
                current_reps = pushup_reps
            elif squat_angle > 0 and squat_angle < 170:
                exercise_type = "squat"
                current_reps = squat_reps
            else:
                exercise_type = "unknown"
                current_reps = 0
            
            # Analyze form
            form_score = fitness_tracker.analyze_form(landmarks, exercise_type)
            
            return {
                "success": True,
                "keypoints": keypoints,
                "exercise_detected": exercise_type,
                "rep_count": current_reps,
                "form_score": form_score,
                "angles": {
                    "pushup_angle": pushup_angle,
                    "squat_angle": squat_angle
                },
                "total_reps": fitness_tracker.rep_counters
            }
        else:
            return {
                "success": False,
                "message": "No pose detected",
                "keypoints": [],
                "exercise_detected": "none",
                "rep_count": 0,
                "form_score": 0
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get current workout statistics"""
    return {
        "rep_counters": fitness_tracker.rep_counters,
        "total_exercises": len(fitness_tracker.exercise_data),
        "session_start": datetime.now().isoformat()
    }

@app.post("/reset_counters")
async def reset_counters():
    """Reset all rep counters"""
    fitness_tracker.rep_counters = {
        'pushup': 0,
        'squat': 0,
        'pullup': 0,
        'plank': 0,
        'yoga_poses': 0
    }
    fitness_tracker.exercise_states = {}
    fitness_tracker.yoga_pose_history = []
    fitness_tracker.current_yoga_pose = None
    fitness_tracker.yoga_pose_duration = {}
    return {"message": "Counters reset successfully"}

@app.get("/yoga_poses")
async def get_yoga_poses():
    """Get list of all supported yoga poses"""
    return {
        "total_poses": len(yoga_poses),
        "poses": yoga_poses,
        "model_loaded": yoga_model is not None
    }

@app.get("/yoga_stats")
async def get_yoga_stats():
    """Get yoga-specific statistics"""
    return {
        "current_pose": fitness_tracker.current_yoga_pose,
        "pose_history": fitness_tracker.yoga_pose_history,
        "pose_durations": fitness_tracker.yoga_pose_duration,
        "unique_poses_count": fitness_tracker.rep_counters['yoga_poses'],
        "total_poses_detected": len(fitness_tracker.yoga_pose_history)
    }

@app.post("/log_exercise")
async def log_exercise(exercise_data: ExerciseStats):
    """Log completed exercise session"""
    fitness_tracker.exercise_data.append(exercise_data.dict())
    return {"message": "Exercise logged successfully"}

@app.get("/exercise_history")
async def get_exercise_history():
    """Get exercise history"""
    return {
        "history": fitness_tracker.exercise_data,
        "total_sessions": len(fitness_tracker.exercise_data)
    }

@app.websocket("/ws/live_analysis/{session_id}")
async def websocket_live_analysis(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for live video analysis"""
    await websocket.accept()
    
    # Get or create session
    session = session_manager.get_session(session_id)
    if not session:
        session_id = session_manager.create_session(websocket)
        session = session_manager.get_session(session_id)
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            
            if data.get('type') == 'frame':
                frame_data = data.get('frame_data')
                timestamp = data.get('timestamp', time.time())
                
                # Process frame
                result = await process_video_frame(frame_data, session['fitness_tracker'])
                
                # Add session info
                result['session_id'] = session_id
                result['frame_count'] = session['frame_count']
                session['frame_count'] += 1
                
                # Send result back
                await websocket.send_json(result)
                
            elif data.get('type') == 'reset':
                # Reset counters for this session
                session['fitness_tracker'].rep_counters = {
                    'pushup': 0, 'squat': 0, 'pullup': 0, 'plank': 0
                }
                session['fitness_tracker'].exercise_states = {}
                await websocket.send_json({'type': 'reset_complete'})
                
    except WebSocketDisconnect:
        session_manager.remove_session(session_id)
    except Exception as e:
        await websocket.send_json({'error': str(e)})
        session_manager.remove_session(session_id)

async def process_video_frame(frame_data: str, fitness_tracker: FitnessTracker):
    """Process a single video frame for pose analysis"""
    try:
        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data.split(',')[-1])  # Remove data:image/jpeg;base64, prefix
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {'success': False, 'error': 'Could not decode frame'}

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # First detect persons using YOLO
        yoga_pose_detected = None
        yoga_confidence = 0.0

        if yolo_detection_model is not None:
            try:
                results = yolo_detection_model(frame, stream=True)
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            conf = float(box.conf[0])
                            id = int(box.cls[0])

                            if id < len(classNames) and classNames[id] == "person" and conf > 0.5:
                                # Crop person from image
                                cropped_img = frame[y1:y2, x1:x2]
                                if cropped_img.size > 0:
                                    # Predict yoga pose
                                    pose_idx, pose_name, yoga_confidence = predict_yoga_pose(cropped_img)
                                    if pose_idx >= 0 and yoga_confidence > 0.3:
                                        yoga_pose_detected = pose_name
                                        fitness_tracker.track_yoga_pose(pose_name, yoga_confidence)
                                        break
            except Exception as e:
                print(f"YOLO person detection error: {e}")

        # Process with MediaPipe for traditional exercises
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract keypoints
            keypoints = []
            for landmark in landmarks:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # Detect traditional exercises
            pushup_angle, pushup_reps = fitness_tracker.detect_pushup(landmarks)
            squat_angle, squat_reps = fitness_tracker.detect_squat(landmarks)

            # Determine primary exercise
            exercise_type = "unknown"
            current_reps = 0

            if yoga_pose_detected:
                exercise_type = "yoga"
                current_reps = fitness_tracker.rep_counters['yoga_poses']
            elif pushup_angle > 0 and pushup_angle < 170:
                exercise_type = "pushup"
                current_reps = pushup_reps
            elif squat_angle > 0 and squat_angle < 170:
                exercise_type = "squat"
                current_reps = squat_reps

            # Analyze form
            form_score = fitness_tracker.analyze_form(landmarks, exercise_type)

            return {
                'success': True,
                'keypoints': keypoints,
                'exercise_detected': exercise_type,
                'rep_count': current_reps,
                'form_score': form_score,
                'yoga_pose': yoga_pose_detected if yoga_pose_detected else None,
                'yoga_confidence': yoga_confidence,
                'angles': {
                    'pushup_angle': pushup_angle,
                    'squat_angle': squat_angle
                },
                'total_reps': fitness_tracker.rep_counters,
                'yoga_pose_history': fitness_tracker.yoga_pose_history[-5:],  # Last 5 poses
                'timestamp': time.time()
            }

        else:
            # No pose detected with MediaPipe, but might have yoga pose from YOLO
            if yoga_pose_detected:
                return {
                    'success': True,
                    'keypoints': [],
                    'exercise_detected': 'yoga',
                    'rep_count': fitness_tracker.rep_counters['yoga_poses'],
                    'form_score': 85.0,  # Default score for yoga
                    'yoga_pose': yoga_pose_detected,
                    'yoga_confidence': yoga_confidence,
                    'angles': {'pushup_angle': 0, 'squat_angle': 0},
                    'total_reps': fitness_tracker.rep_counters,
                    'yoga_pose_history': fitness_tracker.yoga_pose_history[-5:],
                    'timestamp': time.time()
                }
            else:
                return {
                    'success': False,
                    'message': 'No pose detected',
                    'keypoints': [],
                    'exercise_detected': 'none',
                    'rep_count': 0,
                    'form_score': 0,
                    'yoga_pose': None,
                    'yoga_confidence': 0.0,
                    'timestamp': time.time()
                }

    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.post("/analyze_video_frame")
async def analyze_video_frame(request: VideoAnalysisRequest):
    """Analyze a single video frame"""
    result = await process_video_frame(request.frame_data, fitness_tracker)
    return result

@app.get("/camera_stream")
async def camera_stream():
    """Start camera stream for testing"""
    def generate_frames():
        cap = cv2.VideoCapture(0)
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze pose from uploaded video file"""
    try:
        # Validate file type
        if not file.content_type or not (file.content_type.startswith('video/') or file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Read uploaded video file
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video file uploaded")
        
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(contents)
            tmp_file.flush()
            
            try:
                # Process video
                results = await process_video_file(tmp_file.name, fitness_tracker)
                return results
            finally:
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

async def process_video_file(video_path: str, fitness_tracker: FitnessTracker):
    """Process entire video file for pose analysis"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"success": False, "error": "Could not open video file"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frame_results = []
        exercise_summary = {}
        form_scores = []
        frame_count = 0
        
        # Process every nth frame to improve performance
        frame_skip = max(1, int(fps // 5))  # Process ~5 frames per second
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for performance
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract keypoints
                keypoints = []
                for landmark in landmarks:
                    keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                # Detect exercises
                pushup_angle, pushup_reps = fitness_tracker.detect_pushup(landmarks)
                squat_angle, squat_reps = fitness_tracker.detect_squat(landmarks)
                
                # Determine primary exercise
                current_exercise = "unknown"
                if pushup_angle > 0 and pushup_angle < 170:
                    current_exercise = "pushup"
                elif squat_angle > 0 and squat_angle < 170:
                    current_exercise = "squat"
                
                # Analyze form
                form_score = fitness_tracker.analyze_form(landmarks, current_exercise)
                if form_score > 0:
                    form_scores.append(form_score)
                
                # Store frame result
                frame_result = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'exercise_detected': current_exercise,
                    'form_score': form_score,
                    'angles': {
                        'pushup_angle': pushup_angle,
                        'squat_angle': squat_angle
                    },
                    'keypoints': keypoints
                }
                frame_results.append(frame_result)
                
                # Update exercise summary
                if current_exercise not in exercise_summary:
                    exercise_summary[current_exercise] = {
                        'frames': 0,
                        'avg_form_score': 0,
                        'duration': 0
                    }
                exercise_summary[current_exercise]['frames'] += 1
            
            frame_count += 1
        
        cap.release()
        
        # Calculate final statistics
        total_reps = fitness_tracker.rep_counters.copy()
        avg_form_score = np.mean(form_scores) if form_scores else 0
        
        # Calculate exercise durations and averages
        for exercise, data in exercise_summary.items():
            if data['frames'] > 0:
                data['duration'] = data['frames'] * frame_skip / fps
                exercise_form_scores = [f['form_score'] for f in frame_results 
                                      if f['exercise_detected'] == exercise and f['form_score'] > 0]
                data['avg_form_score'] = np.mean(exercise_form_scores) if exercise_form_scores else 0
        
        return {
            "success": True,
            "video_info": {
                "duration": duration,
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": len(frame_results)
            },
            "total_reps": total_reps,
            "exercise_summary": exercise_summary,
            "avg_form_score": avg_form_score,
            "frame_results": frame_results[-50:],  # Return last 50 frames to avoid huge response
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/analyze_video_segment")
async def analyze_video_segment(
    file: UploadFile = File(...),
    start_time: float = 0,
    end_time: float = None
):
    """Analyze a specific segment of uploaded video"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        contents = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(contents)
            tmp_file.flush()
            
            try:
                cap = cv2.VideoCapture(tmp_file.name)
                
                if not cap.isOpened():
                    raise HTTPException(status_code=400, detail="Could not open video file")
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Set start position
                start_frame = int(start_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Calculate end frame
                if end_time is None:
                    end_frame = total_frames
                else:
                    end_frame = min(int(end_time * fps), total_frames)
                
                segment_results = []
                current_frame = start_frame
                
                while current_frame < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)
                    
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        
                        # Detect exercises
                        pushup_angle, _ = fitness_tracker.detect_pushup(landmarks)
                        squat_angle, _ = fitness_tracker.detect_squat(landmarks)
                        
                        current_exercise = "unknown"
                        if pushup_angle > 0 and pushup_angle < 170:
                            current_exercise = "pushup"
                        elif squat_angle > 0 and squat_angle < 170:
                            current_exercise = "squat"
                        
                        form_score = fitness_tracker.analyze_form(landmarks, current_exercise)
                        
                        segment_results.append({
                            'frame_number': current_frame,
                            'timestamp': current_frame / fps,
                            'exercise_detected': current_exercise,
                            'form_score': form_score,
                            'angles': {
                                'pushup_angle': pushup_angle,
                                'squat_angle': squat_angle
                            }
                        })
                    
                    current_frame += 1
                
                cap.release()
                
                return {
                    "success": True,
                    "segment_info": {
                        "start_time": start_time,
                        "end_time": end_time or (total_frames / fps),
                        "processed_frames": len(segment_results)
                    },
                    "results": segment_results
                }
                
            finally:
                os.unlink(tmp_file.name)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video segment: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mediapipe_loaded": pose is not None,
        "yolo_loaded": yolo_model is not None,
        "active_sessions": len(session_manager.active_sessions),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
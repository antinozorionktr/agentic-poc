# backend.py - FastAPI Backend using SORT tracker and dedicated license plate detection

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import json
import base64
from datetime import datetime
from collections import defaultdict, deque
import math
from typing import Dict, List, Tuple, Optional
import logging
import string
import re

# Optional imports for OCR and SORT tracker
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("EasyOCR not available. Install with: pip install easyocr")

# SORT tracker implementation (simplified version)
class SORTTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}

    def register(self, bbox):
        self.objects[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []

        input_centroids = []
        for detection in detections:
            x1, y1, x2, y2, score = detection
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(detections[i])
        else:
            object_centroids = []
            for bbox in self.objects.values():
                x1, y1, x2, y2, _ = bbox
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                object_centroids.append((cx, cy))

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = list(self.objects.keys())[row]
                self.objects[object_id] = detections[col]
                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = list(self.objects.keys())[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(detections[col])

        # Return tracked objects with IDs
        result = []
        for object_id, bbox in self.objects.items():
            x1, y1, x2, y2, score = bbox
            result.append([x1, y1, x2, y2, object_id])
        
        return result

# License Plate utilities (from the repository)
class LicensePlateProcessor:
    def __init__(self):
        # Initialize the OCR reader
        if OCR_AVAILABLE:
            self.reader = easyocr.Reader(['en'], gpu=False)
        else:
            self.reader = None
        
        # Mapping dictionaries for character conversion
        self.dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
        self.dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

    def license_complies_format(self, text):
        """Check if the license plate text complies with the required format."""
        if len(text) != 7:
            return False

        if (text[0] in string.ascii_uppercase or text[0] in self.dict_int_to_char.keys()) and \
           (text[1] in string.ascii_uppercase or text[1] in self.dict_int_to_char.keys()) and \
           (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in self.dict_char_to_int.keys()) and \
           (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in self.dict_char_to_int.keys()) and \
           (text[4] in string.ascii_uppercase or text[4] in self.dict_int_to_char.keys()) and \
           (text[5] in string.ascii_uppercase or text[5] in self.dict_int_to_char.keys()) and \
           (text[6] in string.ascii_uppercase or text[6] in self.dict_int_to_char.keys()):
            return True
        else:
            return False

    def format_license(self, text):
        """Format the license plate text by converting characters using the mapping dictionaries."""
        license_plate_ = ''
        mapping = {0: self.dict_int_to_char, 1: self.dict_int_to_char, 4: self.dict_int_to_char, 
                   5: self.dict_int_to_char, 6: self.dict_int_to_char,
                   2: self.dict_char_to_int, 3: self.dict_char_to_int}
        
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_

    def read_license_plate(self, license_plate_crop):
        """Read the license plate text from the given cropped image."""
        if not OCR_AVAILABLE or self.reader is None:
            return None, None

        detections = self.reader.readtext(license_plate_crop)

        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(' ', '')

            if self.license_complies_format(text):
                return self.format_license(text), score

        return None, None

    def get_car(self, license_plate, vehicle_track_ids):
        """Retrieve the vehicle coordinates and ID based on the license plate coordinates."""
        x1, y1, x2, y2, score, class_id = license_plate

        for j in range(len(vehicle_track_ids)):
            xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                return vehicle_track_ids[j]

        return -1, -1, -1, -1, -1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Traffic Monitoring System with SORT and License Plate Detection")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrafficMonitor:
    def __init__(self, vehicle_model_path='yolov8n.pt', plate_model_path='license_plate_detector.pt'):
        """Initialize traffic monitoring system using SORT tracker methodology"""
        # Load models
        self.coco_model = YOLO(vehicle_model_path)
        
        try:
            self.license_plate_detector = YOLO(plate_model_path)
            logger.info("License plate detector model loaded")
        except Exception as e:
            logger.warning(f"Failed to load license plate detector: {e}. Using vehicle model for plates.")
            self.license_plate_detector = YOLO(vehicle_model_path)
        
        # Initialize SORT tracker
        self.mot_tracker = SORTTracker()
        
        # Initialize license plate processor
        self.plate_processor = LicensePlateProcessor()
        
        # Vehicle class IDs in COCO dataset
        self.vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Reset all stats
        self.reset_statistics()
        
    def reset_statistics(self):
        """Reset all traffic statistics and tracking state"""
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 30
        self.pixel_to_meter = 0.05
        
        # Traffic statistics
        self.vehicle_count = defaultdict(int)
        self.vehicle_speeds = defaultdict(list)
        self.traffic_flow = deque(maxlen=300)
        self.current_vehicles = set()
        self.detected_plates = {}
        self.results = {}  # Store results like in the original code
        
        # Reset tracker
        self.mot_tracker = SORTTracker()
        
        logger.info("Traffic statistics reset")
        
    def process_frame(self, frame):
        """Process a single frame using the SORT methodology"""
        self.frame_count += 1
        frame_nmr = self.frame_count - 1
        
        self.results[frame_nmr] = {}
        
        # Detect vehicles
        detections = self.coco_model(frame)[0]
        detections_ = []
        
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles using SORT
        track_ids = self.mot_tracker.update(np.asarray(detections_))
        
        # Update current vehicles
        current_ids = set()
        for track in track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = track
            current_ids.add(int(car_id))
            
            # Count new vehicles
            if int(car_id) not in self.current_vehicles:
                # Find the class of this vehicle from original detections
                vehicle_class = self._find_vehicle_class(track, detections.boxes.data.tolist())
                class_name = self.coco_model.names[vehicle_class] if vehicle_class != -1 else 'vehicle'
                self.vehicle_count[class_name] += 1

        self.current_vehicles = current_ids
        self.traffic_flow.append(len(current_ids))

        # Detect license plates
        license_plates = self.license_plate_detector(frame)[0]
        
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = self.plate_processor.get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                if license_plate_crop.size > 0:
                    # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # Read license plate number
                    license_plate_text, license_plate_text_score = self.plate_processor.read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        self.results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }
                        
                        # Store in detected plates
                        self.detected_plates[int(car_id)] = {
                            'text': license_plate_text,
                            'confidence': license_plate_text_score,
                            'bbox': [x1, y1, x2, y2],
                            'vehicle_class': self._get_vehicle_class_name(car_id, track_ids, detections.boxes.data.tolist())
                        }
                        
                        logger.info(f"Detected plate: {license_plate_text} for vehicle {car_id}")

        # Draw annotations
        annotated_frame = self._draw_annotations(frame, track_ids, frame_nmr)
        
        # Get statistics
        stats = self._get_statistics()
        
        return annotated_frame, stats

    def _find_vehicle_class(self, track, original_detections):
        """Find the class of a tracked vehicle from original detections"""
        xcar1, ycar1, xcar2, ycar2, car_id = track
        
        # Find the detection that best matches this track
        best_iou = 0
        best_class = -1
        
        for detection in original_detections:
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles:
                iou = self._calculate_iou([xcar1, ycar1, xcar2, ycar2], [x1, y1, x2, y2])
                if iou > best_iou:
                    best_iou = iou
                    best_class = int(class_id)
        
        return best_class

    def _get_vehicle_class_name(self, car_id, track_ids, original_detections):
        """Get vehicle class name for a specific car ID"""
        for track in track_ids:
            if track[4] == car_id:
                vehicle_class = self._find_vehicle_class(track, original_detections)
                return self.coco_model.names[vehicle_class] if vehicle_class != -1 else 'vehicle'
        return 'vehicle'

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def _draw_annotations(self, frame, track_ids, frame_nmr):
        """Draw bounding boxes, tracking info, and license plates"""
        
        # Draw vehicle tracks
        for track in track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = track
            x1, y1, x2, y2 = map(int, [xcar1, ycar1, xcar2, ycar2])
            
            # Get color for this track
            color = self._get_color(int(car_id))
            
            # Draw vehicle bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Check if this vehicle has license plate info
            plate_text = ""
            if frame_nmr in self.results and car_id in self.results[frame_nmr]:
                plate_info = self.results[frame_nmr][car_id].get('license_plate', {})
                plate_text = plate_info.get('text', '')
                
                # Draw license plate bounding box
                if 'bbox' in plate_info:
                    px1, py1, px2, py2 = map(int, plate_info['bbox'])
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
            
            # Draw vehicle label
            label = f"Vehicle #{int(car_id)}"
            if plate_text:
                label += f" [{plate_text}]"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw statistics overlay
        self._draw_stats_overlay(frame)
        
        return frame

    def _draw_stats_overlay(self, frame):
        """Draw statistics overlay on frame"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw statistics
        y_offset = 30
        cv2.putText(frame, "Traffic Statistics", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        for vehicle_type, count in self.vehicle_count.items():
            text = f"{vehicle_type}: {count}"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Current vehicles
        y_offset += 10
        text = f"Current vehicles: {len(self.current_vehicles)}"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # License plates detected
        y_offset += 20
        text = f"Plates detected: {len(self.detected_plates)}"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def _get_color(self, track_id):
        """Get consistent color for track ID"""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
        return colors[track_id % len(colors)]

    def _get_statistics(self):
        """Get current traffic statistics"""
        avg_flow = np.mean(self.traffic_flow) if self.traffic_flow else 0
        
        # Calculate progress percentage
        progress = (self.frame_count / self.total_frames * 100) if self.total_frames > 0 else 0
        
        return {
            'vehicle_count': dict(self.vehicle_count),
            'current_vehicles': len(self.current_vehicles),
            'average_flow': float(avg_flow),
            'average_speed': 0.0,  # Speed calculation can be added later
            'detected_plates': dict(self.detected_plates),
            'plates_count': len(self.detected_plates),
            'timestamp': datetime.now().isoformat(),
            'frame_count': self.frame_count,
            'total_frames': self.total_frames,
            'progress_percentage': float(progress)
        }

# Global traffic monitor instance
traffic_monitor = None

@app.on_event("startup")
async def startup_event():
    """Initialize traffic monitor on startup"""
    global traffic_monitor
    try:
        # Try to load YOLOv8 models
        traffic_monitor = TrafficMonitor()
        logger.info("Traffic monitoring system initialized with SORT tracker and license plate detection")
    except Exception as e:
        logger.warning(f"Failed to load models: {str(e)}")
        # Create a mock version for testing
        traffic_monitor = MockTrafficMonitor()

class MockTrafficMonitor:
    """Mock traffic monitor for testing"""
    def __init__(self):
        self.reset_statistics()
        
    def reset_statistics(self):
        """Reset all statistics"""
        self.frame_count = 0
        self.total_frames = 0
        self.vehicle_count = defaultdict(int)
        self.current_vehicles = set()
        self.detected_plates = {}
        
    def process_frame(self, frame):
        """Mock processing"""
        self.frame_count += 1
        
        # Simulate detections
        if self.frame_count % 30 == 0:
            self.vehicle_count['car'] += 1
            vehicle_id = self.frame_count // 30
            self.current_vehicles.add(vehicle_id)
            
            # Simulate license plate detection
            if self.frame_count % 60 == 0:
                self.detected_plates[vehicle_id] = {
                    'text': f"AB{vehicle_id:02d}CD{1234 + vehicle_id}",
                    'confidence': 0.85,
                    'vehicle_class': 'car'
                }
        
        # Add overlay
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Mock Mode (SORT) - Frame {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw mock detection
        if self.frame_count % 10 < 5:
            cv2.rectangle(frame, (100, 100), (300, 200), (0, 255, 0), 2)
            cv2.putText(frame, "Vehicle #1 [AB01CD1234]", (105, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (150, 180), (250, 200), (0, 255, 255), 2)
        
        progress = (self.frame_count / self.total_frames * 100) if self.total_frames > 0 else 0
        
        stats = {
            'vehicle_count': dict(self.vehicle_count),
            'current_vehicles': len(self.current_vehicles),
            'average_flow': len(self.current_vehicles),
            'average_speed': 0.0,
            'detected_plates': dict(self.detected_plates),
            'plates_count': len(self.detected_plates),
            'timestamp': datetime.now().isoformat(),
            'frame_count': self.frame_count,
            'total_frames': self.total_frames,
            'progress_percentage': progress
        }
        
        return frame, stats

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Traffic Monitoring System with SORT Tracker and License Plate Detection", "status": "active"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data['type'] == 'frame':
                # Reset for new video
                if traffic_monitor.frame_count == 0 or data.get('reset', False):
                    traffic_monitor.reset_statistics()
                    logger.info("Statistics reset for new video processing")
                
                # Decode frame
                frame_data = base64.b64decode(data['frame'])
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if 'total_frames' in data:
                    traffic_monitor.total_frames = data['total_frames']
                
                # Process frame
                processed_frame, stats = traffic_monitor.process_frame(frame)
                
                # Encode response
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                encoded_frame = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                    'type': 'processed',
                    'frame': encoded_frame,
                    'stats': stats
                })
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

@app.get("/stats")
async def get_statistics():
    """Get current traffic statistics"""
    if traffic_monitor is None:
        raise HTTPException(status_code=503, detail="Traffic monitor not initialized")
    
    stats = traffic_monitor._get_statistics()
    return stats

@app.get("/plates")
async def get_detected_plates():
    """Get all detected license plates"""
    if traffic_monitor is None:
        raise HTTPException(status_code=503, detail="Traffic monitor not initialized")
    
    return {"detected_plates": traffic_monitor.detected_plates}

@app.post("/reset")
async def reset_statistics():
    """Reset traffic statistics"""
    if traffic_monitor is None:
        raise HTTPException(status_code=503, detail="Traffic monitor not initialized")
    
    traffic_monitor.reset_statistics()
    return {"message": "Statistics reset successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
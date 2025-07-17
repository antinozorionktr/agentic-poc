# backend.py - Fixed FastAPI Backend with improved tracking and plate detection

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

# Enhanced SORT tracker with better ID management
class EnhancedSORTTracker:
    def __init__(self, max_disappeared=50, max_distance=80, min_hits=3):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits  # Minimum hits before considering a track valid
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.hits = {}  # Track how many times we've seen each object
        self.positions_history = defaultdict(list)
        self.tracked_vehicles = set()  # Keep track of all vehicles we've counted

    def register(self, bbox):
        self.objects[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.hits[self.next_object_id] = 1
        
        # Initialize position history
        x1, y1, x2, y2, score = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        self.positions_history[self.next_object_id] = [(center_x, center_y, datetime.now().timestamp())]
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.hits:
            del self.hits[object_id]
        if object_id in self.positions_history:
            del self.positions_history[object_id]

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
            object_ids = list(self.objects.keys())
            
            for object_id in object_ids:
                bbox = self.objects[object_id]
                x1, y1, x2, y2, _ = bbox
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                object_centroids.append((cx, cy))

            if len(object_centroids) > 0 and len(input_centroids) > 0:
                D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_row_indices = set()
                used_col_indices = set()

                for (row, col) in zip(rows, cols):
                    if row in used_row_indices or col in used_col_indices:
                        continue

                    if row >= len(object_ids) or col >= len(detections):
                        continue

                    if D[row, col] > self.max_distance:
                        continue

                    object_id = object_ids[row]
                    self.objects[object_id] = detections[col]
                    self.disappeared[object_id] = 0
                    self.hits[object_id] += 1  # Increment hit count

                    # Update position history
                    x1, y1, x2, y2, score = detections[col]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    timestamp = datetime.now().timestamp()
                    self.positions_history[object_id].append((center_x, center_y, timestamp))
                    
                    # Keep only last 20 positions for speed calculation
                    if len(self.positions_history[object_id]) > 20:
                        self.positions_history[object_id] = self.positions_history[object_id][-20:]

                    used_row_indices.add(row)
                    used_col_indices.add(col)

                unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
                unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

                if D.shape[0] >= D.shape[1]:
                    for row in unused_row_indices:
                        if row < len(object_ids):
                            object_id = object_ids[row]
                            self.disappeared[object_id] += 1
                            if self.disappeared[object_id] > self.max_disappeared:
                                self.deregister(object_id)
                else:
                    for col in unused_col_indices:
                        if col < len(detections):
                            self.register(detections[col])

        # Return only tracks with sufficient hits
        result = []
        for object_id, bbox in self.objects.items():
            if self.hits[object_id] >= self.min_hits:  # Only return established tracks
                x1, y1, x2, y2, score = bbox
                result.append([x1, y1, x2, y2, object_id])
        
        return result

    def get_speed(self, object_id, pixel_to_meter=0.05, fps=30):
        """Calculate speed for a tracked object in km/h with smoothing"""
        if object_id not in self.positions_history or len(self.positions_history[object_id]) < 5:
            return 0.0
        
        positions = self.positions_history[object_id]
        
        # Use a sliding window for speed calculation
        window_size = min(10, len(positions))
        recent_positions = positions[-window_size:]
        
        speeds = []
        for i in range(1, len(recent_positions)):
            prev_pos = recent_positions[i-1]
            curr_pos = recent_positions[i]
            
            distance_pixels = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            distance_meters = distance_pixels * pixel_to_meter
            
            time_diff = curr_pos[2] - prev_pos[2]
            
            if time_diff > 0:
                speed_mps = distance_meters / time_diff
                speed_kmh = speed_mps * 3.6
                speeds.append(speed_kmh)
        
        if speeds:
            # Use median for more stable speed estimation
            median_speed = np.median(speeds)
            return max(0, min(150, median_speed))  # Cap between 0-150 km/h
        
        return 0.0

# Enhanced License Plate processor with better preprocessing
class EnhancedLicensePlateProcessor:
    def __init__(self):
        # Initialize the OCR reader with GPU if available
        if OCR_AVAILABLE:
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                self.reader = easyocr.Reader(['en'], gpu=gpu_available)
                print(f"EasyOCR initialized with GPU: {gpu_available}")
            except Exception as e:
                print(f"EasyOCR initialization failed: {e}")
                self.reader = None
        else:
            self.reader = None
        
        # Mapping dictionaries for character conversion
        self.dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
        self.dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

    def preprocess_plate_image(self, img):
        """Enhanced preprocessing for better OCR accuracy"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Resize if too small
        height, width = gray.shape
        if width < 100:
            scale_factor = 150 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Apply multiple threshold techniques
        _, thresh1 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return [denoised, thresh1, thresh2]

    def license_complies_format(self, text):
        """Check if the license plate text complies with common formats"""
        if len(text) < 4 or len(text) > 10:
            return False
        
        # Check for common patterns (adjust based on your region)
        # Example patterns: ABC123, AB12CD, 12AB34, etc.
        patterns = [
            r'^[A-Z]{2,3}[0-9]{2,4}$',  # AA123, AAA1234
            r'^[0-9]{2,4}[A-Z]{2,3}$',  # 123AA, 1234AAA
            r'^[A-Z]{2}[0-9]{2}[A-Z]{2}$',  # AB12CD
            r'^[0-9]{2}[A-Z]{2}[0-9]{2}$',  # 12AB34
            r'^[A-Z0-9]{4,10}$'  # General alphanumeric
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        return False

    def format_license(self, text):
        """Format the license plate text"""
        # Remove spaces and special characters
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Apply character substitutions for common OCR mistakes
        corrections = {
            'O': '0', '0': 'O',  # Context-dependent
            'I': '1', '1': 'I',
            'S': '5', '5': 'S',
            'B': '8', '8': 'B',
            'G': '6', '6': 'G',
            'Z': '2', '2': 'Z'
        }
        
        # Simple heuristic: if position should be number/letter
        # This is region-specific and should be adjusted
        return text

    def read_license_plate(self, license_plate_crop):
        """Read the license plate text from the given cropped image with multiple attempts"""
        if not OCR_AVAILABLE or self.reader is None:
            return None, None

        try:
            # Get multiple preprocessed versions
            preprocessed_images = self.preprocess_plate_image(license_plate_crop)
            
            all_detections = []
            
            # Try OCR on each preprocessed version
            for img in preprocessed_images:
                detections = self.reader.readtext(img)
                all_detections.extend(detections)
            
            # Also try on original
            detections = self.reader.readtext(license_plate_crop)
            all_detections.extend(detections)

            best_text = None
            best_score = 0

            for detection in all_detections:
                bbox, text, score = detection
                text = text.upper().replace(' ', '')
                text = re.sub(r'[^A-Z0-9]', '', text)

                if len(text) >= 4 and score > best_score:
                    if self.license_complies_format(text):
                        formatted_text = self.format_license(text)
                        if score > best_score:
                            best_text = formatted_text
                            best_score = score

            return best_text, best_score

        except Exception as e:
            print(f"OCR error: {e}")
            return None, None

    def get_car(self, license_plate, vehicle_track_ids):
        """Retrieve the vehicle coordinates and ID based on the license plate coordinates"""
        try:
            if len(license_plate) < 6:
                return -1, -1, -1, -1, -1
                
            x1, y1, x2, y2, score, class_id = license_plate

            # Calculate license plate center
            plate_cx = (x1 + x2) / 2
            plate_cy = (y1 + y2) / 2

            best_car = None
            best_overlap = 0

            for j in range(len(vehicle_track_ids)):
                if len(vehicle_track_ids[j]) < 5:
                    continue
                    
                xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

                # Check if plate center is inside vehicle bbox
                if xcar1 < plate_cx < xcar2 and ycar1 < plate_cy < ycar2:
                    # Calculate overlap ratio
                    intersection_x1 = max(x1, xcar1)
                    intersection_y1 = max(y1, ycar1)
                    intersection_x2 = min(x2, xcar2)
                    intersection_y2 = min(y2, ycar2)
                    
                    if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                        plate_area = (x2 - x1) * (y2 - y1)
                        overlap_ratio = intersection_area / plate_area if plate_area > 0 else 0
                        
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_car = vehicle_track_ids[j]

            if best_car is not None and best_overlap > 0.5:  # At least 50% overlap
                return best_car

            return -1, -1, -1, -1, -1
        except Exception as e:
            print(f"get_car error: {e}")
            return -1, -1, -1, -1, -1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Traffic Monitoring System with Enhanced Detection")

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
        """Initialize traffic monitoring system with enhanced tracking"""
        # Load models
        self.coco_model = YOLO(vehicle_model_path)
        
        try:
            self.license_plate_detector = YOLO(plate_model_path)
            logger.info("License plate detector model loaded")
        except Exception as e:
            logger.warning(f"Failed to load license plate detector: {e}. Using vehicle model for plates.")
            self.license_plate_detector = YOLO(vehicle_model_path)
        
        # Initialize enhanced tracker
        self.mot_tracker = EnhancedSORTTracker(max_disappeared=50, max_distance=80, min_hits=3)
        
        # Initialize enhanced license plate processor
        self.plate_processor = EnhancedLicensePlateProcessor()
        
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
        self.counted_vehicles = set()  # Track which vehicles we've already counted
        self.detected_plates = {}
        self.plate_detections = defaultdict(list)  # Store all detections for each plate
        self.vehicle_speeds_by_plate = defaultdict(list)
        self.results = {}
        
        # Reset tracker
        self.mot_tracker = EnhancedSORTTracker(max_disappeared=50, max_distance=80, min_hits=3)
        
        logger.info("Traffic statistics reset")
        
    def process_frame(self, frame):
        """Process a single frame with enhanced tracking"""
        self.frame_count += 1
        frame_nmr = self.frame_count - 1
        
        self.results[frame_nmr] = {}
        
        try:
            # Detect vehicles with confidence threshold
            detections = self.coco_model(frame)[0]
            detections_ = []
            
            for detection in detections.boxes.data.tolist():
                if len(detection) >= 6:
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in self.vehicles and score > 0.5:  # Confidence threshold
                        detections_.append([x1, y1, x2, y2, score])

            # Track vehicles using enhanced SORT
            track_ids = self.mot_tracker.update(np.asarray(detections_))
            
            # Update current vehicles and calculate speeds
            current_ids = set()
            for track in track_ids:
                if len(track) >= 5:
                    xcar1, ycar1, xcar2, ycar2, car_id = track
                    current_ids.add(int(car_id))
                    
                    # Calculate speed
                    speed = self.mot_tracker.get_speed(int(car_id), self.pixel_to_meter, self.fps)
                    if speed > 0:  # Only add non-zero speeds
                        self.vehicle_speeds[int(car_id)].append(speed)
                    
                    # Count new vehicles only once
                    if int(car_id) not in self.counted_vehicles:
                        # Find the class of this vehicle from original detections
                        vehicle_class = self._find_vehicle_class(track, detections.boxes.data.tolist())
                        class_name = self.coco_model.names[vehicle_class] if vehicle_class != -1 else 'vehicle'
                        self.vehicle_count[class_name] += 1
                        self.counted_vehicles.add(int(car_id))

            self.current_vehicles = current_ids
            self.traffic_flow.append(len(current_ids))

            # Detect license plates with enhanced detection
            try:
                license_plates = self.license_plate_detector(frame)[0]
                
                for license_plate in license_plates.boxes.data.tolist():
                    if len(license_plate) >= 6:
                        x1, y1, x2, y2, score, class_id = license_plate
                        
                        if score < 0.4:  # Skip low confidence detections
                            continue

                        # Assign license plate to car
                        result = self.plate_processor.get_car(license_plate, track_ids)
                        if len(result) >= 5:
                            xcar1, ycar1, xcar2, ycar2, car_id = result

                            if car_id != -1:
                                # Crop license plate with padding
                                padding = 5
                                y1_int = max(0, int(y1) - padding)
                                y2_int = min(frame.shape[0], int(y2) + padding)
                                x1_int = max(0, int(x1) - padding)
                                x2_int = min(frame.shape[1], int(x2) + padding)
                                
                                if y2_int > y1_int and x2_int > x1_int:
                                    license_plate_crop = frame[y1_int:y2_int, x1_int:x2_int, :]

                                    if license_plate_crop.size > 0:
                                        # Read license plate number
                                        license_plate_text, license_plate_text_score = self.plate_processor.read_license_plate(license_plate_crop)

                                        if license_plate_text is not None and license_plate_text_score is not None and license_plate_text_score > 0.5:
                                            self.results[frame_nmr][car_id] = {
                                                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {
                                                    'bbox': [x1, y1, x2, y2],
                                                    'text': license_plate_text,
                                                    'bbox_score': score,
                                                    'text_score': license_plate_text_score
                                                }
                                            }
                                            
                                            # Calculate speeds
                                            vehicle_speeds = self.vehicle_speeds.get(int(car_id), [])
                                            if vehicle_speeds:
                                                avg_speed = np.mean(vehicle_speeds)
                                                max_speed = np.max(vehicle_speeds)
                                                current_speed = self.mot_tracker.get_speed(int(car_id), self.pixel_to_meter, self.fps)
                                            else:
                                                avg_speed = max_speed = current_speed = 0.0
                                            
                                            # Store detection
                                            self.plate_detections[license_plate_text].append({
                                                'confidence': float(license_plate_text_score),
                                                'frame': frame_nmr,
                                                'speed': current_speed
                                            })
                                            
                                            # Update or create plate entry
                                            if license_plate_text not in self.detected_plates:
                                                self.detected_plates[license_plate_text] = {
                                                    'text': license_plate_text,
                                                    'vehicle_id': int(car_id),
                                                    'confidence': float(license_plate_text_score),
                                                    'bbox': [x1, y1, x2, y2],
                                                    'vehicle_class': self._get_vehicle_class_name(car_id, track_ids, detections.boxes.data.tolist()),
                                                    'average_speed': float(avg_speed),
                                                    'max_speed': float(max_speed),
                                                    'current_speed': float(current_speed),
                                                    'first_seen': datetime.now().isoformat(),
                                                    'last_seen': datetime.now().isoformat(),
                                                    'detection_count': 1
                                                }
                                            else:
                                                # Update existing entry
                                                existing = self.detected_plates[license_plate_text]
                                                existing['last_seen'] = datetime.now().isoformat()
                                                existing['detection_count'] += 1
                                                existing['confidence'] = max(existing['confidence'], float(license_plate_text_score))
                                                existing['average_speed'] = float(avg_speed)
                                                existing['max_speed'] = max(existing['max_speed'], float(max_speed))
                                                existing['current_speed'] = float(current_speed)
                                            
                                            # Track speeds by plate
                                            if current_speed > 0:
                                                self.vehicle_speeds_by_plate[license_plate_text].append(current_speed)
                                            
                                            logger.info(f"Detected plate: {license_plate_text} for vehicle {car_id}, speed: {current_speed:.1f} km/h, confidence: {license_plate_text_score:.2f}")
            except Exception as e:
                logger.warning(f"License plate detection error: {e}")

        except Exception as e:
            logger.error(f"Frame processing error: {e}")

        # Draw annotations
        annotated_frame = self._draw_annotations(frame, track_ids, frame_nmr)
        
        # Get statistics
        stats = self._get_statistics()
        
        return annotated_frame, stats

    def _find_vehicle_class(self, track, original_detections):
        """Find the class of a tracked vehicle from original detections"""
        if len(track) < 5:
            return -1
            
        xcar1, ycar1, xcar2, ycar2, car_id = track
        
        # Find the detection that best matches this track
        best_iou = 0
        best_class = -1
        
        for detection in original_detections:
            if len(detection) >= 6:
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
            if len(track) >= 5 and track[4] == car_id:
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
            if len(track) >= 5:
                xcar1, ycar1, xcar2, ycar2, car_id = track
                x1, y1, x2, y2 = map(int, [xcar1, ycar1, xcar2, ycar2])
                
                # Get color for this track
                color = self._get_color(int(car_id))
                
                # Calculate current speed
                current_speed = self.mot_tracker.get_speed(int(car_id), self.pixel_to_meter, self.fps)
                
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
                
                # Draw vehicle label with speed
                label = f"ID:{int(car_id)}"
                if plate_text:
                    label += f" [{plate_text}]"
                label += f" {current_speed:.1f}km/h"
                
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
        cv2.rectangle(overlay, (10, 10), (350, 280), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw statistics
        y_offset = 30
        cv2.putText(frame, "Traffic Statistics", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        total_vehicles = sum(self.vehicle_count.values())
        cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
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
        unique_plates = len(self.detected_plates)
        text = f"Unique Plates: {unique_plates}"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Average speed
        y_offset += 20
        all_speeds = []
        for speeds in self.vehicle_speeds.values():
            all_speeds.extend([s for s in speeds if s > 0])
        avg_speed = np.mean(all_speeds) if all_speeds else 0
        text = f"Avg speed: {avg_speed:.1f} km/h"
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
        
        # Calculate overall average speed
        all_speeds = []
        for speeds in self.vehicle_speeds.values():
            all_speeds.extend([s for s in speeds if s > 0])
        avg_speed = np.mean(all_speeds) if all_speeds else 0.0
        
        # Calculate progress percentage
        progress = (self.frame_count / self.total_frames * 100) if self.total_frames > 0 else 0
        
        # Calculate average speeds for each plate
        plate_average_speeds = {}
        for plate, speeds in self.vehicle_speeds_by_plate.items():
            if speeds:
                plate_average_speeds[plate] = {
                    'average': float(np.mean(speeds)),
                    'max': float(np.max(speeds)),
                    'min': float(np.min([s for s in speeds if s > 0]) if any(s > 0 for s in speeds) else 0)
                }
        
        return {
            'vehicle_count': dict(self.vehicle_count),
            'current_vehicles': len(self.current_vehicles),
            'average_flow': float(avg_flow),
            'average_speed': float(avg_speed),
            'detected_plates': self.detected_plates,
            'plates_count': len(self.detected_plates),
            'timestamp': datetime.now().isoformat(),
            'frame_count': self.frame_count,
            'total_frames': self.total_frames,
            'progress_percentage': float(progress),
            'plate_average_speeds': plate_average_speeds
        }

    def get_all_detected_plates(self):
        """Get all detected license plates with detailed information"""
        plates_data = []
        
        for plate_text, plate_info in self.detected_plates.items():
            # Get all speeds for this plate
            speeds = self.vehicle_speeds_by_plate.get(plate_text, [])
            valid_speeds = [s for s in speeds if s > 0]
            
            if valid_speeds:
                avg_speed = float(np.mean(valid_speeds))
                max_speed = float(np.max(valid_speeds))
                min_speed = float(np.min(valid_speeds))
            else:
                avg_speed = max_speed = min_speed = 0.0
            
            # Get best confidence from all detections
            all_confidences = [d['confidence'] for d in self.plate_detections.get(plate_text, [])]
            best_confidence = max(all_confidences) if all_confidences else plate_info.get('confidence', 0.0)
            
            plates_data.append({
                'license_plate': plate_text,
                'vehicle_id': plate_info.get('vehicle_id', -1),
                'vehicle_class': plate_info.get('vehicle_class', 'vehicle'),
                'confidence': float(best_confidence),
                'average_speed': avg_speed,
                'max_speed': max_speed,
                'min_speed': min_speed,
                'current_speed': plate_info.get('current_speed', 0.0),
                'detection_count': plate_info.get('detection_count', 0),
                'first_seen': plate_info.get('first_seen', ''),
                'last_seen': plate_info.get('last_seen', '')
            })
        
        return sorted(plates_data, key=lambda x: x['average_speed'], reverse=True)

# Global traffic monitor instance
traffic_monitor = None

@app.on_event("startup")
async def startup_event():
    """Initialize traffic monitor on startup"""
    global traffic_monitor
    try:
        traffic_monitor = TrafficMonitor()
        logger.info("Traffic monitoring system initialized with enhanced tracking and license plate detection")
    except Exception as e:
        logger.warning(f"Failed to load models: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Traffic Monitoring System with Enhanced Tracking and License Plate Detection", "status": "active"}

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
    """Get all detected license plates with detailed information"""
    if traffic_monitor is None:
        raise HTTPException(status_code=503, detail="Traffic monitor not initialized")
    
    plates_data = traffic_monitor.get_all_detected_plates()
    return {
        "detected_plates": plates_data,
        "total_plates": len(plates_data),
        "timestamp": datetime.now().isoformat()
    }

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
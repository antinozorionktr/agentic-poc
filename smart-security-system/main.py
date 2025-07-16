import cv2
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional
import json
import os
from pathlib import Path
import threading
import time
from collections import deque
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSORTTracker:
    """Simple tracking implementation for person detection continuity"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """Register a new object with the next available ID"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        """Deregister an object ID"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, rects):
        """Update tracker with new detections"""
        if len(rects) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Initialize array of input centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col])
        
        return self.objects

class BehaviorAnalyzer:
    """Analyze person behavior for security assessment"""
    
    def __init__(self):
        self.person_tracks = {}
        self.loitering_threshold = 30  # seconds
        self.speed_threshold = 50  # pixels per second
        
    def analyze_behavior(self, person_id: int, bbox: Tuple[int, int, int, int], timestamp: datetime) -> Dict[str, any]:
        """Analyze behavior patterns for a tracked person"""
        x1, y1, x2, y2 = bbox
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        if person_id not in self.person_tracks:
            self.person_tracks[person_id] = {
                'positions': deque(maxlen=50),
                'timestamps': deque(maxlen=50),
                'first_seen': timestamp,
                'behaviors': []
            }
        
        track = self.person_tracks[person_id]
        track['positions'].append(centroid)
        track['timestamps'].append(timestamp)
        
        behaviors = []
        
        # Calculate duration
        duration = (timestamp - track['first_seen']).total_seconds()
        
        # Check for loitering
        if len(track['positions']) > 1:
            if duration > self.loitering_threshold:
                # Check if person has moved significantly
                first_pos = track['positions'][0]
                current_pos = track['positions'][-1]
                distance = math.sqrt((current_pos[0] - first_pos[0])**2 + (current_pos[1] - first_pos[1])**2)
                
                if distance < 100:  # Less than 100 pixels movement
                    behaviors.append("loitering")
        
        # Check for unusual speed
        if len(track['positions']) > 5:
            recent_positions = list(track['positions'])[-5:]
            recent_timestamps = list(track['timestamps'])[-5:]
            
            total_distance = 0
            for i in range(1, len(recent_positions)):
                pos1, pos2 = recent_positions[i-1], recent_positions[i]
                distance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                total_distance += distance
            
            time_diff = (recent_timestamps[-1] - recent_timestamps[0]).total_seconds()
            if time_diff > 0:
                speed = total_distance / time_diff
                if speed > self.speed_threshold:
                    behaviors.append("fast_movement")
        
        # Check for restricted area (example: top-left corner)
        if centroid[0] < 100 and centroid[1] < 100:
            behaviors.append("restricted_area")
        
        track['behaviors'] = behaviors
        
        return {
            'behaviors': behaviors,
            'duration': duration,
            'position_history': list(track['positions']),
            'risk_level': self._calculate_risk_level(behaviors, duration)
        }
    
    def _calculate_risk_level(self, behaviors: List[str], duration: float) -> str:
        """Calculate risk level based on behaviors"""
        risk_score = 0
        
        if "loitering" in behaviors:
            risk_score += 3
        if "fast_movement" in behaviors:
            risk_score += 2
        if "restricted_area" in behaviors:
            risk_score += 4
        
        if duration > 60:  # More than 1 minute
            risk_score += 1
        
        if risk_score >= 5:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"

class AlertSystem:
    """Manage security alerts and notifications"""
    
    def __init__(self):
        self.alerts = []
        self.alert_cooldown = {}
        self.cooldown_period = 30  # seconds
    
    def create_alert(self, alert_type: str, message: str, severity: str = "medium", 
                    location: str = "Unknown", metadata: Dict = None) -> Dict:
        """Create a new security alert"""
        
        # Check cooldown to prevent spam
        current_time = datetime.now()
        cooldown_key = f"{alert_type}_{location}"
        
        if cooldown_key in self.alert_cooldown:
            last_alert_time = self.alert_cooldown[cooldown_key]
            if (current_time - last_alert_time).total_seconds() < self.cooldown_period:
                return None
        
        alert = {
            'id': len(self.alerts),
            'timestamp': current_time,
            'type': alert_type,
            'message': message,
            'severity': severity,
            'location': location,
            'metadata': metadata or {},
            'status': 'active'
        }
        
        self.alerts.append(alert)
        self.alert_cooldown[cooldown_key] = current_time
        
        logger.warning(f"SECURITY ALERT: {alert_type} - {message}")
        
        return alert
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        return [alert for alert in self.alerts if alert['status'] == 'active']
    
    def resolve_alert(self, alert_id: int):
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'resolved'
                break

class SecuritySystem:
    """Main security system class integrating all components"""
    
    def __init__(self, model_path: str = None, device: str = None):
        """Initialize the security system"""
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing SecuritySystem on device: {self.device}")
        
        # Load YOLO model
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                # Use pre-trained YOLOv8 model
                self.model = YOLO('yolov8n.pt')  # nano version for speed
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Initialize components
        self.tracker = DeepSORTTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.alert_system = AlertSystem()
        
        # Configuration
        self.config = {
            'confidence_threshold': 0.5,
            'iou_threshold': 0.4,
            'alert_sensitivity': 'medium',
            'enable_notifications': True,
            'person_class_id': 0  # COCO person class
        }
        
        # Runtime statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'alerts_generated': 0,
            'start_time': datetime.now()
        }
    
    def update_settings(self, new_settings: Dict):
        """Update system configuration"""
        self.config.update(new_settings)
        logger.info(f"Settings updated: {new_settings}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame for person detection and analysis"""
        
        start_time = time.time()
        
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.config['confidence_threshold'], 
                               iou=self.config['iou_threshold'], verbose=False)
            
            detections = []
            person_boxes = []
            
            # Process detection results
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Filter for person class only
                    if int(box.cls) == self.config['person_class_id']:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        
                        person_boxes.append([x1, y1, x2, y2])
                        
                        detection = {
                            'class': 'person',
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'timestamp': datetime.now()
                        }
                        detections.append(detection)
            
            # Update tracker
            tracked_objects = self.tracker.update(person_boxes)
            
            # Analyze behavior for tracked persons
            current_time = datetime.now()
            for person_id, centroid in tracked_objects.items():
                # Find corresponding detection
                for detection in detections:
                    bbox = detection['bbox']
                    det_centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    
                    # Check if this detection matches the tracked object
                    distance = math.sqrt((centroid[0] - det_centroid[0])**2 + 
                                       (centroid[1] - det_centroid[1])**2)
                    
                    if distance < 50:  # Threshold for matching
                        try:
                            # Analyze behavior
                            behavior_analysis = self.behavior_analyzer.analyze_behavior(
                                person_id, tuple(bbox), current_time
                            )
                            
                            detection['person_id'] = person_id
                            detection['behavior_analysis'] = behavior_analysis
                            
                            # Generate alerts based on behavior
                            self._check_and_generate_alerts(person_id, behavior_analysis, bbox)
                        except Exception as e:
                            logger.error(f"Error in behavior analysis: {e}")
                            # Continue processing without behavior analysis
                            detection['person_id'] = person_id
                            detection['behavior_analysis'] = {'behaviors': [], 'risk_level': 'low'}
                        break
            
            # Draw detections and tracking on frame
            processed_frame = self._draw_detections(frame.copy(), detections, tracked_objects)
            
            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['total_detections'] += len(detections)
            
            processing_time = time.time() - start_time
            logger.debug(f"Frame processed in {processing_time:.3f}s, {len(detections)} persons detected")
            
            return processed_frame, detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            # Return original frame with error message
            error_frame = frame.copy()
            cv2.putText(error_frame, f"Processing Error: {str(e)[:50]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return error_frame, []
    
    def _check_and_generate_alerts(self, person_id: int, behavior_analysis: Dict, bbox: List[int]):
        """Check behavior analysis and generate appropriate alerts"""
        
        behaviors = behavior_analysis.get('behaviors', [])
        risk_level = behavior_analysis.get('risk_level', 'low')
        duration = behavior_analysis.get('duration', 0)
        
        # Alert for loitering
        if 'loitering' in behaviors:
            self.alert_system.create_alert(
                'Loitering Detected',
                f'Person {person_id} has been loitering for {duration:.1f} seconds',
                'medium',
                f'Area: {bbox[0]}, {bbox[1]}',
                {'person_id': person_id, 'duration': duration}
            )
        
        # Alert for restricted area
        if 'restricted_area' in behaviors:
            self.alert_system.create_alert(
                'Restricted Area Access',
                f'Person {person_id} detected in restricted area',
                'high',
                f'Restricted Zone',
                {'person_id': person_id, 'bbox': bbox}
            )
        
        # Alert for fast movement
        if 'fast_movement' in behaviors:
            self.alert_system.create_alert(
                'Suspicious Movement',
                f'Person {person_id} moving at unusual speed',
                'medium',
                f'Area: {bbox[0]}, {bbox[1]}',
                {'person_id': person_id}
            )
        
        # High risk alert
        if risk_level == 'high':
            self.alert_system.create_alert(
                'High Risk Detection',
                f'Person {person_id} classified as high risk',
                'high',
                f'Area: {bbox[0]}, {bbox[1]}',
                {'person_id': person_id, 'risk_level': risk_level}
            )
        
        self.stats['alerts_generated'] = len(self.alert_system.alerts)
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                        tracked_objects: Dict) -> np.ndarray:
        """Draw detection boxes and information on frame"""
        
        # Color scheme
        colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 255, 255),  # Yellow
            'high': (0, 0, 255)       # Red
        }
        
        # Draw detection boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            person_id = detection.get('person_id', 'Unknown')
            
            # Get risk level color
            behavior_analysis = detection.get('behavior_analysis', {})
            risk_level = behavior_analysis.get('risk_level', 'low')
            color = colors.get(risk_level, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f'Person {person_id}: {confidence:.2f}'
            if behavior_analysis:
                behaviors = behavior_analysis.get('behaviors', [])
                if behaviors:
                    label += f' ({", ".join(behaviors)})'
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw risk level indicator
            risk_text = f'Risk: {risk_level.upper()}'
            cv2.putText(frame, risk_text, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tracking trails
        for person_id, centroid in tracked_objects.items():
            if person_id in self.behavior_analyzer.person_tracks:
                positions = list(self.behavior_analyzer.person_tracks[person_id]['positions'])
                
                # Draw trail
                for i in range(1, len(positions)):
                    cv2.line(frame, positions[i-1], positions[i], (128, 128, 128), 2)
                
                # Draw current position
                cv2.circle(frame, centroid, 5, (255, 0, 255), -1)
        
        # Draw system info
        self._draw_system_info(frame)
        
        return frame
    
    def _draw_system_info(self, frame: np.ndarray):
        """Draw system information overlay"""
        
        height, width = frame.shape[:2]
        
        # System status
        uptime = datetime.now() - self.stats['start_time']
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        info_lines = [
            f"Security System - Status: ACTIVE",
            f"Uptime: {uptime_str}",
            f"Frames: {self.stats['frames_processed']}",
            f"Detections: {self.stats['total_detections']}",
            f"Alerts: {self.stats['alerts_generated']}",
            f"Device: {self.device.upper()}"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - 300, 10), (width - 10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 20
            cv2.putText(frame, line, (width - 290, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw active alerts indicator
        active_alerts = len(self.alert_system.get_active_alerts())
        if active_alerts > 0:
            alert_text = f"ACTIVE ALERTS: {active_alerts}"
            cv2.putText(frame, alert_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def process_video_file(self, video_path: str, output_path: str = None) -> Dict:
        """Process entire video file"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        frame_count = 0
        detection_summary = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Save frame if writer is available
                if writer:
                    writer.write(processed_frame)
                
                # Store detection summary
                if detections:
                    detection_summary.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'detections': len(detections),
                        'persons': [d for d in detections]
                    })
                
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        # Compile results
        results = {
            'input_file': video_path,
            'output_file': output_path,
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'fps': fps,
            'detection_summary': detection_summary,
            'total_detections': sum(len(d['persons']) for d in detection_summary),
            'alerts_generated': len(self.alert_system.alerts),
            'processing_stats': self.stats
        }
        
        logger.info(f"Video processing complete: {frame_count} frames processed")
        logger.info(f"Total detections: {results['total_detections']}")
        logger.info(f"Alerts generated: {results['alerts_generated']}")
        
        return results
    
    def get_system_status(self) -> Dict:
        """Get current system status and statistics"""
        
        uptime = datetime.now() - self.stats['start_time']
        active_alerts = self.alert_system.get_active_alerts()
        
        return {
            'status': 'active',
            'uptime': str(uptime).split('.')[0],
            'device': self.device,
            'configuration': self.config,
            'statistics': {
                **self.stats,
                'fps': self.stats['frames_processed'] / max(uptime.total_seconds(), 1),
                'alerts_active': len(active_alerts),
                'alerts_total': len(self.alert_system.alerts)
            },
            'active_alerts': active_alerts,
            'tracked_persons': len(self.behavior_analyzer.person_tracks)
        }
    
    def save_configuration(self, config_path: str):
        """Save current configuration to file"""
        config_data = {
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
    
    def load_configuration(self, config_path: str):
        """Load configuration from file"""
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.config.update(config_data.get('config', {}))
            logger.info(f"Configuration loaded from: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

def main():
    """Main function for testing the security system"""
    
    # Initialize security system
    security_system = SecuritySystem()
    
    print("Smart Security System Initialized")
    print("=" * 50)
    print(f"Device: {security_system.device}")
    print(f"Model: {security_system.model.model}")
    print("=" * 50)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting live monitoring... Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, detections = security_system.process_frame(frame)
            
            # Display result
            cv2.imshow('Smart Security System', processed_frame)
            
            # Print detection info
            if detections:
                print(f"Frame: {security_system.stats['frames_processed']} - "
                      f"Persons detected: {len(detections)}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping security system...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        status = security_system.get_system_status()
        print("\nFinal Statistics:")
        print(f"Uptime: {status['uptime']}")
        print(f"Frames processed: {status['statistics']['frames_processed']}")
        print(f"Total detections: {status['statistics']['total_detections']}")
        print(f"Alerts generated: {status['statistics']['alerts_total']}")

if __name__ == "__main__":
    main()
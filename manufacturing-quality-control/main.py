import cv2
import numpy as np
from ultralytics import YOLO
import mlflow
import mlflow.pytorch
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import torch
from dataclasses import dataclass
from collections import defaultdict
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DefectInfo:
    """Store defect information"""
    defect_type: str
    confidence: float
    bbox: List[int]
    area: float
    timestamp: datetime

class QualityControlSystem:
    """Manufacturing Quality Control System using YOLOv8"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the quality control system
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.mlflow_run_id = None
        self.defect_classes = {
            0: "scratch",
            1: "dent", 
            2: "stain",
            3: "crack",
            4: "hole",
            5: "contamination",
            6: "misalignment",
            7: "missing_component"
        }
        self.quality_thresholds = {
            "excellent": 0,
            "good": 2,
            "acceptable": 5,
            "reject": 10
        }
        self.load_model()
        self.setup_mlflow()
        
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Use pretrained model as fallback
            self.model = YOLO("yolov8n.pt")
            logger.info("Using pretrained YOLOv8n model")
            
    def setup_mlflow(self):
        """Setup MLflow for experiment tracking"""
        try:
            mlflow.set_experiment("manufacturing_quality_control")
            self.mlflow_run_id = mlflow.start_run().info.run_id
            mlflow.log_params({
                "model": self.model_path,
                "confidence_threshold": self.confidence_threshold,
                "defect_classes": len(self.defect_classes)
            })
            logger.info("MLflow setup completed")
        except Exception as e:
            logger.error(f"MLflow setup error: {e}")
            
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better defect detection
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Apply CLAHE for better contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return denoised
        
    def detect_anomalies(self, image: np.ndarray) -> Tuple[List[DefectInfo], np.ndarray]:
        """
        Detect anomalies/defects in the image
        
        Args:
            image: Input image
            
        Returns:
            List of detected defects and annotated image
        """
        preprocessed = self.preprocess_image(image)
        results = self.model(preprocessed, conf=self.confidence_threshold)
        
        defects = []
        annotated_image = image.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # Map to defect class
                    defect_type = self.defect_classes.get(cls % len(self.defect_classes), "unknown")
                    
                    # Calculate defect area
                    area = (x2 - x1) * (y2 - y1)
                    
                    defect = DefectInfo(
                        defect_type=defect_type,
                        confidence=conf,
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        area=area,
                        timestamp=datetime.now()
                    )
                    defects.append(defect)
                    
                    # Draw bounding box
                    color = self._get_defect_color(defect_type)
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Add label
                    label = f"{defect_type}: {conf:.2f}"
                    cv2.putText(annotated_image, label, (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return defects, annotated_image
        
    def _get_defect_color(self, defect_type: str) -> Tuple[int, int, int]:
        """Get color for defect type visualization"""
        colors = {
            "scratch": (0, 0, 255),      # Red
            "dent": (0, 165, 255),       # Orange
            "stain": (0, 255, 255),      # Yellow
            "crack": (255, 0, 0),        # Blue
            "hole": (128, 0, 128),       # Purple
            "contamination": (0, 255, 0), # Green
            "misalignment": (255, 192, 203), # Pink
            "missing_component": (128, 128, 128) # Gray
        }
        return colors.get(defect_type, (255, 255, 255))
        
    def calculate_quality_score(self, defects: List[DefectInfo]) -> Dict:
        """
        Calculate quality score based on detected defects
        
        Args:
            defects: List of detected defects
            
        Returns:
            Quality metrics dictionary
        """
        if not defects:
            return {
                "score": 100.0,
                "grade": "excellent",
                "defect_count": 0,
                "defect_summary": {},
                "recommendations": ["Product meets quality standards"]
            }
            
        # Count defects by type
        defect_counts = defaultdict(int)
        total_severity = 0
        
        severity_weights = {
            "crack": 5.0,
            "hole": 4.5,
            "missing_component": 4.0,
            "misalignment": 3.0,
            "dent": 2.5,
            "contamination": 2.0,
            "scratch": 1.5,
            "stain": 1.0
        }
        
        for defect in defects:
            defect_counts[defect.defect_type] += 1
            weight = severity_weights.get(defect.defect_type, 1.0)
            total_severity += weight * defect.confidence
            
        # Calculate score (0-100)
        score = max(0, 100 - (total_severity * 10))
        
        # Determine grade
        defect_count = len(defects)
        if defect_count <= self.quality_thresholds["excellent"]:
            grade = "excellent"
        elif defect_count <= self.quality_thresholds["good"]:
            grade = "good"
        elif defect_count <= self.quality_thresholds["acceptable"]:
            grade = "acceptable"
        else:
            grade = "reject"
            
        # Generate recommendations
        recommendations = []
        if "crack" in defect_counts or "hole" in defect_counts:
            recommendations.append("Critical defects detected - immediate inspection required")
        if "contamination" in defect_counts:
            recommendations.append("Clean production environment")
        if "misalignment" in defect_counts:
            recommendations.append("Check equipment calibration")
        if defect_count > 5:
            recommendations.append("Review manufacturing process")
            
        return {
            "score": round(score, 2),
            "grade": grade,
            "defect_count": defect_count,
            "defect_summary": dict(defect_counts),
            "recommendations": recommendations
        }
        
    def process_batch(self, image_paths: List[str]) -> Dict:
        """
        Process a batch of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Batch processing results
        """
        batch_results = {
            "total_images": len(image_paths),
            "processed": 0,
            "passed": 0,
            "failed": 0,
            "defect_distribution": defaultdict(int),
            "average_score": 0,
            "processing_time": 0,
            "individual_results": []
        }
        
        start_time = datetime.now()
        total_score = 0
        
        for img_path in image_paths:
            try:
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue
                    
                defects, annotated = self.detect_anomalies(image)
                quality = self.calculate_quality_score(defects)
                
                # Update batch statistics
                batch_results["processed"] += 1
                if quality["grade"] in ["excellent", "good", "acceptable"]:
                    batch_results["passed"] += 1
                else:
                    batch_results["failed"] += 1
                    
                total_score += quality["score"]
                
                for defect_type, count in quality["defect_summary"].items():
                    batch_results["defect_distribution"][defect_type] += count
                    
                batch_results["individual_results"].append({
                    "image": img_path,
                    "quality": quality,
                    "defects": len(defects)
                })
                
                # Log to MLflow
                if self.mlflow_run_id:
                    mlflow.log_metric(f"quality_score_{Path(img_path).stem}", quality["score"])
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                
        # Calculate final statistics
        if batch_results["processed"] > 0:
            batch_results["average_score"] = round(total_score / batch_results["processed"], 2)
        batch_results["processing_time"] = (datetime.now() - start_time).total_seconds()
        batch_results["defect_distribution"] = dict(batch_results["defect_distribution"])
        
        return batch_results
        
    def save_results(self, results: Dict, output_path: str = "quality_report.json"):
        """Save processing results to file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        logger.info(f"Results saved to {output_path}")
        
    def close(self):
        """Clean up resources"""
        if self.mlflow_run_id:
            mlflow.end_run()
            

# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    qc_system = QualityControlSystem(confidence_threshold=0.4)
    
    # Test with sample image
    test_image_path = "sample_product.jpg"
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        defects, annotated = qc_system.detect_anomalies(image)
        quality = qc_system.calculate_quality_score(defects)
        
        print(f"Quality Score: {quality['score']}")
        print(f"Grade: {quality['grade']}")
        print(f"Defects found: {quality['defect_count']}")
        print(f"Defect summary: {quality['defect_summary']}")
        print(f"Recommendations: {quality['recommendations']}")
        
        # Save annotated image
        cv2.imwrite("annotated_result.jpg", annotated)
    else:
        print(f"Test image not found: {test_image_path}")
        
    # Process batch example
    batch_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    existing_paths = [p for p in batch_paths if os.path.exists(p)]
    
    if existing_paths:
        batch_results = qc_system.process_batch(existing_paths)
        qc_system.save_results(batch_results)
        print(f"\nBatch Results:")
        print(f"Processed: {batch_results['processed']}")
        print(f"Passed: {batch_results['passed']}")
        print(f"Failed: {batch_results['failed']}")
        print(f"Average Score: {batch_results['average_score']}")
    
    qc_system.close()
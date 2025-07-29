import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import psycopg2
from ultralytics import YOLO
import rasterio
from rasterio.warp import transform_bounds
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Point, Polygon
import folium
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql://postgres:1234@localhost/agri_monitoring"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CropField(Base):
    __tablename__ = "crop_fields"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    geometry = Column(Text)
    crop_type = Column(String)
    area_hectares = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class CropAnalysis(Base):
    __tablename__ = "crop_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(Integer)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String)
    health_score = Column(Float)
    predicted_yield = Column(Float)
    disease_detected = Column(Boolean, default=False)
    disease_type = Column(String, nullable=True)
    segmentation_results = Column(Text)
    ndvi_avg = Column(Float)
    ndvi_std = Column(Float)

class FieldCreate(BaseModel):
    name: str
    geometry: str
    crop_type: str
    area_hectares: float

class AnalysisResult(BaseModel):
    field_id: int
    health_score: float
    predicted_yield: float
    disease_detected: bool
    disease_type: Optional[str]
    ndvi_avg: float
    recommendations: List[str]

class AgriculturalMonitor:
    def __init__(self, model_path: str = "yolov8n-seg.pt"):
        """Initialize the agricultural monitoring system"""
        self.model = YOLO(model_path)
        self.crop_classes = {
            0: 'background',
            1: 'crop',
            2: 'weed',
            3: 'soil',
            4: 'water',
            5: 'diseased_crop'
        }
        
        self.health_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
        
    def preprocess_satellite_image(self, image_path: str) -> np.ndarray:
        """Preprocess Sentinel-2 satellite imagery"""
        try:
            with rasterio.open(image_path) as src:
                red = src.read(4)
                green = src.read(3)
                blue = src.read(2)
                nir = src.read(8)
                
                rgb = np.stack([red, green, blue], axis=-1)
                
                rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
                
                ndvi = self.calculate_ndvi(red, nir)
                
                return rgb, ndvi
                
        except Exception as e:
            logger.error(f"Error preprocessing satellite image: {str(e)}")
            raise
    
    def calculate_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index"""
        # Avoid division by zero
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
        return np.clip(ndvi, -1, 1)
    
    def segment_crops(self, image: np.ndarray) -> Dict:
        """Perform crop segmentation using YOLOv8"""
        try:
            results = self.model(image)
            
            segmentation_data = {
                'masks': [],
                'classes': [],
                'confidence': [],
                'areas': []
            }
            
            for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for mask, cls, conf in zip(masks, classes, confidences):
                        segmentation_data['masks'].append(mask)
                        segmentation_data['classes'].append(int(cls))
                        segmentation_data['confidence'].append(float(conf))
                        segmentation_data['areas'].append(np.sum(mask))
            
            return segmentation_data
            
        except Exception as e:
            logger.error(f"Error in crop segmentation: {str(e)}")
            raise
    
    def assess_crop_health(self, image: np.ndarray, ndvi: np.ndarray, 
                          segmentation: Dict) -> Dict:
        """Assess crop health based on NDVI and segmentation results"""
        try:
            health_metrics = {
                'overall_health': 0.0,
                'ndvi_stats': {},
                'disease_probability': 0.0,
                'stress_indicators': []
            }
            
            # Calculate NDVI statistics
            valid_ndvi = ndvi[ndvi != 0]
            health_metrics['ndvi_stats'] = {
                'mean': float(np.mean(valid_ndvi)),
                'std': float(np.std(valid_ndvi)),
                'min': float(np.min(valid_ndvi)),
                'max': float(np.max(valid_ndvi))
            }
            
            ndvi_mean = health_metrics['ndvi_stats']['mean']
            if ndvi_mean > 0.6:
                health_score = 0.9
            elif ndvi_mean > 0.4:
                health_score = 0.7
            elif ndvi_mean > 0.2:
                health_score = 0.5
            else:
                health_score = 0.3
            
            diseased_pixels = 0
            total_crop_pixels = 0
            
            for mask, cls in zip(segmentation['masks'], segmentation['classes']):
                if cls == 5:  # Diseased crop class
                    diseased_pixels += np.sum(mask)
                elif cls == 1:  # Healthy crop class
                    total_crop_pixels += np.sum(mask)
            
            if total_crop_pixels > 0:
                disease_ratio = diseased_pixels / (diseased_pixels + total_crop_pixels)
                health_metrics['disease_probability'] = disease_ratio
                health_score *= (1 - disease_ratio * 0.5)  # Reduce health score
            
            health_metrics['overall_health'] = health_score
            
            if ndvi_mean < 0.2:
                health_metrics['stress_indicators'].append('Low vegetation index')
            if health_metrics['ndvi_stats']['std'] > 0.3:
                health_metrics['stress_indicators'].append('Uneven growth')
            if disease_ratio > 0.1:
                health_metrics['stress_indicators'].append('Disease detected')
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Error in health assessment: {str(e)}")
            raise
    
    def predict_yield(self, health_score: float, area_hectares: float, 
                     crop_type: str) -> float:
        """Predict crop yield based on health metrics and historical data"""
        base_yields = {
            'wheat': 3000,
            'corn': 9000,
            'rice': 4500,
            'soybean': 2700,
            'cotton': 800,
            'default': 3000
        }
        
        base_yield = base_yields.get(crop_type.lower(), base_yields['default'])
        
        predicted_yield = base_yield * health_score * area_hectares
        
        return predicted_yield
    
    def detect_diseases(self, image: np.ndarray, segmentation: Dict) -> Dict:
        """Detect crop diseases from segmentation results"""
        disease_info = {
            'detected': False,
            'type': None,
            'severity': 'low',
            'affected_area_pct': 0.0
        }
        
        total_pixels = image.shape[0] * image.shape[1]
        diseased_pixels = 0
        
        for mask, cls, conf in zip(segmentation['masks'], 
                                  segmentation['classes'], 
                                  segmentation['confidence']):
            if cls == 5 and conf > 0.5:  # Diseased crop with high confidence
                diseased_pixels += np.sum(mask)
        
        if diseased_pixels > 0:
            affected_pct = (diseased_pixels / total_pixels) * 100
            disease_info['detected'] = True
            disease_info['affected_area_pct'] = affected_pct
            
            # Determine disease type and severity (simplified)
            if affected_pct > 20:
                disease_info['severity'] = 'high'
                disease_info['type'] = 'severe_blight'
            elif affected_pct > 10:
                disease_info['severity'] = 'medium'
                disease_info['type'] = 'leaf_spot'
            else:
                disease_info['severity'] = 'low'
                disease_info['type'] = 'early_stress'
        
        return disease_info
    
    def generate_recommendations(self, health_metrics: Dict, 
                               disease_info: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        health_score = health_metrics['overall_health']
        ndvi_mean = health_metrics['ndvi_stats']['mean']
        
        # Health-based recommendations
        if health_score < 0.4:
            recommendations.append("Consider soil testing and nutrient supplementation")
            recommendations.append("Increase irrigation frequency")
        elif health_score < 0.6:
            recommendations.append("Monitor crop development closely")
            recommendations.append("Consider targeted fertilizer application")
        
        # NDVI-based recommendations
        if ndvi_mean < 0.3:
            recommendations.append("Investigate potential nutrient deficiencies")
            recommendations.append("Check irrigation system efficiency")
        
        # Disease-based recommendations
        if disease_info['detected']:
            severity = disease_info['severity']
            if severity == 'high':
                recommendations.append("Apply appropriate fungicide treatment immediately")
                recommendations.append("Consider quarantine measures for affected areas")
            elif severity == 'medium':
                recommendations.append("Monitor disease progression closely")
                recommendations.append("Apply preventive treatments to healthy areas")
            else:
                recommendations.append("Implement preventive disease management practices")
        
        # Stress indicator recommendations
        for indicator in health_metrics['stress_indicators']:
            if 'Low vegetation' in indicator:
                recommendations.append("Increase nitrogen fertilizer application")
            elif 'Uneven growth' in indicator:
                recommendations.append("Check field drainage and soil compaction")
        
        return list(set(recommendations))  # Remove duplicates

# FastAPI Application
app = FastAPI(title="Agricultural Monitoring Platform", version="1.0.0")

# Initialize the monitoring system
monitor = AgriculturalMonitor()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    """Create database tables"""
    Base.metadata.create_all(bind=engine)

@app.post("/fields/", response_model=dict)
async def create_field(field: FieldCreate, db: Session = Depends(get_db)):
    """Create a new crop field"""
    try:
        db_field = CropField(**field.dict())
        db.add(db_field)
        db.commit()
        db.refresh(db_field)
        return {"message": "Field created successfully", "field_id": db_field.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/", response_model=AnalysisResult)
async def analyze_field(
    field_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Analyze crop field from uploaded image"""
    try:
        # Save uploaded image
        image_path = f"uploads/{field_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        os.makedirs("uploads", exist_ok=True)
        
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # For satellite imagery, use preprocessing
        if image.filename.lower().endswith(('.tif', '.tiff')):
            img_rgb, ndvi = monitor.preprocess_satellite_image(image_path)
        else:
            # For drone imagery, calculate simple NDVI approximation
            # This is simplified - real NDVI needs NIR band
            ndvi = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
        
        # Perform segmentation
        segmentation = monitor.segment_crops(img_rgb)
        
        # Assess health
        health_metrics = monitor.assess_crop_health(img_rgb, ndvi, segmentation)
        
        # Detect diseases
        disease_info = monitor.detect_diseases(img_rgb, segmentation)
        
        # Get field info for yield prediction
        field = db.query(CropField).filter(CropField.id == field_id).first()
        if not field:
            raise HTTPException(status_code=404, detail="Field not found")
        
        # Predict yield
        predicted_yield = monitor.predict_yield(
            health_metrics['overall_health'],
            field.area_hectares,
            field.crop_type
        )
        
        # Generate recommendations
        recommendations = monitor.generate_recommendations(health_metrics, disease_info)
        
        # Save analysis to database
        analysis = CropAnalysis(
            field_id=field_id,
            image_path=image_path,
            health_score=health_metrics['overall_health'],
            predicted_yield=predicted_yield,
            disease_detected=disease_info['detected'],
            disease_type=disease_info.get('type'),
            ndvi_avg=health_metrics['ndvi_stats']['mean'],
            ndvi_std=health_metrics['ndvi_stats']['std'],
            segmentation_results=str(segmentation)
        )
        
        db.add(analysis)
        db.commit()
        
        return AnalysisResult(
            field_id=field_id,
            health_score=health_metrics['overall_health'],
            predicted_yield=predicted_yield,
            disease_detected=disease_info['detected'],
            disease_type=disease_info.get('type'),
            ndvi_avg=health_metrics['ndvi_stats']['mean'],
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in field analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fields/{field_id}/history")
async def get_field_history(field_id: int, db: Session = Depends(get_db)):
    """Get analysis history for a field"""
    try:
        analyses = db.query(CropAnalysis).filter(
            CropAnalysis.field_id == field_id
        ).order_by(CropAnalysis.analysis_date.desc()).all()
        
        return [
            {
                "date": analysis.analysis_date,
                "health_score": analysis.health_score,
                "predicted_yield": analysis.predicted_yield,
                "disease_detected": analysis.disease_detected,
                "ndvi_avg": analysis.ndvi_avg
            }
            for analysis in analyses
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/summary")
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get dashboard summary statistics"""
    try:
        # Get all fields
        fields = db.query(CropField).all()
        total_fields = len(fields)
        total_area = sum(field.area_hectares for field in fields)
        
        # Get recent analyses
        recent_analyses = db.query(CropAnalysis).order_by(
            CropAnalysis.analysis_date.desc()
        ).limit(10).all()
        
        avg_health = np.mean([a.health_score for a in recent_analyses]) if recent_analyses else 0
        total_yield = sum(a.predicted_yield for a in recent_analyses)
        disease_alerts = sum(1 for a in recent_analyses if a.disease_detected)
        
        return {
            "total_fields": total_fields,
            "total_area_hectares": total_area,
            "average_health_score": avg_health,
            "total_predicted_yield": total_yield,
            "disease_alerts": disease_alerts,
            "recent_analyses": len(recent_analyses)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Data Processing Utilities
class DataProcessor:
    """Utility class for processing Agriculture-Vision dataset"""
    
    @staticmethod
    def prepare_training_data(dataset_path: str, output_path: str):
        """Prepare Agriculture-Vision dataset for YOLOv8 training"""
        import shutil
        from sklearn.model_selection import train_test_split
        
        # Create YOLO format directory structure
        for split in ['train', 'val']:
            os.makedirs(f"{output_path}/{split}/images", exist_ok=True)
            os.makedirs(f"{output_path}/{split}/labels", exist_ok=True)
        
        # Process images and masks
        image_dir = os.path.join(dataset_path, "images")
        mask_dir = os.path.join(dataset_path, "masks")
        
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
        
        for split, files in [('train', train_files), ('val', val_files)]:
            for filename in files:
                # Copy image
                src_img = os.path.join(image_dir, filename)
                dst_img = f"{output_path}/{split}/images/{filename}"
                shutil.copy2(src_img, dst_img)
                
                # Convert mask to YOLO format
                mask_file = filename.replace('.jpg', '.png')
                src_mask = os.path.join(mask_dir, mask_file)
                
                if os.path.exists(src_mask):
                    DataProcessor.convert_mask_to_yolo(
                        src_mask, 
                        f"{output_path}/{split}/labels/{filename.replace('.jpg', '.txt')}"
                    )
    
    @staticmethod
    def convert_mask_to_yolo(mask_path: str, output_path: str):
        """Convert segmentation mask to YOLO format"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape
        
        # Find contours for each class
        unique_classes = np.unique(mask)
        
        with open(output_path, 'w') as f:
            for class_id in unique_classes:
                if class_id == 0:  # Skip background
                    continue
                
                # Create binary mask for this class
                binary_mask = (mask == class_id).astype(np.uint8)
                
                # Find contours
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    # Simplify contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Convert to normalized coordinates
                    points = []
                    for point in approx:
                        x, y = point[0]
                        points.extend([x/width, y/height])
                    
                    # Write to file
                    if len(points) >= 6:  # At least 3 points
                        f.write(f"{class_id-1} {' '.join(map(str, points))}\n")

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

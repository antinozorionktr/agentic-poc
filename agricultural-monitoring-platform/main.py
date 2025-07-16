from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import io
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
import base64
import json
from typing import List, Dict, Optional
import logging
from pydantic import BaseModel
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agricultural Monitoring Platform", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# Pydantic models
class CropAnalysis(BaseModel):
    crop_type: str
    health_score: float
    area_hectares: float
    disease_detected: bool
    disease_type: Optional[str] = None
    confidence: float

class SegmentationResult(BaseModel):
    segments: List[Dict]
    analysis: List[CropAnalysis]
    total_area: float
    health_summary: Dict

class YieldPrediction(BaseModel):
    estimated_yield: float
    confidence: float
    factors: Dict

# Global model instance
model = None

def load_yolo_model():
    """Load YOLOv8 segmentation model"""
    global model
    try:
        # Try to load custom trained model, fallback to pretrained
        model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n-seg.pt")
        model = YOLO(model_path)
        logger.info(f"Loaded YOLO model: {model_path}")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        # Load default segmentation model
        model = YOLO("yolov8n-seg.pt")

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def init_database():
    """Initialize database tables"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crop_analyses (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255),
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                crop_data JSONB,
                geospatial_data JSONB,
                health_metrics JSONB
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS satellite_data (
                id SERIAL PRIMARY KEY,
                acquisition_date TIMESTAMP,
                satellite_type VARCHAR(100),
                coordinates JSONB,
                ndvi_data JSONB,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
    finally:
        conn.close()

def analyze_crop_health(mask: np.ndarray, image: np.ndarray) -> Dict:
    """Analyze crop health from segmentation mask and original image"""
    try:
        # Extract crop region
        crop_region = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(crop_region, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(crop_region, cv2.COLOR_RGB2LAB)
        
        # Calculate vegetation indices (simplified NDVI approximation)
        red = image[:,:,0].astype(np.float32)
        green = image[:,:,1].astype(np.float32) 
        blue = image[:,:,2].astype(np.float32)
        
        # Green-Red Vegetation Index (GRVI)
        with np.errstate(divide='ignore', invalid='ignore'):
            grvi = (green - red) / (green + red)
            grvi = np.nan_to_num(grvi)
        
        # Health metrics
        masked_grvi = grvi[mask > 0]
        health_score = np.mean(masked_grvi) * 100 if len(masked_grvi) > 0 else 0
        health_score = max(0, min(100, health_score + 50))  # Normalize to 0-100
        
        # Disease detection (simplified - based on color variance)
        crop_pixels = crop_region[mask > 0]
        if len(crop_pixels) > 0:
            color_std = np.std(crop_pixels, axis=0)
            disease_indicator = np.mean(color_std)
            disease_detected = disease_indicator > 30  # Threshold for disease detection
        else:
            disease_detected = False
            disease_indicator = 0
        
        return {
            "health_score": float(health_score),
            "disease_detected": disease_detected,
            "disease_confidence": float(disease_indicator / 50),
            "vegetation_index": float(np.mean(masked_grvi)) if len(masked_grvi) > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Health analysis error: {e}")
        return {
            "health_score": 50.0,
            "disease_detected": False,
            "disease_confidence": 0.0,
            "vegetation_index": 0.0
        }

def calculate_area(mask: np.ndarray, pixel_size: float = 1.0) -> float:
    """Calculate area in hectares from segmentation mask"""
    pixel_count = np.sum(mask > 0)
    area_m2 = pixel_count * (pixel_size ** 2)
    area_hectares = area_m2 / 10000  # Convert to hectares
    return area_hectares

def predict_yield(crop_analysis: Dict, area_hectares: float) -> YieldPrediction:
    """Predict crop yield based on health metrics and area"""
    try:
        health_score = crop_analysis.get("health_score", 50)
        vegetation_index = crop_analysis.get("vegetation_index", 0)
        
        # Simplified yield prediction model
        # In practice, this would use ML models trained on historical data
        base_yield_per_hectare = 3.5  # tons per hectare (example for wheat)
        
        # Adjust based on health score
        health_factor = health_score / 100
        vegetation_factor = max(0, min(1, vegetation_index + 0.5))
        
        # Disease penalty
        disease_penalty = 0.8 if crop_analysis.get("disease_detected", False) else 1.0
        
        estimated_yield = (base_yield_per_hectare * area_hectares * 
                          health_factor * vegetation_factor * disease_penalty)
        
        # Confidence based on data quality
        confidence = min(0.95, health_factor * 0.7 + vegetation_factor * 0.3)
        
        factors = {
            "health_factor": health_factor,
            "vegetation_factor": vegetation_factor,
            "disease_penalty": disease_penalty,
            "base_yield_per_hectare": base_yield_per_hectare
        }
        
        return YieldPrediction(
            estimated_yield=round(estimated_yield, 2),
            confidence=round(confidence, 3),
            factors=factors
        )
        
    except Exception as e:
        logger.error(f"Yield prediction error: {e}")
        return YieldPrediction(estimated_yield=0.0, confidence=0.0, factors={})

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    load_yolo_model()
    init_database()

@app.get("/")
async def root():
    return {"message": "Agricultural Monitoring Platform API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/analyze/crop-segmentation", response_model=SegmentationResult)
async def analyze_crop_image(file: UploadFile = File(...)):
    """Perform crop segmentation and analysis on uploaded image"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        
        # Run YOLO segmentation
        results = model(image_np)
        
        segments = []
        analyses = []
        total_area = 0
        
        for r in results:
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
                scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
                classes = r.boxes.cls.cpu().numpy() if r.boxes is not None else []
                
                for i, mask in enumerate(masks):
                    # Resize mask to image size
                    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
                    
                    # Calculate area
                    area = calculate_area(mask_resized)
                    total_area += area
                    
                    # Analyze crop health
                    health_analysis = analyze_crop_health(mask_resized, image_np)
                    
                    # Determine crop type (simplified - would use trained model)
                    crop_types = ["wheat", "corn", "soybean", "rice", "barley"]
                    crop_type = crop_types[int(classes[i]) % len(crop_types)] if len(classes) > i else "unknown"
                    
                    # Create segment data
                    segment_data = {
                        "id": i,
                        "crop_type": crop_type,
                        "confidence": float(scores[i]) if len(scores) > i else 0.0,
                        "area_hectares": round(area, 4),
                        "bbox": boxes[i].tolist() if len(boxes) > i else [],
                        "mask_data": base64.b64encode(
                            cv2.imencode('.png', (mask_resized * 255).astype(np.uint8))[1]
                        ).decode('utf-8')
                    }
                    segments.append(segment_data)
                    
                    # Create analysis
                    analysis = CropAnalysis(
                        crop_type=crop_type,
                        health_score=health_analysis["health_score"],
                        area_hectares=area,
                        disease_detected=health_analysis["disease_detected"],
                        disease_type="leaf_spot" if health_analysis["disease_detected"] else None,
                        confidence=float(scores[i]) if len(scores) > i else 0.0
                    )
                    analyses.append(analysis)
        
        # Calculate health summary
        if analyses:
            avg_health = np.mean([a.health_score for a in analyses])
            disease_count = sum([1 for a in analyses if a.disease_detected])
            health_summary = {
                "average_health_score": round(avg_health, 2),
                "total_segments": len(analyses),
                "diseased_segments": disease_count,
                "healthy_segments": len(analyses) - disease_count,
                "overall_status": "healthy" if avg_health > 70 else "attention_needed" if avg_health > 50 else "critical"
            }
        else:
            health_summary = {
                "average_health_score": 0,
                "total_segments": 0,
                "diseased_segments": 0,
                "healthy_segments": 0,
                "overall_status": "no_crops_detected"
            }
        
        # Store in database
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO crop_analyses (filename, crop_data, health_metrics)
                    VALUES (%s, %s, %s)
                """, (
                    file.filename,
                    json.dumps([a.dict() for a in analyses]),
                    json.dumps(health_summary)
                ))
                conn.commit()
            except Exception as e:
                logger.error(f"Database insert error: {e}")
            finally:
                conn.close()
        
        return SegmentationResult(
            segments=segments,
            analysis=analyses,
            total_area=round(total_area, 4),
            health_summary=health_summary
        )
        
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/predict/yield")
async def predict_crop_yield(file: UploadFile = File(...)):
    """Predict crop yield from image analysis"""
    try:
        # First run segmentation
        segmentation_result = await analyze_crop_image(file)
        
        predictions = []
        total_yield = 0
        
        for analysis in segmentation_result.analysis:
            health_metrics = {
                "health_score": analysis.health_score,
                "disease_detected": analysis.disease_detected,
                "vegetation_index": 0.5  # Placeholder
            }
            
            yield_pred = predict_yield(health_metrics, analysis.area_hectares)
            total_yield += yield_pred.estimated_yield
            
            predictions.append({
                "crop_type": analysis.crop_type,
                "area_hectares": analysis.area_hectares,
                "predicted_yield": yield_pred.estimated_yield,
                "confidence": yield_pred.confidence,
                "factors": yield_pred.factors
            })
        
        return {
            "individual_predictions": predictions,
            "total_estimated_yield": round(total_yield, 2),
            "yield_unit": "tons",
            "prediction_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Yield prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Yield prediction failed: {str(e)}")

@app.get("/history/analyses")
async def get_analysis_history(limit: int = 10):
    """Get historical crop analyses"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, filename, upload_date, crop_data, health_metrics
            FROM crop_analyses
            ORDER BY upload_date DESC
            LIMIT %s
        """, (limit,))
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")
    finally:
        conn.close()

@app.post("/satellite/process")
async def process_satellite_data(
    acquisition_date: str,
    satellite_type: str = "Sentinel-2",
    coordinates: Dict = None
):
    """Process satellite imagery (placeholder for GDAL processing)"""
    try:
        # In a real implementation, this would:
        # 1. Download Sentinel-2 data using GDAL
        # 2. Calculate NDVI and other vegetation indices
        # 3. Perform change detection
        # 4. Store results in database
        
        # Simulated processing
        simulated_ndvi = np.random.rand(100, 100) * 0.8 + 0.1
        
        processed_data = {
            "acquisition_date": acquisition_date,
            "satellite_type": satellite_type,
            "coordinates": coordinates or {"lat": 40.7128, "lon": -74.0060},
            "ndvi_stats": {
                "mean": float(np.mean(simulated_ndvi)),
                "std": float(np.std(simulated_ndvi)),
                "min": float(np.min(simulated_ndvi)),
                "max": float(np.max(simulated_ndvi))
            },
            "processing_status": "completed",
            "processing_time": datetime.now().isoformat()
        }
        
        # Store in database
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO satellite_data (acquisition_date, satellite_type, coordinates, ndvi_data)
                    VALUES (%s, %s, %s, %s)
                """, (
                    acquisition_date,
                    satellite_type,
                    json.dumps(coordinates),
                    json.dumps(processed_data["ndvi_stats"])
                ))
                conn.commit()
            except Exception as e:
                logger.error(f"Satellite data storage error: {e}")
            finally:
                conn.close()
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Satellite processing error: {e}")
        raise HTTPException(status_code=500, detail="Satellite data processing failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
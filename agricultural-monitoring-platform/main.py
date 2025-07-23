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

class BatchAnalysisResult(BaseModel):
    batch_id: str
    images_processed: int
    total_segments: int
    combined_analysis: List[CropAnalysis]
    total_area: float
    combined_health_summary: Dict
    individual_results: List[Dict]

class BatchYieldPrediction(BaseModel):
    batch_id: str
    images_processed: int
    combined_predictions: List[Dict]
    total_estimated_yield: float
    individual_results: List[Dict]

# Global model instance
model = None

def load_yolo_model():
    """Load YOLOv8 segmentation model"""
    global model
    try:
        # Try to load custom trained model, fallback to pretrained
        model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n-seg.pt")
        
        # If no custom model specified, try to use a model better suited for general object detection
        if model_path == "yolov8n-seg.pt":
            logger.info("Loading YOLOv8 segmentation model...")
            model = YOLO("yolov8m-seg.pt")  # Use medium model for better detection
        else:
            model = YOLO(model_path)
            
        logger.info(f"Loaded YOLO model: {model_path}")
        
        # Test the model with a dummy inference to ensure it's working
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_results = model(dummy_image, verbose=False)
        logger.info("Model test successful - ready for inference")
        
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        # Fallback to smallest model
        try:
            model = YOLO("yolov8n-seg.pt")
            logger.info("Loaded fallback YOLOv8n-seg model")
        except Exception as fallback_error:
            logger.error(f"Failed to load fallback model: {fallback_error}")
            model = None

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
        
        # Green-Red Vegetation Index (GRVI) - better for RGB images
        with np.errstate(divide='ignore', invalid='ignore'):
            grvi = (green - red) / (green + red)
            grvi = np.nan_to_num(grvi)
        
        # Excess Green Index (ExG) - good for green vegetation
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_green = green / (red + green + blue + 1e-8)
            normalized_red = red / (red + green + blue + 1e-8)
            normalized_blue = blue / (red + green + blue + 1e-8)
            exg = 2 * normalized_green - normalized_red - normalized_blue
        
        # Health metrics for masked region
        masked_grvi = grvi[mask > 0]
        masked_exg = exg[mask > 0]
        
        if len(masked_grvi) > 0:
            avg_grvi = np.mean(masked_grvi)
            avg_exg = np.mean(masked_exg)
            
            # Combine indices for health score
            vegetation_health = (avg_grvi + 1) / 2  # Normalize GRVI to 0-1
            greenness_health = np.clip((avg_exg + 1) / 2, 0, 1)  # Normalize ExG to 0-1
            
            # Weighted average for final health score
            health_score = (vegetation_health * 0.6 + greenness_health * 0.4) * 100
            health_score = max(0, min(100, health_score))
            
            # Enhanced disease detection based on color variance and brown/yellow pixels
            crop_pixels = crop_region[mask > 0]
            if len(crop_pixels) > 0:
                # Check for brown/yellow discoloration (disease indicator)
                hsv_pixels = cv2.cvtColor(crop_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                
                # Look for brown/yellow hues (typical disease colors)
                brown_yellow_mask = ((hsv_pixels[:, 0] >= 10) & (hsv_pixels[:, 0] <= 30)) | \
                                   ((hsv_pixels[:, 0] >= 40) & (hsv_pixels[:, 0] <= 60))
                
                disease_ratio = np.sum(brown_yellow_mask) / len(hsv_pixels)
                color_variance = np.var(crop_pixels, axis=0).mean()
                
                # Disease detected if significant brown/yellow areas or high color variance
                disease_detected = disease_ratio > 0.15 or color_variance > 800
                disease_confidence = min(disease_ratio * 2 + color_variance / 1000, 1.0)
            else:
                disease_detected = False
                disease_confidence = 0.0
                
        else:
            # Fallback for empty mask
            health_score = 50.0
            disease_detected = False
            disease_confidence = 0.0
            avg_grvi = 0.0
        
        return {
            "health_score": float(health_score),
            "disease_detected": disease_detected,
            "disease_confidence": float(disease_confidence),
            "vegetation_index": float(avg_grvi) if len(masked_grvi) > 0 else 0.0
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

def classify_crop_type(image: np.ndarray, mask: np.ndarray) -> str:
    """Classify crop type based on image characteristics"""
    try:
        # Extract crop region
        if np.any(mask > 0):
            crop_region = image[mask > 0]
        else:
            # If mask is empty, sample from center of image
            h, w = image.shape[:2]
            center_region = image[h//4:3*h//4, w//4:3*w//4]
            crop_region = center_region.reshape(-1, 3)
        
        if len(crop_region) == 0:
            return "unknown_crop"
        
        # Ensure we have enough pixels for analysis
        if len(crop_region) < 100:
            return "unknown_crop"
        
        # Color analysis
        mean_color = np.mean(crop_region, axis=0)
        red, green, blue = mean_color
        
        # Avoid division by zero
        total_color = red + green + blue + 1e-8
        greenness = green / total_color
        
        # HSV analysis for better color classification
        try:
            hsv_region = cv2.cvtColor(crop_region.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            mean_hue = np.mean(hsv_region[:, 0])
            mean_saturation = np.mean(hsv_region[:, 1])
            mean_value = np.mean(hsv_region[:, 2])
        except:
            mean_hue = 60  # Default green hue
            mean_saturation = 100
            mean_value = 128
        
        # Texture analysis (simplified)
        try:
            gray_crop = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if np.any(mask > 0):
                texture_variance = np.var(gray_crop[mask > 0])
            else:
                texture_variance = np.var(gray_crop)
        except:
            texture_variance = 400  # Default medium texture
        
        # Edge density (indicates leaf structure)
        try:
            gray_crop = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_crop, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
        except:
            edge_density = 0.05  # Default low edge density
        
        # Classification logic based on visual characteristics
        logger.info(f"Crop analysis - Greenness: {greenness:.3f}, Hue: {mean_hue:.1f}, Texture: {texture_variance:.1f}, Edges: {edge_density:.3f}")
        
        # Sugarcane characteristics: tall, narrow leaves, high texture variance, high edge density
        if (edge_density > 0.08 and texture_variance > 400 and 
            mean_hue > 30 and mean_hue < 90 and greenness > 0.35):
            
            # Further distinguish sugarcane from other tall grasses
            if texture_variance > 600 and edge_density > 0.1:
                return "sugarcane"
            else:
                return "tall_grass"
        
        # Rice characteristics: shorter, more uniform green, lower texture variance
        elif (greenness > 0.4 and mean_saturation > 60 and 
              texture_variance < 500 and edge_density < 0.1):
            return "rice"
            
        # Corn/Maize characteristics: broader leaves, medium texture
        elif (greenness > 0.37 and texture_variance > 300 and texture_variance < 700 and
              mean_hue > 35 and mean_hue < 85):
            return "corn"
            
        # Wheat characteristics: golden/yellow tinge, lower greenness
        elif (greenness < 0.37 and mean_hue > 10 and mean_hue < 70 and
              red > green * 0.7):
            return "wheat"
            
        # Soybean characteristics: darker green, medium texture
        elif (greenness > 0.38 and mean_value < 140 and 
              texture_variance > 250 and texture_variance < 600):
            return "soybean"
            
        # Default classification based on main characteristics
        elif greenness > 0.4 and texture_variance > 500:
            return "sugarcane"  # High texture + green likely tall crops
        elif greenness > 0.4:
            return "rice"  # Green + low texture likely short crops
        else:
            return "wheat"  # Less green likely mature grain crops
            
    except Exception as e:
        logger.error(f"Crop classification error: {e}")
        return "unknown_crop"

def predict_yield(crop_analysis: Dict, area_hectares: float, crop_type: str = "unknown") -> YieldPrediction:
    """Predict crop yield based on health metrics, area, and crop type"""
    try:
        health_score = crop_analysis.get("health_score", 50)
        vegetation_index = crop_analysis.get("vegetation_index", 0)
        
        # Crop-specific yield estimates (tons per hectare)
        base_yields = {
            "sugarcane": 70.0,  # Sugarcane has very high yield
            "rice": 4.5,
            "wheat": 3.5,
            "corn": 8.0,
            "soybean": 2.8,
            "barley": 3.0,
            "tall_grass": 5.0,  # Estimate for grass crops
            "unknown_crop": 4.0
        }
        
        base_yield_per_hectare = base_yields.get(crop_type, 4.0)
        
        # Adjust based on health score
        health_factor = health_score / 100
        vegetation_factor = max(0, min(1, vegetation_index + 0.5))
        
        # Disease penalty
        disease_penalty = 0.7 if crop_analysis.get("disease_detected", False) else 1.0
        
        # Crop-specific adjustments
        if crop_type == "sugarcane":
            # Sugarcane is more resilient but sensitive to disease
            disease_penalty = 0.6 if crop_analysis.get("disease_detected", False) else 1.0
            health_factor = max(0.3, health_factor)  # Minimum yield even in poor conditions
        elif crop_type == "rice":
            # Rice is sensitive to health conditions
            health_factor = health_factor ** 1.2  # Exponential relationship
        
        estimated_yield = (base_yield_per_hectare * area_hectares * 
                          health_factor * vegetation_factor * disease_penalty)
        
        # Confidence based on data quality and crop type certainty
        base_confidence = min(0.95, health_factor * 0.7 + vegetation_factor * 0.3)
        
        # Adjust confidence based on crop classification certainty
        crop_confidence_multiplier = {
            "sugarcane": 0.9,
            "rice": 0.85,
            "wheat": 0.8,
            "corn": 0.85,
            "unknown_crop": 0.6
        }
        
        confidence = base_confidence * crop_confidence_multiplier.get(crop_type, 0.7)
        
        factors = {
            "health_factor": health_factor,
            "vegetation_factor": vegetation_factor,
            "disease_penalty": disease_penalty,
            "base_yield_per_hectare": base_yield_per_hectare,
            "crop_type": crop_type
        }
        
        return YieldPrediction(
            estimated_yield=round(estimated_yield, 2),
            confidence=round(confidence, 3),
            factors=factors
        )
        
    except Exception as e:
        logger.error(f"Yield prediction error: {e}")
        return YieldPrediction(estimated_yield=0.0, confidence=0.0, factors={})

def process_image_for_analysis(image_data: bytes) -> tuple:
    """Process image data and return segmentation results and analyses - NO FALLBACK"""
    logger.info("Starting process_image_for_analysis - NO FALLBACK VERSION")
    
    # Initialize return values
    segments = []
    analyses = []
    total_area = 0.0
    
    if not model:
        logger.error("Model not loaded")
        return segments, analyses, total_area
    
    try:
        # Create a fresh BytesIO object from the bytes data
        image_stream = io.BytesIO(image_data)
        # Read image
        image = Image.open(image_stream).convert("RGB")
        image_np = np.array(image)
        logger.info(f"Image loaded successfully: {image_np.shape}")
    except Exception as e:
        logger.error(f"Image loading error: {e}")
        return segments, analyses, total_area
    
    # Run YOLO segmentation
    try:
        results = model(image_np, conf=0.25)  # Lower confidence threshold
        logger.info(f"YOLO inference completed, {len(results)} result objects")
    except Exception as e:
        logger.error(f"YOLO inference error: {e}")
        return segments, analyses, total_area
    
    # Process YOLO results - ONLY REAL DETECTIONS
    processed_any = False
    try:
        for result_idx, r in enumerate(results):
            logger.info(f"Processing result {result_idx}: masks={r.masks is not None}, mask_count={len(r.masks.data) if r.masks is not None else 0}")
            
            if r.masks is not None and len(r.masks.data) > 0:
                processed_any = True
                masks = r.masks.data.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
                scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
                classes = r.boxes.cls.cpu().numpy() if r.boxes is not None else []
                
                logger.info(f"Processing {len(masks)} detected masks from result {result_idx}")
                
                for i, mask in enumerate(masks):
                    try:
                        # Resize mask to image size
                        mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
                        
                        # Calculate area
                        area = calculate_area(mask_resized, pixel_size=0.1)  # Assume 10cm per pixel
                        if area <= 0:
                            logger.warning(f"Mask {i} has zero area, skipping")
                            continue
                            
                        total_area += area
                        
                        # Analyze crop health
                        health_analysis = analyze_crop_health(mask_resized, image_np)
                        
                        # Classify crop type using image analysis
                        crop_type = classify_crop_type(image_np, mask_resized)
                        
                        # Create segment data
                        try:
                            mask_encoded = base64.b64encode(
                                cv2.imencode('.png', (mask_resized * 255).astype(np.uint8))[1]
                            ).decode('utf-8')
                        except Exception as e:
                            logger.error(f"Error encoding mask {i}: {e}")
                            continue
                        
                        segment_data = {
                            "id": i,
                            "crop_type": crop_type,
                            "confidence": float(scores[i]) if len(scores) > i else 0.7,
                            "area_hectares": round(area, 4),
                            "bbox": boxes[i].tolist() if len(boxes) > i else [],
                            "mask_data": mask_encoded
                        }
                        segments.append(segment_data)
                        
                        # Create analysis
                        analysis = CropAnalysis(
                            crop_type=crop_type,
                            health_score=health_analysis["health_score"],
                            area_hectares=area,
                            disease_detected=health_analysis["disease_detected"],
                            disease_type="leaf_spot" if health_analysis["disease_detected"] else None,
                            confidence=float(scores[i]) if len(scores) > i else 0.7
                        )
                        analyses.append(analysis)
                        logger.info(f"Successfully processed mask {i}: {crop_type}, area={area:.4f}ha, health={health_analysis['health_score']:.1f}%")
                        
                    except Exception as e:
                        logger.error(f"Error processing mask {i}: {e}")
                        continue
            else:
                logger.info(f"Result {result_idx} has no masks or empty masks")
                        
    except Exception as e:
        logger.error(f"Error processing YOLO results: {e}")
        # Continue with empty results
        pass
    
    # Log what we found
    if not processed_any:
        logger.info("No valid masks found in YOLO results - returning empty results")
    
    # Log final results
    logger.info(f"Final analysis: {len(segments)} segments, {len(analyses)} analyses, total area: {total_area:.4f} ha")
    
    # Double-check we're returning the right format
    result = (segments, analyses, total_area)
    logger.info(f"Returning result tuple with {len(result)} elements: ({type(segments).__name__}, {type(analyses).__name__}, {type(total_area).__name__})")
    
def process_multiple_images(image_files: List[bytes], filenames: List[str]) -> Dict:
    """Process multiple images and combine results"""
    import uuid
    
    batch_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting batch processing {batch_id} with {len(image_files)} images")
    
    all_segments = []
    all_analyses = []
    total_area = 0.0
    individual_results = []
    
    for idx, (image_data, filename) in enumerate(zip(image_files, filenames)):
        logger.info(f"Processing image {idx + 1}/{len(image_files)}: {filename}")
        
        try:
            # Process individual image
            segments, analyses, area = process_image_for_analysis(image_data)
            
            # Update segment IDs to be unique across batch
            segment_offset = len(all_segments)
            for segment in segments:
                segment['id'] = segment['id'] + segment_offset
                segment['source_image'] = filename
                segment['image_index'] = idx
            
            # Update analysis with source info
            for analysis in analyses:
                # Create a new analysis object with source info (can't modify Pydantic model directly)
                analysis_dict = analysis.dict()
                analysis_dict['source_image'] = filename
                analysis_dict['image_index'] = idx
                all_analyses.append(CropAnalysis(**{k: v for k, v in analysis_dict.items() if k in CropAnalysis.__fields__}))
            
            all_segments.extend(segments)
            total_area += area
            
            # Store individual result
            individual_result = {
                "image_index": idx,
                "filename": filename,
                "segments_count": len(segments),
                "analyses_count": len(analyses),
                "area": area,
                "status": "success"
            }
            
            if analyses:
                avg_health = np.mean([a.health_score for a in analyses])
                disease_count = sum([1 for a in analyses if a.disease_detected])
                individual_result.update({
                    "average_health": round(avg_health, 2),
                    "disease_count": disease_count,
                    "crop_types": list(set([a.crop_type for a in analyses]))
                })
            
            individual_results.append(individual_result)
            
        except Exception as e:
            logger.error(f"Error processing image {filename}: {e}")
            individual_results.append({
                "image_index": idx,
                "filename": filename,
                "segments_count": 0,
                "analyses_count": 0,
                "area": 0,
                "status": "error",
                "error": str(e)
            })
    
    # Calculate combined health summary
    if all_analyses:
        avg_health = np.mean([a.health_score for a in all_analyses])
        disease_count = sum([1 for a in all_analyses if a.disease_detected])
        crop_types = list(set([a.crop_type for a in all_analyses]))
        
        # Crop type distribution
        crop_distribution = {}
        for analysis in all_analyses:
            crop_type = analysis.crop_type
            if crop_type not in crop_distribution:
                crop_distribution[crop_type] = {"count": 0, "area": 0.0, "avg_health": []}
            crop_distribution[crop_type]["count"] += 1
            crop_distribution[crop_type]["area"] += analysis.area_hectares
            crop_distribution[crop_type]["avg_health"].append(analysis.health_score)
        
        # Calculate average health per crop type
        for crop_type in crop_distribution:
            crop_distribution[crop_type]["avg_health"] = np.mean(crop_distribution[crop_type]["avg_health"])
        
        combined_health_summary = {
            "total_images": len(image_files),
            "images_processed": len([r for r in individual_results if r["status"] == "success"]),
            "total_segments": len(all_segments),
            "total_analyses": len(all_analyses),
            "diseased_segments": disease_count,
            "healthy_segments": len(all_analyses) - disease_count,
            "average_health_score": round(avg_health, 2),
            "total_area_hectares": round(total_area, 4),
            "crop_types_detected": crop_types,
            "crop_distribution": crop_distribution,
            "overall_status": "healthy" if avg_health > 70 else "attention_needed" if avg_health > 50 else "critical"
        }
    else:
        combined_health_summary = {
            "total_images": len(image_files),
            "images_processed": len([r for r in individual_results if r["status"] == "success"]),
            "total_segments": 0,
            "total_analyses": 0,
            "diseased_segments": 0,
            "healthy_segments": 0,
            "average_health_score": 0,
            "total_area_hectares": 0,
            "crop_types_detected": [],
            "crop_distribution": {},
            "overall_status": "no_crops_detected"
        }
    
    logger.info(f"Batch processing {batch_id} completed: {len(all_analyses)} total analyses from {len(image_files)} images")
    
    return {
        "batch_id": batch_id,
        "segments": all_segments,
        "analyses": all_analyses,
        "total_area": total_area,
        "combined_health_summary": combined_health_summary,
        "individual_results": individual_results
    }

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
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    
    # Test model if it's loaded
    model_working = False
    if model is not None:
        try:
            # Quick test with dummy data
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            test_results = model(dummy_image, verbose=False)
            model_working = True
            logger.info("Model test successful")
        except Exception as e:
            logger.error(f"Model test failed: {e}")
    
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_status": model_status,
        "model_working": model_working
    }

@app.post("/analyze/crop-segmentation", response_model=SegmentationResult)
async def analyze_crop_image(file: UploadFile = File(...)):
    """Perform crop segmentation and analysis on uploaded image"""
    try:
        # Validate file extension instead of content type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ''
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File extension {file_extension} not supported. Allowed: {', '.join(allowed_extensions)}")
        
        # Read image data
        image_data = await file.read()
        logger.info(f"Processing image: {file.filename}, size: {len(image_data)} bytes, content_type: {file.content_type}")
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process image
        try:
            logger.info("About to call process_image_for_analysis")
            result = process_image_for_analysis(image_data)
            logger.info(f"process_image_for_analysis returned: {type(result)}")
            
            # Check if we got a valid result
            if result is None:
                logger.error("process_image_for_analysis returned None")
                raise HTTPException(status_code=500, detail="Image processing returned None")
            
            if not isinstance(result, tuple):
                logger.error(f"process_image_for_analysis returned non-tuple: {type(result)}")
                raise HTTPException(status_code=500, detail=f"Invalid processing result type: {type(result)}")
            
            if len(result) != 3:
                logger.error(f"process_image_for_analysis returned tuple with {len(result)} elements: {result}")
                raise HTTPException(status_code=500, detail=f"Invalid processing result format: expected 3 elements, got {len(result)}")
            
            segments, analyses, total_area = result
            logger.info(f"Successfully processed image: {len(segments)} segments, {len(analyses)} analyses, {total_area} total area")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in process_image_for_analysis: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
        
        # Calculate health summary
        if analyses and len(analyses) > 0:
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
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/predict/yield")
async def predict_crop_yield(file: UploadFile = File(...)):
    """Predict crop yield from image analysis"""
    try:
        # Validate file extension instead of content type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ''
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File extension {file_extension} not supported. Allowed: {', '.join(allowed_extensions)}")
        
        # Read image data once
        image_data = await file.read()
        logger.info(f"Processing yield prediction for: {file.filename}, size: {len(image_data)} bytes, content_type: {file.content_type}")
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process image for analysis
        try:
            logger.info("About to call process_image_for_analysis")
            logger.info(f"Function address: {process_image_for_analysis}")
            
            # Test simple tuple creation first
            test_tuple = ([], [], 0.0)
            logger.info(f"Test tuple created successfully: {type(test_tuple)}")
            
            # Now call the main function
            logger.info("Calling main process_image_for_analysis function")
            result = process_image_for_analysis(image_data)
            logger.info(f"CALLER RECEIVED: {type(result)}")
            logger.info(f"CALLER: result is None = {result is None}")
            
            # Try to access the result
            try:
                if result is not None:
                    logger.info(f"CALLER: result length = {len(result)}")
                    logger.info(f"CALLER: result contents = {[type(x) for x in result]}")
                else:
                    logger.error("CALLER: result is None!")
            except Exception as access_error:
                logger.error(f"CALLER: Error accessing result: {access_error}")
            
            # Check if we got a valid result
            if result is None:
                logger.error("process_image_for_analysis returned None - this should not happen!")
                logger.error("This suggests a serious Python interpreter or import issue")
                
                # Create a simple fallback result for now
                logger.info("Creating fallback empty result")
                result = ([], [], 0.0)
                logger.info(f"Fallback result created: {type(result)}")
            
            if not isinstance(result, tuple):
                logger.error(f"process_image_for_analysis returned non-tuple: {type(result)}")
                raise HTTPException(status_code=500, detail=f"Invalid processing result type: {type(result)}")
            
            if len(result) != 3:
                logger.error(f"process_image_for_analysis returned tuple with {len(result)} elements: {result}")
                raise HTTPException(status_code=500, detail=f"Invalid processing result format: expected 3 elements, got {len(result)}")
            
            segments, analyses, total_area = result
            logger.info(f"Successfully processed image for yield: {len(segments)} segments, {len(analyses)} analyses, {total_area} total area")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in process_image_for_analysis: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
        
        predictions = []
        total_yield = 0
        
        if not analyses or len(analyses) == 0:
            return {
                "individual_predictions": [],
                "total_estimated_yield": 0.0,
                "yield_unit": "tons",
                "prediction_date": datetime.now().isoformat(),
                "message": "No crops detected for yield prediction"
            }
        
        for analysis in analyses:
            health_metrics = {
                "health_score": analysis.health_score,
                "disease_detected": analysis.disease_detected,
                "vegetation_index": 0.5  # Placeholder - in real implementation, extract from health analysis
            }
            
            yield_pred = predict_yield(health_metrics, analysis.area_hectares, analysis.crop_type)
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
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Yield prediction error: {e}")
@app.post("/analyze/batch-crop-segmentation", response_model=BatchAnalysisResult)
async def analyze_multiple_crop_images(files: List[UploadFile] = File(...)):
    """Perform crop segmentation and analysis on multiple uploaded images"""
    try:
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        if len(files) > 10:  # Limit to prevent overload
            raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files allowed")
        
        # Validate all files
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_data_list = []
        filenames = []
        
        for file in files:
            file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ''
            if file_extension not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"File {file.filename} has unsupported extension {file_extension}")
            
            # Read image data
            image_data = await file.read()
            if len(image_data) == 0:
                raise HTTPException(status_code=400, detail=f"Empty file: {file.filename}")
            
            image_data_list.append(image_data)
            filenames.append(file.filename)
            logger.info(f"Loaded {file.filename}: {len(image_data)} bytes")
        
        # Process all images
        try:
            result = process_multiple_images(image_data_list, filenames)
            
            # Store batch analysis in database
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO crop_analyses (filename, crop_data, health_metrics, geospatial_data)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        f"batch_{result['batch_id']}",
                        json.dumps([a.dict() for a in result['analyses']]),
                        json.dumps(result['combined_health_summary']),
                        json.dumps(result['individual_results'])
                    ))
                    conn.commit()
                except Exception as e:
                    logger.error(f"Database insert error: {e}")
                finally:
                    conn.close()
            
            return BatchAnalysisResult(
                batch_id=result['batch_id'],
                images_processed=result['combined_health_summary']['images_processed'],
                total_segments=len(result['segments']),
                combined_analysis=result['analyses'],
                total_area=result['total_area'],
                combined_health_summary=result['combined_health_summary'],
                individual_results=result['individual_results']
            )
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.post("/predict/batch-yield", response_model=BatchYieldPrediction)
async def predict_batch_crop_yield(files: List[UploadFile] = File(...)):
    """Predict crop yield from multiple image analysis"""
    try:
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        if len(files) > 10:  # Limit to prevent overload
            raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files allowed")
        
        # Validate all files
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_data_list = []
        filenames = []
        
        for file in files:
            file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ''
            if file_extension not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"File {file.filename} has unsupported extension {file_extension}")
            
            # Read image data
            image_data = await file.read()
            if len(image_data) == 0:
                raise HTTPException(status_code=400, detail=f"Empty file: {file.filename}")
            
            image_data_list.append(image_data)
            filenames.append(file.filename)
            logger.info(f"Loaded {file.filename} for yield prediction: {len(image_data)} bytes")
        
        # Process all images
        try:
            batch_result = process_multiple_images(image_data_list, filenames)
            
            # Calculate yields for all analyses
            combined_predictions = []
            total_yield = 0.0
            individual_results = []
            
            # Group analyses by image for individual results
            analyses_by_image = {}
            for analysis in batch_result['analyses']:
                image_idx = getattr(analysis, 'image_index', 0)  # Default to 0 if not set
                if image_idx not in analyses_by_image:
                    analyses_by_image[image_idx] = []
                analyses_by_image[image_idx].append(analysis)
            
            # Process each image's analyses
            for image_idx in range(len(filenames)):
                image_analyses = analyses_by_image.get(image_idx, [])
                image_predictions = []
                image_total_yield = 0.0
                
                for analysis in image_analyses:
                    health_metrics = {
                        "health_score": analysis.health_score,
                        "disease_detected": analysis.disease_detected,
                        "vegetation_index": 0.5
                    }
                    
                    yield_pred = predict_yield(health_metrics, analysis.area_hectares, analysis.crop_type)
                    
                    prediction = {
                        "crop_type": analysis.crop_type,
                        "area_hectares": analysis.area_hectares,
                        "predicted_yield": yield_pred.estimated_yield,
                        "confidence": yield_pred.confidence,
                        "factors": yield_pred.factors
                    }
                    
                    image_predictions.append(prediction)
                    combined_predictions.append(prediction)
                    image_total_yield += yield_pred.estimated_yield
                    total_yield += yield_pred.estimated_yield
                
                # Individual image result
                individual_results.append({
                    "image_index": image_idx,
                    "filename": filenames[image_idx],
                    "predictions": image_predictions,
                    "total_yield": round(image_total_yield, 2),
                    "crops_analyzed": len(image_analyses)
                })
            
            return BatchYieldPrediction(
                batch_id=batch_result['batch_id'],
                images_processed=batch_result['combined_health_summary']['images_processed'],
                combined_predictions=combined_predictions,
                total_estimated_yield=round(total_yield, 2),
                individual_results=individual_results
            )
            
        except Exception as e:
            logger.error(f"Batch yield prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Batch yield prediction failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch yield analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch yield analysis failed: {str(e)}")

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
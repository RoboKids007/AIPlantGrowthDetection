#!/usr/bin/env python3
"""
Plant Disease Detection - FastAPI Web Application
A modern web interface for plant disease detection using YOLO.
"""

import os
import io
import cv2
import numpy as np
import uvicorn
import glob
from pathlib import Path
from PIL import Image
from typing import List, Optional
import logging
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant disease detection using YOLOv8",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Create directories
Path("static").mkdir(exist_ok=True)
Path("static/uploads").mkdir(exist_ok=True)
Path("static/results").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global model variable
model = None
MODEL_PATH = "plant_disease_yolov8n_trained.pt"

# Class names mapping
CLASS_NAMES = [
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf spot',
    'Bell_pepper leaf', 'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot',
    'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf early blight',
    'Potato leaf late blight', 'Potato leaf', 'Raspberry leaf', 'Soyabean leaf',
    'Soybean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf',
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf bacterial spot',
    'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus',
    'Tomato leaf', 'Tomato mold leaf', 'Tomato two spotted spider mites leaf',
    'grape leaf black rot', 'grape leaf'
]

# Disease information database
DISEASE_INFO = {
    'Apple Scab Leaf': {
        'severity': 'High',
        'treatment': 'Apply fungicide sprays (captan, myclobutanil) every 7-14 days. Improve air circulation by pruning. Remove and destroy infected leaves immediately.',
        'prevention': 'Plant scab-resistant varieties. Avoid overhead watering. Apply preventive fungicides in spring.',
        'description': 'Fungal disease causing dark, scabby lesions on leaves and fruit. Can severely reduce fruit quality and tree health.',
        'immediate_action': 'Remove all fallen leaves and infected plant parts immediately',
        'long_term': 'Consider replacing with resistant varieties like Liberty or Enterprise'
    },
    'Apple rust leaf': {
        'severity': 'Medium',
        'treatment': 'Apply systemic fungicides (propiconazole). Remove nearby juniper trees if possible (alternate host).',
        'prevention': 'Plant rust-resistant apple varieties. Maintain proper tree spacing for air circulation.',
        'description': 'Rust disease causing bright orange spots on leaves, leading to premature defoliation.',
        'immediate_action': 'Spray with fungicide at first sign of orange spots',
        'long_term': 'Remove juniper trees within 1-2 miles if feasible'
    },
    'Bell_pepper leaf spot': {
        'severity': 'Medium', 
        'treatment': 'Apply copper-based fungicides (copper sulfate) weekly. Remove and destroy infected plants immediately.',
        'prevention': 'Use 3-year crop rotation. Avoid overhead irrigation. Plant certified disease-free seeds.',
        'description': 'Bacterial or fungal leaf spots that can rapidly spread and reduce fruit yield significantly.',
        'immediate_action': 'Isolate infected plants and apply copper spray',
        'long_term': 'Implement drip irrigation system and improve soil drainage'
    },
    'Corn Gray leaf spot': {
        'severity': 'High',
        'treatment': 'Apply triazole fungicides (propiconazole, tebuconazole) at first symptoms. Use resistant hybrids.',
        'prevention': 'Practice 2-3 year rotation with non-host crops. Manage crop residue by tillage.',
        'description': 'Fungal disease causing rectangular gray lesions between leaf veins, can cause 50% yield loss.',
        'immediate_action': 'Apply fungicide immediately when lesions appear',
        'long_term': 'Switch to resistant corn varieties and improve field drainage'
    },
    'Corn leaf blight': {
        'severity': 'High',
        'treatment': 'Apply strobilurin or triazole fungicides immediately. Use resistant corn hybrids for next season.',
        'prevention': 'Rotate with soybeans or other non-host crops. Bury crop residue completely.',
        'description': 'Serious fungal disease causing large brown lesions that can destroy entire leaves and reduce yields by 30-50%.',
        'immediate_action': 'Apply fungicide within 24 hours of symptom detection',
        'long_term': 'Plant only resistant hybrids and practice minimum 2-year rotation'
    },
    'Corn rust leaf': {
        'severity': 'Medium',
        'treatment': 'Apply fungicides only if infection is severe before grain filling. Monitor weather conditions.',
        'prevention': 'Plant rust-resistant varieties. Monitor for early symptoms during humid weather.',
        'description': 'Orange to reddish-brown pustules on leaves, typically occurs late in growing season.',
        'immediate_action': 'Monitor closely - treatment needed only if severe early in season',
        'long_term': 'Select varieties with good rust resistance ratings'
    },
    'Potato leaf early blight': {
        'severity': 'High',
        'treatment': 'Apply chlorothalonil or copper fungicides every 7-10 days. Remove lower infected leaves.',
        'prevention': 'Use certified seed potatoes. Practice 3-4 year rotation. Ensure proper plant spacing.',
        'description': 'Common fungal disease causing characteristic target-spot lesions that expand rapidly in warm, humid conditions.',
        'immediate_action': 'Remove infected lower leaves and apply fungicide spray',
        'long_term': 'Improve air circulation and avoid overhead watering'
    },
    'Potato leaf late blight': {
        'severity': 'Very High',
        'treatment': 'URGENT: Apply systemic fungicides (metalaxyl + chlorothalonil) immediately. Destroy severely infected plants.',
        'prevention': 'Use certified seed potatoes. Apply preventive fungicides in cool, wet weather. Plant resistant varieties.',
        'description': 'DEVASTATING disease that can destroy entire potato crops within days. The same pathogen that caused the Irish Potato Famine.',
        'immediate_action': 'EMERGENCY: Apply fungicide within hours, destroy infected plants completely',
        'long_term': 'Only plant certified disease-free seed potatoes and resistant varieties'
    },
    'Squash Powdery mildew leaf': {
        'severity': 'Medium',
        'treatment': 'Apply sulfur-based fungicides or potassium bicarbonate. Improve air circulation around plants.',
        'prevention': 'Plant resistant varieties. Avoid overhead watering. Provide adequate spacing between plants.',
        'description': 'White powdery coating on leaves that reduces photosynthesis and fruit quality.',
        'immediate_action': 'Spray with baking soda solution (1 tsp per quart water) as emergency treatment',
        'long_term': 'Select powdery mildew resistant squash varieties'
    },
    'Tomato Early blight leaf': {
        'severity': 'High',
        'treatment': 'Apply chlorothalonil or mancozeb fungicides every 7-10 days. Remove lower infected leaves.',
        'prevention': 'Use 3-year crop rotation. Apply mulch to prevent soil splash. Ensure proper plant spacing.',
        'description': 'Fungal disease causing characteristic target-spot lesions, starting on lower leaves and progressing upward.',
        'immediate_action': 'Remove infected lower leaves and apply fungicide spray',
        'long_term': 'Practice crop rotation and improve garden sanitation'
    },
    'Tomato Septoria leaf spot': {
        'severity': 'Medium',
        'treatment': 'Apply fungicides (chlorothalonil, copper) every 10-14 days. Remove infected leaves promptly.',
        'prevention': 'Mulch around plants. Avoid overhead watering. Provide good air circulation.',
        'description': 'Small circular spots with dark borders and light centers, causing rapid defoliation.',
        'immediate_action': 'Remove spotted leaves and improve air circulation',
        'long_term': 'Install drip irrigation and use disease-resistant varieties'
    },
    'Tomato leaf bacterial spot': {
        'severity': 'High',
        'treatment': 'Apply copper-based bactericides immediately. Remove and destroy severely infected plants.',
        'prevention': 'Use certified disease-free seeds. Avoid overhead irrigation. Practice crop rotation.',
        'description': 'Bacterial disease causing dark spots with yellow halos, leading to severe defoliation.',
        'immediate_action': 'Spray with copper bactericide and remove infected plants',
        'long_term': 'Use only certified seeds and implement strict sanitation'
    },
    'Tomato leaf late blight': {
        'severity': 'Very High',
        'treatment': 'URGENT: Apply systemic fungicides immediately. Destroy infected plants completely.',
        'prevention': 'Plant resistant varieties. Ensure excellent air circulation. Monitor weather conditions.',
        'description': 'CRITICAL: Rapidly spreading disease with water-soaked lesions that can destroy entire crop within days.',
        'immediate_action': 'EMERGENCY: Apply fungicide and destroy infected plants within hours',
        'long_term': 'Only grow resistant varieties and improve garden drainage'
    },
    'Tomato leaf mosaic virus': {
        'severity': 'High',
        'treatment': 'Remove infected plants, control aphids',
        'prevention': 'Use virus-free seed, control vectors',
        'description': 'Viral disease causing mottled yellow-green patterns'
    },
    'Tomato leaf yellow virus': {
        'severity': 'High',
        'treatment': 'Remove infected plants, control whiteflies',
        'prevention': 'Use resistant varieties, control vectors',
        'description': 'Viral disease causing yellowing and stunting'
    },
    'Tomato mold leaf': {
        'severity': 'Medium',
        'treatment': 'Improve ventilation, fungicide application',
        'prevention': 'Proper spacing, avoid high humidity',
        'description': 'Fungal growth in high humidity conditions'
    },
    'Tomato two spotted spider mites leaf': {
        'severity': 'Medium',
        'treatment': 'Miticide application, increase humidity',
        'prevention': 'Regular monitoring, predatory mites',
        'description': 'Tiny mites causing stippling and webbing'
    },
    'grape leaf black rot': {
        'severity': 'High',
        'treatment': 'Fungicide sprays, remove infected fruit',
        'prevention': 'Pruning for air circulation, sanitation',
        'description': 'Fungal disease causing black circular lesions'
    }
}

def load_model():
    """Load the trained YOLO model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            # Fallback to nano model if trained model not found
            model = YOLO('yolov8n.pt')
            logger.warning(f"Trained model not found, using default YOLOv8n")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    load_model()
    logger.info("Plant Disease Detection API started successfully")

def process_image(image: Image.Image) -> dict:
    """Process image and return detection results with smart resizing"""
    try:
        # Get original dimensions
        original_width, original_height = image.size
        logger.info(f"Original image size: {original_width}x{original_height}")
        
        # Smart resize to maintain aspect ratio for YOLO
        target_size = 640
        
        # Calculate scaling factor
        scale = min(target_size / original_width, target_size / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image maintaining aspect ratio
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a square canvas and paste the resized image (letterboxing)
        canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))  # Gray padding
        
        # Calculate position to center the image
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        
        # Paste the resized image onto the canvas
        canvas.paste(resized_image, (x_offset, y_offset))
        
        # Convert to numpy array for YOLO
        img_array = np.array(canvas)
        
        logger.info(f"Processed for YOLO: {target_size}x{target_size}")
        
        # Run inference on processed image
        results = model(img_array, conf=0.25)
        result = results[0]
        
        detections = []
        # Use original image for annotation display
        annotated_image = np.array(image)
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Scale detection boxes back to original image coordinates
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                # Adjust box coordinates from letterboxed image back to original
                x1, y1, x2, y2 = box
                
                # Remove letterbox offset
                x1 = (x1 - x_offset) / scale
                y1 = (y1 - y_offset) / scale  
                x2 = (x2 - x_offset) / scale
                y2 = (y2 - y_offset) / scale
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, original_width))
                y1 = max(0, min(y1, original_height))
                x2 = max(0, min(x2, original_width))  
                y2 = max(0, min(y2, original_height))
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class name
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
                
                # Determine if it's diseased or healthy
                is_diseased = any(disease in class_name.lower() for disease in 
                                ['scab', 'spot', 'blight', 'rust', 'mildew', 'rot', 'virus', 'mold', 'mites'])
                
                # Get disease info
                disease_info = DISEASE_INFO.get(class_name, {
                    'severity': 'Unknown',
                    'treatment': 'Consult agricultural specialist',
                    'prevention': 'Follow good agricultural practices',
                    'description': 'Disease information not available'
                })
                
                detection = {
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'is_diseased': is_diseased,
                    'severity': disease_info.get('severity', 'Unknown'),
                    'treatment': disease_info.get('treatment', 'Consult specialist'),
                    'prevention': disease_info.get('prevention', 'Good practices'),
                    'description': disease_info.get('description', 'No description')
                }
                detections.append(detection)
                
                # Draw bounding box
                color = (255, 0, 0) if is_diseased else (0, 255, 0)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return {
            'detections': detections,
            'annotated_image': annotated_image,
            'total_detections': len(detections),
            'diseased_count': sum(1 for d in detections if d['is_diseased']),
            'healthy_count': sum(1 for d in detections if not d['is_diseased'])
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Predict diseases in uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        results = process_image(image)
        
        # Save original and annotated images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"original_{timestamp}.jpg"
        annotated_filename = f"annotated_{timestamp}.jpg"
        
        # Save original
        original_path = f"static/uploads/{original_filename}"
        image.save(original_path)
        
        # Save annotated
        annotated_path = f"static/results/{annotated_filename}"
        annotated_image = Image.fromarray(results['annotated_image'])
        annotated_image.save(annotated_path)
        
        # Convert both images to base64 for display
        # Original image to base64
        original_buffered = io.BytesIO()
        image.save(original_buffered, format="JPEG")
        original_base64 = base64.b64encode(original_buffered.getvalue()).decode()
        
        # Annotated image to base64
        annotated_buffered = io.BytesIO()
        annotated_image.save(annotated_buffered, format="JPEG")
        annotated_base64 = base64.b64encode(annotated_buffered.getvalue()).decode()
        
        return {
            'success': True,
            'filename': file.filename,
            'original_url': f"/static/uploads/{original_filename}",
            'annotated_url': f"/static/results/{annotated_filename}",
            'original_image': original_base64,
            'annotated_image': annotated_base64,
            'detections': results['detections'],
            'summary': {
                'total_detections': results['total_detections'],
                'diseased_count': results['diseased_count'],
                'healthy_count': results['healthy_count']
            },
            'timestamp': timestamp
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        'model_type': 'YOLOv8',
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'model_path': MODEL_PATH,
        'input_size': 640
    }

@app.get("/test-images/")
async def get_test_images():
    """Get list of test images from validation folder"""
    try:
        valid_folder = Path("archive/valid/images")
        if not valid_folder.exists():
            return []
        
        # Get first 20 images for quick testing
        image_files = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        
        for ext in extensions:
            for img_path in valid_folder.glob(ext):
                if len(image_files) < 20:  # Limit to 20 images
                    image_files.append({
                        "name": img_path.name,
                        "url": f"/test-image/{img_path.name}"
                    })
        
        return image_files
    except Exception as e:
        logger.error(f"Error loading test images: {e}")
        return []

@app.get("/test-image/{image_name}")
async def serve_test_image(image_name: str):
    """Serve a test image from validation folder"""
    try:
        image_path = Path("archive/valid/images") / image_name
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(
            path=str(image_path),
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"}
        )
    except Exception as e:
        logger.error(f"Error serving test image: {e}")
        raise HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

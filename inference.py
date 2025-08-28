#!/usr/bin/env python3
"""
Plant Disease Detection - Inference Script
This script uses a trained YOLO model to detect plant diseases in images.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

class PlantDiseaseDetector:
    def __init__(self, model_path, conf_threshold=0.25):
        """
        Initialize the Plant Disease Detector
        
        Args:
            model_path (str): Path to the trained YOLO model
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Class names from the dataset
        self.class_names = [
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
    
    def detect_image(self, image_path, save_path=None, show=True):
        """
        Detect plant diseases in a single image
        
        Args:
            image_path (str): Path to the input image
            save_path (str): Path to save the result image
            show (bool): Whether to display the result
        """
        # Run inference
        results = self.model(image_path, conf=self.conf_threshold)
        
        # Get the first result
        result = results[0]
        
        # Load original image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw detections
        annotated_image = image_rgb.copy()
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                
                # Get class name
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                
                # Determine color based on disease type
                if any(disease in class_name.lower() for disease in ['scab', 'spot', 'blight', 'rust', 'mildew', 'rot', 'virus', 'mold', 'mites']):
                    color = (255, 0, 0)  # Red for diseased
                else:
                    color = (0, 255, 0)  # Green for healthy
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save result if path provided
        if save_path:
            result_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, result_bgr)
            print(f"Result saved to: {save_path}")
        
        # Show result if requested
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(annotated_image)
            plt.axis('off')
            plt.title(f"Plant Disease Detection - {Path(image_path).name}")
            plt.show()
        
        return result
    
    def detect_batch(self, input_dir, output_dir=None):
        """
        Detect plant diseases in multiple images
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save results
        """
        input_path = Path(input_dir)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            print(f"Processing: {image_file.name}")
            
            save_path = None
            if output_dir:
                save_path = output_path / f"detected_{image_file.name}"
            
            try:
                self.detect_image(image_file, save_path=save_path, show=False)
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
        
        print("Batch processing completed!")

def main():
    parser = argparse.ArgumentParser(description='Plant Disease Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained YOLO model')
    parser.add_argument('--source', type=str, required=True, help='Image file or directory path')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PlantDiseaseDetector(args.model, args.conf)
    
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Single image
        save_path = None
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"detected_{source_path.name}"
        
        detector.detect_image(source_path, save_path=save_path, show=not args.no_show)
    
    elif source_path.is_dir():
        # Batch processing
        detector.detect_batch(source_path, args.output)
    
    else:
        print(f"Error: {args.source} is not a valid file or directory")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plant Leaf Disease Detection - YOLO Training Script
This script trains a YOLOv8 model on the PlantDoc dataset for plant disease detection.
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlantDiseaseTrainer:
    def __init__(self, data_path="archive", model_size="n", epochs=100, batch_size=16, img_size=640):
        """
        Initialize the Plant Disease Trainer
        
        Args:
            data_path (str): Path to the dataset directory
            model_size (str): YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            img_size (int): Image size for training
        """
        self.data_path = Path(data_path)
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Create directories for outputs
        self.output_dir = Path("runs/train")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate dataset path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_path}")
            
        logger.info(f"Initialized trainer with model: yolov8{model_size}, epochs: {epochs}")
    
    def setup_data_yaml(self):
        """Setup and validate the data.yaml file"""
        data_yaml_path = self.data_path / "data.yaml"
        
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
        
        # Read the original data.yaml
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update paths to be absolute
        base_path = self.data_path.absolute()
        data_config['train'] = str(base_path / "train" / "images")
        data_config['val'] = str(base_path / "valid" / "images") 
        data_config['test'] = str(base_path / "test" / "images")
        
        # Create updated data.yaml
        updated_yaml_path = "data_config.yaml"
        with open(updated_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"Created updated data config: {updated_yaml_path}")
        logger.info(f"Number of classes: {data_config['nc']}")
        logger.info(f"Classes: {data_config['names']}")
        
        return updated_yaml_path, data_config
    
    def validate_dataset(self):
        """Validate that all required directories and files exist"""
        required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels", "test/images", "test/labels"]
        
        for dir_path in required_dirs:
            full_path = self.data_path / dir_path
            if not full_path.exists():
                logger.warning(f"Directory not found: {full_path}")
            else:
                file_count = len(list(full_path.glob("*")))
                logger.info(f"{dir_path}: {file_count} files")
    
    def train_model(self):
        """Train the YOLO model"""
        try:
            # Setup data configuration
            data_yaml_path, data_config = self.setup_data_yaml()
            
            # Validate dataset
            self.validate_dataset()
            
            # Initialize YOLO model
            model_name = f"yolov8{self.model_size}.pt"
            logger.info(f"Loading model: {model_name}")
            model = YOLO(model_name)
            
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            # Training arguments
            train_args = {
                'data': data_yaml_path,
                'epochs': self.epochs,
                'imgsz': self.img_size,
                'batch': self.batch_size,
                'device': device,
                'workers': 4,
                'patience': 50,
                'save': True,
                'save_period': 10,
                'cache': False,
                'optimizer': 'auto',
                'verbose': True,
                'seed': 42,
                'deterministic': True,
                'single_cls': False,
                'rect': False,
                'cos_lr': False,
                'close_mosaic': 10,
                'resume': False,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'multi_scale': False,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'split': 'val',
                'save_json': False,
                'save_hybrid': False,
                'conf': None,
                'iou': 0.7,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'plots': True,
                'source': None,
                'vid_stride': 1,
                'stream_buffer': False,
                'visualize': False,
                'augment': False,
                'agnostic_nms': False,
                'classes': None,
                'retina_masks': False,
                'embed': None,
                'project': 'runs/detect',
                'name': f'plant_disease_yolov8{self.model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            }
            
            logger.info("Starting training...")
            logger.info(f"Training arguments: {train_args}")
            
            # Train the model
            results = model.train(**train_args)
            
            # Save the trained model with a custom name
            model_save_path = f"plant_disease_yolov8{self.model_size}_trained.pt"
            model.save(model_save_path)
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Best model saved as: {model_save_path}")
            logger.info(f"Training results: {results}")
            
            return results, model_save_path
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate_model(self, model_path):
        """Evaluate the trained model"""
        try:
            logger.info("Starting model evaluation...")
            model = YOLO(model_path)
            
            # Validate on test set
            data_yaml_path = "data_config.yaml"
            results = model.val(data=data_yaml_path, split='test')
            
            logger.info("Evaluation completed!")
            logger.info(f"Evaluation results: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for plant disease detection')
    parser.add_argument('--data', type=str, default='archive', help='Dataset directory path')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], 
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model after training')
    
    args = parser.parse_args()
    
    logger.info("="*50)
    logger.info("Plant Leaf Disease Detection - YOLO Training")
    logger.info("="*50)
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Model: YOLOv8{args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Image size: {args.img_size}")
    
    try:
        # Initialize trainer
        trainer = PlantDiseaseTrainer(
            data_path=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size
        )
        
        # Train model
        results, model_path = trainer.train_model()
        
        # Evaluate if requested
        if args.evaluate:
            trainer.evaluate_model(model_path)
        
        logger.info("="*50)
        logger.info("Training completed successfully!")
        logger.info(f"Trained model saved as: {model_path}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

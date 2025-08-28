#!/usr/bin/env python3
"""
Demo script for Plant Disease Detection Web App
Creates a demo with test images from the dataset
"""

import shutil
import random
from pathlib import Path

def create_demo_images():
    """Copy some test images for demo purposes"""
    
    # Create demo directory
    demo_dir = Path("demo_images")
    demo_dir.mkdir(exist_ok=True)
    
    # Source directories
    test_images_dir = Path("archive/test/images")
    
    if not test_images_dir.exists():
        print("Test images directory not found!")
        return
    
    # Get all test images
    image_files = list(test_images_dir.glob("*.jpg"))
    
    if len(image_files) == 0:
        print("No test images found!")
        return
    
    # Select 10 random images for demo
    demo_images = random.sample(image_files, min(10, len(image_files)))
    
    print(f"Creating demo with {len(demo_images)} images...")
    
    for i, img_path in enumerate(demo_images, 1):
        # Copy to demo directory with simpler names
        dest_path = demo_dir / f"demo_{i:02d}.jpg"
        shutil.copy2(img_path, dest_path)
        print(f"  ✓ {dest_path.name}")
    
    print(f"\nDemo images created in '{demo_dir}' directory")
    print("You can use these images to test the web application!")

def print_usage_instructions():
    """Print instructions for using the web app"""
    
    print("\n" + "="*60)
    print("🌱 Plant Disease Detection Web App - Instructions")
    print("="*60)
    print()
    print("1. Start the web application:")
    print("   run_web_app.bat")
    print()
    print("2. Open your browser and go to:")
    print("   http://localhost:8000")
    print()
    print("3. Upload an image:")
    print("   - Click 'Choose Image' or drag & drop")
    print("   - Use images from 'demo_images' folder")
    print("   - Or use your own plant images")
    print()
    print("4. View results:")
    print("   - See detected diseases with bounding boxes")
    print("   - Get treatment recommendations")
    print("   - Check severity levels")
    print()
    print("🎯 Features:")
    print("  ✅ Real-time AI detection")
    print("  ✅ 30+ disease types")
    print("  ✅ Treatment recommendations") 
    print("  ✅ Beautiful, responsive UI")
    print("  ✅ Mobile-friendly")
    print()
    print("📱 API Endpoints:")
    print("  • POST /predict/ - Upload image for analysis")
    print("  • GET /health - Check API health")
    print("  • GET /model/info - Get model information")
    print("  • GET /docs - Interactive API documentation")
    print()

if __name__ == "__main__":
    print("Setting up Plant Disease Detection Demo...")
    
    # Create demo images
    create_demo_images()
    
    # Print instructions
    print_usage_instructions()
    
    print("Setup complete! Run 'run_web_app.bat' to start the web application.")

#!/usr/bin/env python3
"""
API Client Example for Plant Disease Detection
Test the FastAPI endpoints programmatically
"""

import requests
import json
from pathlib import Path
import time

class PlantDiseaseAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"error": "API not running. Please start the web app first."}
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_image(self, image_path):
        """Send image for disease prediction"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/predict/", files=files)
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def batch_predict(self, image_dir, max_images=5):
        """Predict multiple images"""
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg"))[:max_images]
        
        results = []
        for img_path in image_files:
            print(f"Analyzing {img_path.name}...")
            result = self.predict_image(img_path)
            result['filename'] = img_path.name
            results.append(result)
            time.sleep(1)  # Be nice to the API
        
        return results

def run_demo():
    """Run a complete API demo"""
    print("🌱 Plant Disease Detection API Demo")
    print("="*50)
    
    # Initialize client
    client = PlantDiseaseAPIClient()
    
    # 1. Health check
    print("\n1. Checking API health...")
    health = client.health_check()
    print(f"   Status: {health}")
    
    if "error" in health:
        print("❌ API not running! Please start the web app first:")
        print("   run_web_app.bat")
        return
    
    # 2. Model info
    print("\n2. Getting model information...")
    model_info = client.get_model_info()
    if "error" not in model_info:
        print(f"   Model: {model_info.get('model_type', 'Unknown')}")
        print(f"   Classes: {model_info.get('num_classes', 0)}")
    else:
        print(f"   Error: {model_info['error']}")
    
    # 3. Test prediction
    print("\n3. Testing image prediction...")
    
    # Check for demo images
    demo_dir = Path("demo_images")
    if demo_dir.exists():
        image_files = list(demo_dir.glob("*.jpg"))
        if image_files:
            test_image = image_files[0]
            print(f"   Using: {test_image}")
            
            result = client.predict_image(test_image)
            if "error" not in result and result.get("success"):
                summary = result["summary"]
                print(f"   ✅ Success!")
                print(f"   📊 Detections: {summary['total_detections']}")
                print(f"   🦠 Diseased: {summary['diseased_count']}")
                print(f"   🌿 Healthy: {summary['healthy_count']}")
                
                # Show detected diseases
                detections = result["detections"]
                if detections:
                    print("\n   🔍 Detected diseases:")
                    for det in detections:
                        if det["is_diseased"]:
                            print(f"      • {det['class_name']} ({det['confidence']:.2f}) - {det['severity']}")
                else:
                    print("   🎉 No diseases detected!")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print("   ⚠️ No demo images found. Run 'python setup_demo.py' first.")
    else:
        print("   ⚠️ Demo directory not found. Run 'python setup_demo.py' first.")
    
    # 4. Batch prediction demo
    if demo_dir.exists() and list(demo_dir.glob("*.jpg")):
        print("\n4. Running batch prediction demo...")
        batch_results = client.batch_predict(demo_dir, max_images=3)
        
        total_detections = sum(r.get("summary", {}).get("total_detections", 0) for r in batch_results if "error" not in r)
        total_diseased = sum(r.get("summary", {}).get("diseased_count", 0) for r in batch_results if "error" not in r)
        
        print(f"   📈 Batch Results:")
        print(f"      Images processed: {len(batch_results)}")
        print(f"      Total detections: {total_detections}")
        print(f"      Total diseased areas: {total_diseased}")
    
    print("\n" + "="*50)
    print("🎯 Demo Complete!")
    print("\nNext steps:")
    print("• Open http://localhost:8000 in your browser")
    print("• Try the interactive web interface")
    print("• Upload your own plant images")
    print("• Check out the API docs at http://localhost:8000/docs")

def test_api_endpoints():
    """Test all API endpoints"""
    client = PlantDiseaseAPIClient()
    
    print("Testing API Endpoints...")
    
    # Test health endpoint
    health = client.health_check()
    print(f"Health: {'✅' if 'error' not in health else '❌'}")
    
    # Test model info endpoint
    model_info = client.get_model_info()
    print(f"Model Info: {'✅' if 'error' not in model_info else '❌'}")
    
    # Test prediction endpoint (if demo images exist)
    demo_dir = Path("demo_images")
    if demo_dir.exists() and list(demo_dir.glob("*.jpg")):
        test_image = list(demo_dir.glob("*.jpg"))[0]
        result = client.predict_image(test_image)
        print(f"Prediction: {'✅' if result.get('success') else '❌'}")
    else:
        print("Prediction: ⚠️ (No test images)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_api_endpoints()
    else:
        run_demo()

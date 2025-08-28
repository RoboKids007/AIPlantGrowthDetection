# Plant Leaf Disease Detection with YOLO
GowthamSetup.bat

# Plant Leaf Disease Detection with YOLO

This project implements a plant leaf disease detection system using YOLOv8 (You Only Look Once) deep learning model. The system can identify 30 different types of plant diseases and healthy leaves across 13 plant species.

## 📋 Dataset Information

- **Dataset**: PlantDoc Dataset (Roboflow format)
- **Classes**: 30 (29 disease types + healthy leaves)
- **Plant Species**: 13 (Apple, Bell Pepper, Blueberry, Cherry, Corn, Peach, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato, Grape)
- **Total Images**: ~2,569 images with 8,851 labels
- **Format**: YOLO format (images + text annotation files)

### Classes Include:
- **Apple**: Scab Leaf, Healthy Leaf, Rust Leaf
- **Bell Pepper**: Leaf Spot, Healthy Leaf
- **Corn**: Gray Leaf Spot, Leaf Blight, Rust Leaf
- **Potato**: Early Blight, Late Blight, Healthy Leaf
- **Tomato**: Early Blight, Septoria Leaf Spot, Bacterial Spot, Late Blight, Mosaic Virus, Yellow Virus, Healthy Leaf, Mold Leaf, Two Spotted Spider Mites
- **And more...**

## 🚀 Quick Start

### 1. Setup Environment (with GPU support)

**Option A - Automatic Setup (Recommended for Windows):**
```cmd
run_setup_and_train.bat
```
This will automatically detect your GPU and install the appropriate PyTorch version.

**Option B - Manual GPU Setup:**
```cmd
setup_gpu.bat
```
This script will specifically set up CUDA-enabled PyTorch for your NVIDIA GPU.

**Option C - Manual Setup:**
```bash
python setup.py
```

### 2. Verify GPU Setup

Test if your GPU is properly configured:
```bash
python test_gpu.py
```

This will verify:
- ✅ PyTorch installation with CUDA support
- ✅ GPU detection and memory info  
- ✅ GPU tensor operations
- ✅ YOLO model GPU compatibility
- 🚀 Performance benchmark vs CPU

### 2. Train the Model

#### Basic Training (Recommended for beginners):
```bash
python main.py --model n --epochs 50 --batch-size 16
```

#### Advanced Training:
```bash
python main.py --model s --epochs 100 --batch-size 32 --img-size 640 --evaluate
```

#### GPU Training (if available):
```bash
python main.py --model m --epochs 200 --batch-size 64
```

## 🌐 Web Application (NEW!)

### **🚀 Beautiful Web Interface**
```cmd
# Setup and run the web application
run_web_app.bat
```

Then open your browser to: **http://localhost:8000**

### **✨ Web App Features:**
- 🎨 **Beautiful, Modern UI** - Responsive design that works on all devices
- 📤 **Drag & Drop Upload** - Easy image upload with drag and drop
- ⚡ **Real-time Analysis** - Instant AI-powered disease detection  
- 📊 **Detailed Results** - Visual detection with bounding boxes
- 💊 **Treatment Advice** - Specific recommendations for each disease
- 📱 **Mobile Friendly** - Works perfectly on phones and tablets
- 🔄 **Interactive** - Switch between original and annotated images

### **🎯 Demo Setup:**
```cmd
# Create demo images and instructions
python setup_demo.py

# Test the API programmatically  
python api_demo.py
```

### **📡 API Endpoints:**
- `POST /predict/` - Upload image for analysis
- `GET /health` - Check API health
- `GET /model/info` - Get model information
- `GET /docs` - Interactive API documentation (Swagger UI)

### **🔧 API Usage Example:**
```python
import requests

# Upload image for prediction
with open('plant_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict/', files=files)
    result = response.json()
    
print(f"Detected {result['summary']['total_detections']} issues")
```

## 📁 Project Structure

```
PlantLeafDetections/
├── archive/                    # Dataset directory
│   ├── data.yaml              # Dataset configuration
│   ├── train/                 # Training data
│   │   ├── images/
│   │   └── labels/
│   ├── valid/                 # Validation data
│   │   ├── images/
│   │   └── labels/
│   └── test/                  # Test data
│       ├── images/
│       └── labels/
├── main.py                    # Main training script
├── inference.py               # Inference script
├── setup.py                   # Environment setup script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── runs/                      # Training outputs
    └── train/                 # Training results
```

## ⚙️ Configuration Options

### Model Sizes
- `n` (nano): Fastest, smallest model (~3.2M parameters)
- `s` (small): Good balance (~11.2M parameters)  
- `m` (medium): Better accuracy (~25.9M parameters)
- `l` (large): High accuracy (~43.7M parameters)
- `x` (xlarge): Best accuracy (~68.2M parameters)

### Training Parameters
```bash
python main.py [OPTIONS]

Options:
  --data TEXT          Dataset directory path [default: archive]
  --model TEXT         Model size (n/s/m/l/x) [default: n]
  --epochs INTEGER     Number of training epochs [default: 100]
  --batch-size INTEGER Batch size [default: 16]
  --img-size INTEGER   Image size [default: 640]
  --evaluate          Evaluate model after training
  --help              Show help message
```

### Inference Parameters
```bash
python inference.py [OPTIONS]

Options:
  --model TEXT        Path to trained YOLO model [required]
  --source TEXT       Image file or directory path [required]
  --output TEXT       Output directory for results
  --conf FLOAT        Confidence threshold [default: 0.25]
  --no-show          Do not display results
  --help             Show help message
```

## 📊 Expected Training Results

### Performance Metrics (Approximate)
- **YOLOv8n**: mAP50 ~0.75-0.85, Training time ~2-3 hours (GPU)
- **YOLOv8s**: mAP50 ~0.80-0.90, Training time ~3-4 hours (GPU)
- **YOLOv8m**: mAP50 ~0.85-0.92, Training time ~4-6 hours (GPU)

### Training Outputs
- Trained model: `plant_disease_yolov8{size}_trained.pt`
- Training logs: `training.log`
- Results: `runs/train/plant_disease_yolov8{size}_{timestamp}/`
- Plots: Loss curves, precision-recall curves, confusion matrix

## 🖥️ System Requirements

### Minimum Requirements:
- Python 3.8+
- 8GB RAM
- 5GB free disk space
- CPU training supported

### Recommended Requirements:
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 6GB+ VRAM
- CUDA 11.0+
- 10GB+ free disk space

## 📦 Dependencies

Core packages installed automatically:
- `ultralytics` - YOLOv8 implementation
- `torch` - PyTorch deep learning framework
- `opencv-python` - Computer vision library
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical computations
- `pillow` - Image processing
- `pyyaml` - YAML file handling

## 🔧 Troubleshooting

### Common Issues:

1. **Out of Memory Error**:
   - Reduce batch size: `--batch-size 8`
   - Use smaller model: `--model n`
   - Reduce image size: `--img-size 416`

2. **Slow Training**:
   - Enable GPU if available
   - Increase batch size: `--batch-size 32`
   - Use mixed precision training (automatic in YOLOv8)

3. **Low Accuracy**:
   - Increase training epochs: `--epochs 200`
   - Use larger model: `--model s` or `--model m`
   - Check dataset quality and labels

4. **Import Errors**:
   - Run `python setup.py` to install dependencies
   - Check Python version (3.8+ required)

### GPU Setup:
```bash
# Check if GPU is available
python test_gpu.py

# Setup GPU-enabled PyTorch (Windows)
setup_gpu.bat

# Or install manually
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Expected GPU Performance:
With your **RTX 4090**, you should see:
- **Training speed**: 5-10x faster than CPU
- **Batch size**: Can use 32-64 (vs 8-16 on CPU)
- **Memory usage**: ~8-12GB VRAM for YOLOv8m
- **Training time**: 30-60 minutes for 100 epochs (vs 5-10 hours on CPU)

## 📈 Model Usage Examples

### Python Script Usage:
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('plant_disease_yolov8n_trained.pt')

# Run inference
results = model('path/to/image.jpg')

# Print results
for result in results:
    boxes = result.boxes
    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}")
```

### Advanced Inference:
```python
# Batch prediction with custom confidence
results = model(['img1.jpg', 'img2.jpg'], conf=0.3)

# Save results
for i, result in enumerate(results):
    result.save(f'result_{i}.jpg')
```

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PlantDoc Dataset Paper](https://arxiv.org/pdf/1911.10317.pdf)
- [Original PlantDoc GitHub](https://github.com/pratikkayal/PlantDoc-Dataset)

## 📄 License

This project uses the PlantDoc dataset under CC BY 4.0 license. Please cite the original authors:

```bibtex
@misc{singh2019plantdoc,
    title={PlantDoc: A Dataset for Visual Plant Disease Detection},
    author={Davinder Singh and Naman Jain and Pranjali Jain and Pratik Kayal and Sudhakar Kumawat and Nipun Batra},
    year={2019},
    eprint={1911.10317},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

---

**Happy Training! 🌱🔍**


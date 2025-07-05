# Chessboard Vision Project

## 📌 Introduction

This project is part of the **Computer Vision** course in the Master's in Artificial Intelligence. The goal is to detect chess pieces on a board from images, determine their positions using bounding boxes, and represent them in an 8x8 matrix format. The solution is implemented in Python and follows a structured image processing pipeline.

## 🎯 Task Overview

- **Input:** Chessboard image
- **Output:**
  - Total number of black/white pieces
  - Bounding boxes around detected pieces
  - 8x8 matrix representing piece positions
- **Dataset:** 50 images for development, 10 unseen test images
- **Deliverables:**
  - Python script (single file)
  - Short report (max 2 pages)

## 🛠️ Workflow

1. **Preprocessing:** Load and preprocess the image (grayscale conversion, thresholding, etc.).
2. **Chessboard Detection:** Identify the board and correct for perspective distortion.
3. **Piece Detection:** Use image processing techniques or machine learning to detect pieces.
4. **Bounding Box Extraction:** Identify the coordinates of detected pieces.
5. **8x8 Board Mapping:** Assign pieces to board squares and generate a matrix representation.
6. **JSON Output:** Structure the results as per the project’s requirements.
7. **Evaluation & Testing:** Validate with sample images and test dataset.

# Chessboard Vision Project

## 📌 Introduction

This project is part of the **Computer Vision** course in the Master's in Artificial Intelligence. The goal is to detect chess pieces on a board from images using various computer vision and machine learning approaches. The project is divided into three main tasks, each with increasing complexity and different methodological approaches.

## 🎯 Task Overview

### Task 1: Traditional Computer Vision Approach

- **Input:** Chessboard image
- **Output:**
  - Total number of black/white pieces
  - Bounding boxes around detected pieces
  - 8x8 matrix representing piece positions (0/1 values)
- **Method:** Traditional image processing techniques
- **Dataset:** 50 development images, 10 test images

### Task 2: CNN-based Piece Counting

- **Input:** Chess game image
- **Output:** Total number of pieces within the chess board
- **Method:** CNN-based architectures (ResNet, YOLO)
- **Extra:** Quantitative comparison between different architectures

### Task 3: Advanced Object Detection

- **Chess Pieces Detection:** YOLO, Faster R-CNN models
- **Board Digital Twin:** Complete board state identification (piece type and color)
- **Method:** Object detection + traditional methods from Task 1
- **Dataset:** [Chess Pieces Dataset](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f)

## 📁 Project Structure

### Core Implementation Files

- **`chessboard-vision.py`** - Main script for Task 1 (traditional CV approach)
- **`chessboard_vision_2.py`** - Enhanced version with improved algorithms
- **`chessboard_vision_2_test.py`** - Testing and validation script

### Jupyter Notebooks

- **`chessboard-vision.ipynb`** - Initial development and experimentation for Task 1
- **`chessboard-vision-final.ipynb`** - Final implementation and results for Task 1
- **`chessboard-task2-resnet.ipynb`** - ResNet-based CNN implementation for Task 2
- **`chessboard-task2-yolo.ipynb`** - YOLO-based approach for Task 2
- **`chessboard-task3.ipynb`** - Advanced object detection for Task 3
- **`chess_table.ipynb`** - Chess piece classification and board analysis
- **`new_try.ipynb`** - Experimental approaches and alternative methods

### Data and Configuration

- **`images/`** - Sample dataset with 50 chess board images
- **`input.json`** - Input configuration file specifying image paths
- **`requirements.txt`** - Python dependencies
- **`docs/VC_2425_Project.pdf`** - Project specification document

## 🛠️ Implementation Workflow

1. **Image Preprocessing:** Load, resize, and enhance image quality
2. **Chessboard Detection:** Identify board boundaries and apply perspective correction
3. **Piece Segmentation:** Detect and isolate chess pieces using color/texture analysis
4. **Bounding Box Generation:** Extract precise coordinates for each detected piece
5. **Board Mapping:** Map detected pieces to 8x8 grid positions
6. **Classification:** Distinguish between black and white pieces (and piece types for Task 3)
7. **Output Generation:** Create JSON results with counts, positions, and matrices

## 🚀 How to Run the Project

### Quick Start

1. **Clone the Repository:**

```bash
git clone <repository-url>
cd chessboard-vision
```

2. **Set up Environment:**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

### Running Different Tasks

#### Task 1: Traditional Computer Vision

```bash
# Using the main script
python chessboard-vision.py

# Using the enhanced version
python chessboard_vision_2.py input.json output.json

# Testing version with detailed visualizations
python chessboard_vision_2_test.py input.json
```

#### Task 2: CNN-based Approaches

```bash
# Open and run the respective notebooks
jupyter notebook chessboard-task2-resnet.ipynb
jupyter notebook chessboard-task2-yolo.ipynb
```

#### Task 3: Advanced Object Detection

```bash
# Open the Task 3 notebook
jupyter notebook chessboard-task3.ipynb
```

### Input Configuration

Create an `input.json` file with the following structure:

```json
{
  "image_files": [
    "images/G000_IMG062.jpg",
    "images/G000_IMG087.jpg",
    "images/G006_IMG048.jpg"
  ]
}
```

### Output Format

The scripts generate `output.json` with the following structure:

```json
[
  {
    "image": "path/to/image.jpg",
    "num_pieces": 24,
    "board": [[0,1,0,1,0,1,0,1], ...],
    "detected_pieces": [
      {"xmin": 100, "ymin": 150, "xmax": 140, "ymax": 190},
      ...
    ]
  }
]
```

## 📊 Results and Evaluation

### Task 1 Results

- **Chessboard Detection:** Perspective correction using traditional CV methods
- **Piece Segmentation:** Color-based segmentation with morphological operations
- **Accuracy:** Evaluated on 50 development images + 10 test images

### Task 2 Results

- **ResNet Approach:** CNN-based piece counting with transfer learning
- **YOLO Approach:** Real-time object detection for piece counting
- **Comparison:** Quantitative metrics (accuracy, precision, recall, F1-score)

### Task 3 Results

- **Object Detection:** YOLO/Faster R-CNN for piece type and color classification
- **Digital Twin:** Complete board state reconstruction
- **Evaluation:** Qualitative analysis with success/failure cases

## 🔧 Technical Details

### Dependencies

- OpenCV (cv2) - Image processing
- NumPy - Numerical operations
- Matplotlib - Visualization
- TensorFlow/PyTorch - Deep learning models
- JSON - Data serialization

### Key Algorithms

- **Perspective Correction:** Homography transformation
- **Color Segmentation:** HSV color space filtering
- **Morphological Operations:** Opening, closing, erosion, dilation
- **Contour Detection:** Connected component analysis
- **CNN Architectures:** ResNet, YOLO, Faster R-CNN

## 📝 Development Notes

### Notebook Descriptions

- **`chessboard-vision.ipynb`** - Prototype development with step-by-step visualization
- **`chessboard-vision-final.ipynb`** - Clean implementation with final results
- **`chessboard-task2-resnet.ipynb`** - ResNet implementation with training pipeline
- **`chessboard-task2-yolo.ipynb`** - YOLO model training and inference
- **`chessboard-task3.ipynb`** - Advanced detection with piece classification
- **`chess_table.ipynb`** - Board analysis and piece mapping experiments
- **`new_try.ipynb`** - Alternative approaches and experimental methods

### File Organization

- Scripts ending with `_test.py` include detailed visualizations and debugging
- Notebooks are organized by task number for easy navigation
- All image processing utilities are modularized in class-based structure

## 🎯 Future Improvements

- Real-time video processing
- Enhanced piece type classification
- Robustness to lighting variations
- Multi-view board reconstruction
- Integration with chess engines

## 📚 References

- [Chess Pieces Dataset](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f)
- Course materials: `docs/VC_2425_Project.pdf`
- OpenCV Documentation
- YOLO and Faster R-CNN papers

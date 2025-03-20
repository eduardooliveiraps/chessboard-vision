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

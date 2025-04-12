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

```
chessboard-vision
├─ chessboard-vision-2.ipynb
├─ chessboard-vision.ipynb
├─ chessboard-vision.py
├─ docs
│  └─ VC_2425_Project.pdf
├─ images
│  ├─ G000_IMG062.jpg
│  ├─ G000_IMG087.jpg
│  ├─ G000_IMG102.jpg
│  ├─ G006_IMG048.jpg
│  ├─ G006_IMG086.jpg
│  ├─ G006_IMG119.jpg
│  ├─ G019_IMG082.jpg
│  ├─ G028_IMG015.jpg
│  ├─ G028_IMG062.jpg
│  ├─ G028_IMG098.jpg
│  ├─ G028_IMG101.jpg
│  ├─ G033_IMG043.jpg
│  ├─ G033_IMG075.jpg
│  ├─ G033_IMG088.jpg
│  ├─ G033_IMG101.jpg
│  ├─ G038_IMG074.jpg
│  ├─ G038_IMG088.jpg
│  ├─ G038_IMG103.jpg
│  ├─ G038_IMG105.jpg
│  ├─ G041_IMG042.jpg
│  ├─ G041_IMG048.jpg
│  ├─ G041_IMG088.jpg
│  ├─ G041_IMG098.jpg
│  ├─ G047_IMG053.jpg
│  ├─ G047_IMG068.jpg
│  ├─ G047_IMG102.jpg
│  ├─ G047_IMG107.jpg
│  ├─ G056_IMG017.jpg
│  ├─ G056_IMG077.jpg
│  ├─ G056_IMG097.jpg
│  ├─ G058_IMG044.jpg
│  ├─ G058_IMG074.jpg
│  ├─ G058_IMG100.jpg
│  ├─ G061_IMG080.jpg
│  ├─ G061_IMG092.jpg
│  ├─ G061_IMG098.jpg
│  ├─ G072_IMG083.jpg
│  ├─ G072_IMG098.jpg
│  ├─ G076_IMG072.jpg
│  ├─ G076_IMG089.jpg
│  ├─ G076_IMG095.jpg
│  ├─ G078_IMG092.jpg
│  ├─ G083_IMG073.jpg
│  ├─ G083_IMG089.jpg
│  ├─ G087_IMG093.jpg
│  ├─ G087_IMG099.jpg
│  ├─ G091_IMG053.jpg
│  ├─ G091_IMG074.jpg
│  ├─ G091_IMG102.jpg
│  └─ G099_IMG094.jpg
└─ README.md

```
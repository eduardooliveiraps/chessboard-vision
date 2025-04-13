# Chess Image Processing System
import numpy as np
import cv2
import os
import json
import sys

class ChessboardDetector:
    """Base class for detecting and processing chessboards from images"""
    
    def __init__(self, window_width=800, window_height=600):
        """Initialize with display window parameters"""
        self.window_width = window_width
        self.window_height = window_height
    
    def resize_image(self, image):
        """Resize image to fit display window"""
        height, width = image.shape[:2]
        scaling_factor = min(self.window_width / width, self.window_height / height)
        return cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    def detect_chessboard(self, image):
        """Detect chessboard contour in an image"""
        #Kernel
        kernel = np.ones((5, 5), np.uint8)
        # Aplly blur
        blurred = cv2.GaussianBlur(image, (3, 3), 3)
        
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Interval to black (matrix: 0-180, saturation: 0-255, value: 0-100)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 100])
        
        # Interval to white (matix: 0-130, saturation: 0-14, value: 200-255)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([130, 14, 255])

        # Create the masks (black,white) 
        mask_black = cv2.inRange(hsv_img, lower_black, upper_black)
        mask_white = cv2.inRange(hsv_img, lower_white, upper_white)

        #Combine both and aplly the morphology Closing
        mask = cv2.bitwise_or(mask_black, mask_white)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        #aplly the mask to the image
        result = cv2.bitwise_and(blurred, blurred, mask=mask)
        
        #aplly gray scale
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        #aplly dilate in gray
        gray_dilate = cv2.dilate(gray,kernel,iterations = 6)

        chessboard_contour = None
        
         # Find chessboard contour
        _, thresh_sobel = cv2.threshold(gray_dilate, 50, 255, cv2.THRESH_BINARY )
        contours, _ = cv2.findContours(thresh_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort by area, get largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:25]
                
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            for approx_param in [0.01, 0.02, 0.03, 0.04, 0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
                approx = cv2.approxPolyDP(contour, approx_param * peri, True)
                
                # Check if it's a quadrilateral with appropriate area
                if len(approx) == 4 and cv2.contourArea(approx) > (image.shape[0] * image.shape[1]) * 0.15:
                    chessboard_contour = approx
                    return chessboard_contour
        
        return None
    
    def get_perspective_transform(self, contour, image):
        """Calculate perspective transform matrices from contour"""
        if contour is None:
            return None, None, None, None
            
        # Calculate transform parameters
        rect = np.zeros((4, 2), dtype="float32")
        pts = contour.reshape(4, 2)
        
        # Sort corners - top to bottom
        y_sorted = pts[np.argsort(pts[:, 1]), :] 
        top_two = y_sorted[:2]
        bottom_two = y_sorted[2:]
        
        # Sort left to right
        top_left = top_two[np.argmin(top_two[:, 0])]
        top_right = top_two[np.argmax(top_two[:, 0])]
        bottom_left = bottom_two[np.argmin(bottom_two[:, 0])]
        bottom_right = bottom_two[np.argmax(bottom_two[:, 0])]
        
        # Assign corners
        rect[0] = top_left
        rect[1] = top_right
        rect[2] = bottom_right
        rect[3] = bottom_left
        
        # Calculate dimensions
        width_top = np.sqrt(((rect[0][0] - rect[1][0])**2) + ((rect[0][1] - rect[1][1])**2))
        width_bottom = np.sqrt(((rect[2][0] - rect[3][0])**2) + ((rect[2][1] - rect[3][1])**2))
        width = int(max(width_top, width_bottom))
        
        height_left = np.sqrt(((rect[0][0] - rect[3][0])**2) + ((rect[0][1] - rect[3][1])**2))
        height_right = np.sqrt(((rect[1][0] - rect[2][0])**2) + ((rect[1][1] - rect[2][1])**2))
        height = int(max(height_left, height_right))
        
        # Force square if dimensions are close
        if abs(width - height) < min(width, height) * 0.2:
            size = max(width, height)
            width, height = size, size
        
        # Define destination points
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Calculate transform matrices
        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = cv2.getPerspectiveTransform(dst, rect)
        
        return M, M_inv, width, height
        
    def extract_chessboard(self, image_path):
        """Detect and extract a chessboard from an image with perspective correction"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Detect chessboard contour
        contour = self.detect_chessboard(img)
        if contour is None:
            return None, None, None
        
        # Get perspective transform
        M, M_inv, width, height = self.get_perspective_transform(contour, img)
        if M is None:
            return None, None, None
        
        # Apply transform
        warped = cv2.warpPerspective(img, M, (width, height))
        
        # Check if transform failed
        if np.all(warped == warped[0, 0]):
            # Try fallback method
            pts = contour.reshape(4, 2)
            s = pts.sum(axis=1)
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            M_inv = cv2.getPerspectiveTransform(dst, rect)
            warped = cv2.warpPerspective(img, M, (width, height))
            
            if np.all(warped == warped[0, 0]):
                return None, None, None
        
        return warped, M, M_inv


class ChessPieceDetector(ChessboardDetector):
    """Detect chess pieces in a chessboard image"""
    
    def detect_pieces(self, warped):
        """Detect both black and white chess pieces"""
        if warped is None:
            return None, None
            
        blurred = cv2.GaussianBlur(warped, (5, 5), 0)

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Create masks for both colors at once
        # Black pieces - dark with low saturation
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([255, 80, 30])
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        
        # White pieces
        white_lower = np.array([17, 50, 80])
        white_upper = np.array([40, 155, 215]) 
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Apply morphological operations to clean up the masks
        kernel_open_white = np.ones((17, 17), np.uint8)
        kernel_close_white = np.ones((16, 16), np.uint8)
        kernel_open_black = np.ones((3, 3), np.uint8)
        kernel_close_black = np.ones((15, 15), np.uint8)
        
        # Process black mask
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_close_black)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_open_black)
        
        # Process white mask
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close_white)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_open_white)
        white_mask = cv2.erode(white_mask,np.ones((4,1),np.uint8),iterations = 2)
        
        return black_mask, white_mask


class ChessboardAnalyzer:
    """Analyze chess positions from warped images"""
    
    def __init__(self, h_margin=170, v_margin=130):
        """Initialize with grid margin parameters"""
        self.h_margin = h_margin
        self.v_margin = v_margin
        
    def create_board_matrix(self, warped_image, black_mask, white_mask):
        """Create an 8x8 matrix representing the chessboard state"""
        height, width = warped_image.shape[:2]
        
        # Calculate grid cell dimensions
        cell_width = (width - 2 * self.h_margin) / 8
        cell_height = (height - 2 * self.v_margin) / 8
        
        # Initialize the chessboard matrix
        board_matrix = np.zeros((8, 8), dtype=int)
        
        # Function to determine which cell a point belongs to
        def get_cell_position(x, y):
            if (x < self.h_margin or x >= width - self.h_margin or 
                y < self.v_margin or y >= height - self.v_margin):
                return None  # Outside the chessboard grid
            
            col = int((x - self.h_margin) // cell_width)
            row = int((y - self.v_margin) // cell_height)
            
            # Ensure we're within bounds (due to rounding)
            if 0 <= row < 8 and 0 <= col < 8:
                return (row, col)
            else:
                return None
        
        # Distance from corner to use as reference point
        corner_margin = 38 # Pixels from corner
        
        # Process black and white pieces
        for mask in [black_mask, white_mask]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= 4500:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.25 <= aspect_ratio <= 2.0:
                        # Define reference point (bottom-left corner + margin)
                        ref_x = x + w/2
                        ref_y = y + h - corner_margin
                        
                        cell = get_cell_position(ref_x, ref_y)
                        if cell:
                            row, col = cell
                            board_matrix[row][col] = 1
        
        return board_matrix
        
    def map_to_original_coords(self, black_mask, white_mask, M_inv):
        """Map chess piece coordinates from warped image to original image"""
        # Results list
        pieces = []
        
        # Process both masks
        for mask in [black_mask, white_mask]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= 4500:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.25 <= aspect_ratio <= 2.0:
                        # Define the four corners of the bounding box in the warped image
                        warped_corners = np.array([
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]
                        ], dtype=np.float32)
                        
                        # Reshape for perspective transform
                        warped_corners = warped_corners.reshape(-1, 1, 2)
                        
                        # Apply inverse perspective transform to get coordinates in original image
                        original_corners = cv2.perspectiveTransform(warped_corners, M_inv)
                        original_corners = original_corners.reshape(4, 2)
                        
                        # Calculate the bounding box in original image
                        x_values = original_corners[:, 0]
                        y_values = original_corners[:, 1]
                        xmin, ymin = int(min(x_values)), int(min(y_values))
                        xmax, ymax = int(max(x_values)), int(max(y_values))
                        
                        # Add piece to results
                        pieces.append({
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax
                        })
        
        return pieces


class ChessImageProcessor:
    """Main class for processing chess images"""
    
    def __init__(self, h_margin=170, v_margin=130):
        """Initialize with parameters"""
        self.chessboard_detector = ChessboardDetector()
        self.piece_detector = ChessPieceDetector()
        self.analyzer = ChessboardAnalyzer(h_margin, v_margin)
        
    def save_visualization(self, image_path, step_name, image):
        """Save visualization results for a specific step"""
        # Create directory structure
        base_dir = "visualize_results"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # Extract image filename without path and extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.join(base_dir, image_name)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        # Save the image
        output_path = os.path.join(image_dir, f"{step_name}.jpg")
        cv2.imwrite(output_path, image)
        
    def draw_grid_on_warped(self, warped_image, h_margin, v_margin):
        """Draw grid lines on the warped image to visualize cell alignment"""
        grid_image = warped_image.copy()
        height, width = grid_image.shape[:2]
        
        # Calculate grid cell dimensions
        cell_width = (width - 2 * h_margin) / 8
        cell_height = (height - 2 * v_margin) / 8
        
        # Draw horizontal grid lines
        for i in range(9):
            y = int(v_margin + i * cell_height)
            cv2.line(grid_image, (h_margin, y), (width - h_margin, y), (0, 255, 0), 2)
        
        # Draw vertical grid lines
        for i in range(9):
            x = int(h_margin + i * cell_width)
            cv2.line(grid_image, (x, v_margin), (x, height - v_margin), (0, 255, 0), 2)
        
        return grid_image
    
    def draw_bounding_boxes(self, original_image, mapped_pieces):
        """Draw bounding boxes on original image to visualize piece detection"""
        image_with_boxes = original_image.copy()
        
        for piece in mapped_pieces:
            xmin = piece["xmin"]
            ymin = piece["ymin"]
            xmax = piece["xmax"]
            ymax = piece["ymax"]
            
            # Draw rectangle
            cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        return image_with_boxes
    
    def visualize_board_matrix(self, board_matrix, warped_image, h_margin, v_margin):
        """Visualize the board matrix mapping on the warped image"""
        vis_image = warped_image.copy()
        height, width = vis_image.shape[:2]
        
        # Calculate grid cell dimensions
        cell_width = (width - 2 * h_margin) / 8
        cell_height = (height - 2 * v_margin) / 8
        
        # Draw grid
        grid_image = self.draw_grid_on_warped(vis_image, h_margin, v_margin)
        
        # Mark cells with pieces
        for row in range(8):
            for col in range(8):
                if board_matrix[row][col] == 1:
                    # Calculate cell center
                    center_x = int(h_margin + col * cell_width + cell_width / 2)
                    center_y = int(v_margin + row * cell_height + cell_height / 2)
                    
                    # Draw a circle to mark the cell
                    cv2.circle(grid_image, (center_x, center_y), 15, (0, 0, 255), -1)
        
        return grid_image
    
    def process_image(self, image_path):
        """Process a single image through the entire pipeline"""
        if image_path is None:
            raise ValueError("Image path must be specified")
            
        print(f"Processing image: {image_path}")
        
        # Load the original image for visualization
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Step 1: Extract chessboard
        warped, M, M_inv = self.chessboard_detector.extract_chessboard(image_path)
        if warped is None:
            print(f"Failed to detect chessboard in {image_path}")
            return None
        
        # Save the warped image visualization
        self.save_visualization(image_path, "1_chessboard_detection", warped)
        
        # Step 2: Draw grid on warped image
        grid_image = self.draw_grid_on_warped(warped, self.analyzer.h_margin, self.analyzer.v_margin)
        self.save_visualization(image_path, "2_grid_drawing_alignment", grid_image)
        
        # Step 3: Detect chess pieces
        black_mask, white_mask = self.piece_detector.detect_pieces(warped)
        if black_mask is None:
            print(f"Failed to detect pieces in {image_path}")
            return None
        
        # Save piece segmentation masks
        # Combine black and white masks for visualization
        combined_mask = cv2.cvtColor(black_mask.copy(), cv2.COLOR_GRAY2BGR)
        # Color white mask as red
        white_mask_colored = cv2.cvtColor(white_mask.copy(), cv2.COLOR_GRAY2BGR)
        white_mask_colored[:,:,2] = white_mask  # Red channel
        combined_mask = cv2.addWeighted(combined_mask, 1, white_mask_colored, 1, 0)
        self.save_visualization(image_path, "3_piece_segmentation", combined_mask)
        
        # Step 4: Create board matrix
        board_matrix = self.analyzer.create_board_matrix(warped, black_mask, white_mask)
        
        # Visualize board matrix
        board_vis = self.visualize_board_matrix(board_matrix, warped, self.analyzer.h_margin, self.analyzer.v_margin)
        self.save_visualization(image_path, "5_correct_board_mapping", board_vis)
        
        # Step 5: Map to original coordinates
        mapped_pieces = self.analyzer.map_to_original_coords(black_mask, white_mask, M_inv)
        
        # Draw bounding boxes on original image
        boxed_image = self.draw_bounding_boxes(original_image, mapped_pieces)
        self.save_visualization(image_path, "4_bounding_box_accuracy", boxed_image)
        
        # Step 6: Prepare and return the result
        result = {
            "image": image_path,
            "num_pieces": len(mapped_pieces),
            "board": board_matrix.tolist(),
            "detected_pieces": mapped_pieces
        }
        
        # Save the matrix as text file for better visualization
        matrix_dir = os.path.join("visualize_results", os.path.splitext(os.path.basename(image_path))[0])
        with open(os.path.join(matrix_dir, "board_matrix.txt"), "w") as f:
            for row in board_matrix:
                f.write(" ".join(map(str, row)) + "\n")
        
        return result


def process_json_input(input_file, output_file):
    """Process images from input JSON file and save results to output JSON file"""
    try:
        # Read input JSON
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Initialize processor
        processor = ChessImageProcessor()
        
        # Process each image
        results = []
        for image_path in input_data.get("image_files", []):
            result = processor.process_image(image_path)
            if result is not None:
                results.append(result)
        
        # Write output JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Processing complete. Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing JSON input: {e}")
        return False
    
    return True

def folder_to_json(folder):
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
    data = {
        "image_files": image_files
    }
    with open("input.json", "w") as f:
        json.dump(data, f, indent=2)
    return True


if __name__ == "__main__":
    # If arguments are provided, use them as input and output files
    if len(sys.argv) > 1:
        if(sys.argv[1].endswith('.json')):
            input_file = sys.argv[1]
            try:
                output_file = sys.argv[2]
            except IndexError:
                output_file = 'output.json'
        else:
            if(os.path.isdir(sys.argv[1])):
                dir = sys.argv[1]
            else:
                dir = 'images'
            folder_to_json(dir)
            input_file = 'input.json'
            output_file = 'output.json'
    
    process_json_input(input_file, output_file)
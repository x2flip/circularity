import cv2
import numpy as np

def detect_irregularities_by_shape(edges, circularity_threshold=0.8, min_contour_area=10):
    regular_contours = []
    irregular_contours = []
    regular_circularities = []
    irregular_circularities = []
    regular_indices = []
    irregular_indices = []
    problematic_contour_idx = None

    # Morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Avoid division by zero and filter small contours
        if perimeter == 0 or area < min_contour_area:
            continue

        # Calculate circularity
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # Debugging: Print out contour information
        print(f"Contour {idx}: Area = {area}, Perimeter = {perimeter}, Circularity = {circularity}")
        print(f"Contour {idx} Points: {contour}")

        if circularity < circularity_threshold:
            if circularity < 0.1 and problematic_contour_idx is None:
                problematic_contour_idx = len(irregular_contours)
            irregular_contours.append(contour)
            irregular_circularities.append(circularity)
            irregular_indices.append(idx)
        else:
            regular_contours.append(contour)
            regular_circularities.append(circularity)
            regular_indices.append(idx)

    return (regular_contours, regular_circularities, regular_indices), (irregular_contours, irregular_circularities, irregular_indices), problematic_contour_idx

def draw_contours(image, contours, circularities, indices, color):
    for idx, (contour, circularity, original_idx) in enumerate(zip(contours, circularities, indices)):
        cv2.drawContours(image, [contour], -1, color, 1)
        # Compute the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Label the contour with its original index and circularity
            cv2.putText(image, f"{original_idx}: {circularity:.3f}", (cX - 20, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def enhance_contrast(image): 
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    return enhanced_image

def main(image_path='test.jpg', output_path='marked.jpg', circularity_threshold=0.8, min_contour_area=10):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    enhanced_image = enhance_contrast(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bilateral_filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(bilateral_filtered, (3, 3), 1)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect irregularities based on circularity
    (regular_contours, regular_circularities, regular_indices), (irregular_contours, irregular_circularities, irregular_indices), problematic_contour_idx = detect_irregularities_by_shape(edges, circularity_threshold, min_contour_area)
    
    # Highlight and label regular contours in green
    draw_contours(image, regular_contours, regular_circularities, regular_indices, (0, 255, 0))
    
    # Highlight and label irregular contours in red
    draw_contours(image, irregular_contours, irregular_circularities, irregular_indices, (0, 0, 255))
    
    # Highlight the problematic contour in blue for visualization
    if problematic_contour_idx is not None and problematic_contour_idx < len(irregular_contours):
        cv2.drawContours(image, [irregular_contours[problematic_contour_idx]], -1, (255, 0, 0), 2)
        print(f"Problematic Contour {problematic_contour_idx} Points: {irregular_contours[problematic_contour_idx]}")
    else:
        print("No problematic contour detected.")

    # Display the result
    cv2.imshow("Detected Irregularities", image)
    cv2.imshow("Enhanced Image", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    cv2.imwrite(output_path, image)

    # Log the circularities
    print("Regular contours circularities:")
    for idx, circularity in enumerate(regular_circularities):
        print(f"Contour {regular_indices[idx]}: {circularity:.2f}")

    print("\nIrregular contours circularities:")
    for idx, circularity in enumerate(irregular_circularities):
        print(f"Contour {irregular_indices[idx]}: {circularity:.2f}")

# Example usage with display scaling
main(image_path='test.jpg', output_path='marked.jpg', circularity_threshold=0.864, min_contour_area=10)


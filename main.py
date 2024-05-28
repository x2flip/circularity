import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_irregularities_by_shape(edges, circularity_threshold=0.8, min_contour_area=10, perimeter_threshold=50.0, area_threshold=0.0):
    regular_contours = []
    irregular_contours = []
    regular_circularities = []
    irregular_circularities = []
    regular_perimeters = []
    irregular_perimeters = []
    regular_areas = []
    irregular_areas = []
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

        if perimeter < perimeter_threshold and circularity < circularity_threshold:
            irregular_contours.append(contour)
            irregular_circularities.append(circularity)
            irregular_indices.append(idx)
            irregular_perimeters.append(perimeter)
            irregular_areas.append(area)
        else:
            regular_contours.append(contour)
            regular_circularities.append(circularity)
            regular_indices.append(idx)
            regular_perimeters.append(perimeter)
            regular_areas.append(area)

    return (regular_contours, regular_circularities, regular_indices, regular_perimeters, regular_areas), (irregular_contours, irregular_circularities, irregular_indices, irregular_perimeters, irregular_areas), problematic_contour_idx

def draw_contours(image, contours, circularities, indices, color):
    for idx, (contour, circularity, original_idx) in enumerate(zip(contours, circularities, indices)):
        cv2.drawContours(image, [contour], -1, color, 1)
        # Compute the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Label the contour with its original index and circularity
            cv2.putText(image, f"{original_idx}", (cX - 20, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def enhance_contrast(image): 
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    return enhanced_image

def main(image_path='test.jpg', output_path='marked.jpg', circularity_threshold=0.8, min_contour_area=10, perimeter_threshold=49.0, threshold_output='threshold.jpg', area_threshold=0.0):
    # Load the image
    image = cv2.imread(image_path, -1)
    if image is None:
        print("Error: Could not load image.")
        return

    image = cv2.convertScaleAbs(image, alpha=(255/65535))

    #enhanced_image = enhance_contrast(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur1 = cv2.GaussianBlur(gray, (1, 1), 1)
    blur2 = cv2.GaussianBlur(gray, (3, 3), 1)
    blur3 = cv2.GaussianBlur(gray, (5, 5), 1)
    blur4 = cv2.GaussianBlur(gray, (7, 7), 1)

    # Global Threshold
    #ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    # Otsu's thresholding
    #ret4,th4 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
    # Otsu's thresholding after Gaussian filtering
    ret1,th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret2,th2 = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret3,th3 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret4,th4 = cv2.threshold(blur3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret5,th5 = cv2.threshold(blur4,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Histogram
    #plt.hist(gray.ravel(),256,(0,256)); 

    #bilateral_filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    
    # Apply Canny edge detection
    edges1 = cv2.Canny(th1, 0, 1000)
    edges2 = cv2.Canny(th2, 50, 150)
    edges3 = cv2.Canny(th3, 50, 150)
    edges4 = cv2.Canny(th4, 50, 150)
    edges5 = cv2.Canny(th5, 50, 150)
    
    # Detect irregularities based on circularity
    (regular_contours, 
     regular_circularities, 
     regular_indices, 
     regular_perimeters,
     regular_areas
     ), (irregular_contours, 
         irregular_circularities, 
         irregular_indices, 
         irregular_perimeters,
         irregular_areas), problematic_contour_idx = detect_irregularities_by_shape(
                 edges1, 
                 circularity_threshold, 
                 min_contour_area, 
                 perimeter_threshold=perimeter_threshold,
                 area_threshold=area_threshold
                 )
    
    # Highlight and label regular contours in green
    draw_contours(image, regular_contours, regular_circularities, regular_indices, (0, 255, 0))
    
    # Highlight and label irregular contours in red
    draw_contours(image, irregular_contours, irregular_circularities, irregular_indices, (0, 0, 255))

    # Highlight and label regular contours in green
    draw_contours(th5, regular_contours, regular_circularities, regular_indices, (0, 255, 0))
    
    # Highlight and label irregular contours in red
    draw_contours(th5, irregular_contours, irregular_circularities, irregular_indices, (0, 0, 255))
    
    # Highlight the problematic contour in blue for visualization
    if problematic_contour_idx is not None and problematic_contour_idx < len(irregular_contours):
        cv2.drawContours(image, [irregular_contours[problematic_contour_idx]], -1, (255, 0, 0), 2)
        print(f"Problematic Contour {problematic_contour_idx} Points: {irregular_contours[problematic_contour_idx]}")
    else:
        print("No problematic contour detected.")


    # Log the circularities
    # print("Regular contours circularities:")
    # for idx, circularity in enumerate(regular_circularities):
    #     print(f"Contour {regular_indices[idx]}: {circularity:.2f}")

    # print("\nIrregular contours circularities:")
    # for idx, circularity in enumerate(irregular_circularities):
    #     print(f"Contour {irregular_indices[idx]}: Circularity: {circularity:.3f}")

    for idx, perimeter in enumerate(irregular_perimeters):
        print(f"Irregular Perimeter: Contour {irregular_indices[idx]}: \nPerimeter: {perimeter:.10f} \nCircularity: {irregular_circularities[idx]:.10f} \nArea: {irregular_areas[idx]:.10f}\n\n")

    sortedPerims = regular_perimeters.sort()
    for i in range(10):
        print(f"Index {regular_indices[i]} Perimeter {regular_perimeters[i]}")


    # Display the result
    cv2.imshow("Thresholded Image", th5)
    cv2.imshow("Edges 1 Image", edges1)
    cv2.imshow("Edges 2 Image", edges2)
    cv2.imshow("Edges 3 Image", edges3)
    cv2.imshow("Edges 4 Image", edges4)
    cv2.imshow("Edges 5 Image", edges5)
    cv2.imshow("Detected Irregularities", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    cv2.imwrite(output_path, image)
    cv2.imwrite(threshold_output, th5)
# Example usage with display scaling
main(image_path='high.tif', output_path='markedhigh.jpg', circularity_threshold=0.88, min_contour_area=10, perimeter_threshold=127.3, area_threshold=0.0)


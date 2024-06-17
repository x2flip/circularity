import cv2
import numpy as np

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
        cv2.drawContours(image, [contour], -1, color, 3)
        # Compute the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Label the contour with its original index and circularity
            cv2.putText(image, f"{original_idx}", (cX - 20, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def main(image_path='test.jpg', output_path='marked.jpg', circularity_threshold=0.8, min_contour_area=10, perimeter_threshold=49.0, threshold_output='threshold.jpg', area_threshold=0.0):
    # Load the image
    image = cv2.imread(image_path, -1)
    if image is None:
        print("Error: Could not load image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Display the result
    cv2.imshow("Grayscale result", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply Gaussian blur
    # blur1 = cv2.GaussianBlur(gray, (1, 1), 1)

    # Otsu's thresholding after Gaussian filtering
    # Only doing grayscale for now
    ret1,thresholded = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Display the result
    cv2.imshow("Thresholded Result", thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply Canny edge detection
    edges = cv2.Canny(thresholded, 20, 150)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply morphological opening to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
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
                 edges, 
                 circularity_threshold, 
                 min_contour_area, 
                 perimeter_threshold=perimeter_threshold,
                 area_threshold=area_threshold
                 )
    
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


    for idx, _ in enumerate(irregular_contours):
        print(f"Irregular contour at Contour index: {irregular_contours[idx]}")

    for idx, perimeter in enumerate(irregular_perimeters):
        print(f"Irregular Perimeter: Contour {irregular_indices[idx]}: \nPerimeter: {perimeter:.10f} \nCircularity: {irregular_circularities[idx]:.10f} \nArea: {irregular_areas[idx]:.10f}\n\n")



    # Display the result
    cv2.imshow("Detected Irregularities", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    cv2.imwrite(output_path, image)
# Example usage with display scaling
main(image_path='pics/Test_100MP.TIF', output_path='finished/100MP_Finished.jpg', circularity_threshold=0.88, min_contour_area=10, perimeter_threshold=127.3, area_threshold=0.0)


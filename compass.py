import cv2
import numpy as np
from PIL import ImageGrab
import time
import math

#IMPORTANT: MAKE SURE THE COMPASS MIDDLE IS AT THE VERY CENTER !!!
#The accuracy is heavily dependent on the initial area
#TO EXIT THE PROGRAM PRESS ESC
DEBUG = True

# Step 1: Capture the screen and select the compass area
print("--- Taking initial screenshot for ROI selection --- ")
full_screenshot = np.array(ImageGrab.grab())  # Capture the full screen
# Convert to BGR for OpenCV and to display correct colours when selecting ROI
full_screenshot = cv2.cvtColor(full_screenshot, cv2.COLOR_RGB2BGR)  

print("Please select the compass area on the screen.")
roi = cv2.selectROI("Select Compass Area", full_screenshot, showCrosshair=True)
print(roi)
cv2.destroyWindow("Select Compass Area")  # Close the ROI selection window

# Save the ROI coordinates for later use
box_x, box_y, box_w, box_h = roi
time.sleep(1)

# Define the range for red color in HSV
lower_red1 = np.array([0, 100, 100])  # Lower bound of red
upper_red1 = np.array([10, 255, 255])  # Upper bound of red
lower_red2 = np.array([170, 100, 100])  # Lower bound for the second range of red
upper_red2 = np.array([180, 255, 255])  # Upper bound for the second range of red

blue_low = np.array([80, 100, 100])   
blue_up = np.array([140, 255, 255])

lower_white = np.array([10, 0, 200])  # Lower bound of white
upper_white = np.array([255, 50, 255])  # Upper bound of white


while True:
    # grab region of interest
    compass_image = np.array(ImageGrab.grab(bbox=(box_x, box_y, box_x + box_w, box_y + box_h)))
    # Convert to BGR for OpenCV
    compass_image_BGR = cv2.cvtColor(compass_image, cv2.COLOR_RGB2BGR)
    
    # Convert the image to HSV color space to detect red
    
    compass_image_hsv = cv2.cvtColor(compass_image_BGR, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv", hsv)
    
    # Threshold the image to get only the red pixels
    mask1 = cv2.inRange(compass_image_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(compass_image_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    blue_mask = cv2.inRange(compass_image_hsv, blue_low, blue_up)  
    white_mask = cv2.inRange(compass_image_hsv, lower_white, upper_white)  
    
    #---------- RED ----------
    # Find contours in the red mask
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #If no contours found, exit
    if not red_contours:
        print("No red regions found!")
        exit()
    
    # Find the largest contour based on area
    largest_red_contour = max(red_contours, key=cv2.contourArea)
    
    
    #---------- BLUE ----------
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not blue_contours:
        print("No blue regions found!")
        exit()
    
    largest_blue_contour = max(blue_contours, key=cv2.contourArea)
    
    
    #---------- WHITE ----------
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not white_contours:
        print("No white regions found!")
        #exit()
    
    largest_white_contour = max(white_contours, key=cv2.contourArea)

    # Display DEBUG images
    if DEBUG == True:
        
        #cv2.imshow("Compass Image", compass_image_BGR)
        #cv2.waitKey(1) 
        
        #---------- RED ----------
        # Create a mask for the largest red cluster
        largest_red_cluster_image = np.zeros_like(compass_image)
        # Draw the largest contour on the image (in red)
        cv2.drawContours(largest_red_cluster_image, [largest_red_contour], -1, (0, 0, 255), thickness=cv2.FILLED)
        # Show the red pixels only in the detected area of the largest cluster
        #cv2.namedWindow("Largest Red Cluster", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("Largest Red Cluster", 100, 100)
        #cv2.imshow("Largest Red Cluster", largest_red_cluster_image)
        #cv2.waitKey(1) 
        
        #---------- BLUE ----------
        # Find contours in the red mask
        #Create an image to show only the largest cluster of blue pixels
        largest_blue_cluster_image = np.zeros_like(compass_image)
        
        # Draw the largest contour on the image (in blue)
        cv2.drawContours(largest_blue_cluster_image, [largest_blue_contour], -1, (255, 0, 0), thickness=cv2.FILLED)
        
        # Show the image with the largest blue cluster
        #cv2.namedWindow("Largest Blue Cluster", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("Largest Blue Cluster", 100, 100)
        #cv2.imshow("Largest Blue Cluster", largest_blue_cluster_image)
        #cv2.waitKey(1) 
        
        #Create an image to show only the largest cluster of blue pixels
        largest_white_cluster_image = np.zeros_like(compass_image)
        
        # Draw the largest contour on the image (in blue)
        cv2.drawContours(largest_white_cluster_image, [largest_white_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        
        all_images = cv2.hconcat([compass_image_BGR, largest_blue_cluster_image, largest_red_cluster_image])
        cv2.imshow("BGR, Largest blue cluster, Largest red cluster, Largest white cluster", all_images)
        cv2.waitKey(1)
        
    
    # Find the longest distance between red and blue tip points
    max_distance = 0
    red_tip_point = (0, 0)
    blue_tip_point = (0, 0)
    mid_point = (0,0)
    
    
    for red_point in largest_red_contour:
        for blue_point in largest_blue_contour:
            red_x, red_y = red_point[0]
            blue_x, blue_y = blue_point[0]
            distance = np.sqrt((blue_x - red_x) ** 2 + (blue_y - red_y) ** 2)  # Euclidean distance
            if distance > max_distance:
                max_distance = distance
                red_tip_point = (red_x, red_y)
                blue_tip_point = (blue_x, blue_y)
                
                # Draw a line and display the current distance
                if DEBUG == True:
                    time.sleep(0.1)
                    temp_image = compass_image_BGR.copy()  # Create a copy of the image to overlay
                    cv2.line(temp_image, (red_x, red_y), (blue_x, blue_y), (255, 255, 0), 2)  # Cyan line
                    
                    cv2.namedWindow("Compass Image with Distance", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Compass Image with Distance", 200, 200)
                    cv2.imshow("Compass Image with Distance", temp_image)
                    cv2.waitKey(1)  # Small delay to allow visualization
                    
    #print("blue_tip_point:", blue_tip_point, "red_tip_point: ", red_tip_point)
    # Findig mid point for angle visualization
    #mid_point = ((red_tip_point[0] + blue_tip_point[0]) / 2, (red_tip_point[1] + blue_tip_point[1]) / 2)
    mid_point = ((0.33*red_tip_point[0] + 0.67*blue_tip_point[0]), (0.33*red_tip_point[1] + 0.67*blue_tip_point[1]))
    mid_point = (int(mid_point[0]), int(mid_point[1]))
            
    # Calculate the angle of the tip relative to the center using arctan2
    dx = red_tip_point[0] - blue_tip_point[0] 
    dy = red_tip_point[1] - blue_tip_point[1]
    

    # Calculate the angle in radians
    angle_rad = np.arctan2(dy, dx)

    # Convert the angle to degrees
    angle_deg = -np.degrees(angle_rad)

    # Normalize the angle to be between -180 and 180 degrees
    #if angle_deg > 180:
    #    angle_deg -= 360  # Normalize to [-180, 180]

    # Print the angle of the tip relative to the center
    print(f"Angle: {angle_deg:.2f} degrees")
    
    # Drawing an arrow
    end_x = int(blue_tip_point[0] + 20 * math.cos(angle_rad))
    end_y = int(blue_tip_point[1] + 20 * math.sin(angle_rad)) # y-axis is inverted in image coordinates
    arrowed_image = compass_image_BGR.copy()
    
    cv2.circle(arrowed_image, mid_point, 2, (255, 255, 0), -1) 
    
    
    height, width = red_mask.shape
    center = width // 2, height // 2
    radius = 7  # Radius of the semicircle
    
    # Draw the full semicircle as a base
    #cv2.ellipse(arrowed_image, center, (radius, radius), 0, 0, 180, (200, 200, 200), 2)
    """if angle_deg > 0:
        start_angle = 0
        end_angle = angle_deg
    else:
        start_angle = 180 + angle_deg
        end_angle = 180"""
    start_angle = 0
    end_angle = angle_deg
    cv2.ellipse(arrowed_image, (mid_point), (radius, radius), 0, start_angle, -end_angle, (255, 255, 255), 2)
    
    angle_text = f"{angle_deg:.2f} deg"
    text_position = (8, 12)  # Position of the text (x, y)
    font_scale = 0.4
    font_color = (255, 255, 255)  # White color
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(arrowed_image, angle_text, text_position, font, font_scale, font_color, font_thickness)
    
    cv2.arrowedLine(arrowed_image, (blue_tip_point), (red_tip_point), (0, 255, 0), 1, tipLength=0.3)
    arrowed_image = cv2.resize(arrowed_image, (200, 200))
    cv2.imshow("Arrowed Compass", arrowed_image)
    cv2.waitKey(1)


    # Check for exit key (e.g., ESC key)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break the loop
        break
    
    time.sleep(1)

# Wait for a key press and then close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()

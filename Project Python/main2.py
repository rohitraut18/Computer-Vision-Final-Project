import cv2
import numpy as np
import threading

def pedestrian_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        
        resized_frame = cv2.resize(frame, (480, 320))
        found, _ = hog.detectMultiScale(resized_frame, winStride=(8, 8), padding=(0, 0), scale=1.05)
        
        for (x, y, w, h) in found:
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        cv2.imshow("Pedestrian Detection", resized_frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def lane_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred_frame, 50, 150)
        
        # Define region of interest (ROI)
        mask = np.zeros_like(edges)
        height, width = edges.shape
        polygon = np.array([[(0, height), (width, height), (width // 2, height // 2)]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        
        # Apply the mask to extract the region of interest
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Apply Hough Transform to detect lines
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        
        # Draw lines on the frame
        line_image = np.zeros_like(resized_frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Combine the lines with the original frame
        lane_frame = cv2.addWeighted(resized_frame, 0.8, line_image, 1, 0)
        
        # Show the result
        cv2.imshow("Lane Detection", lane_frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def stop_light_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (480, 320))
        
        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        
        # Define the lower and upper bounds for red color (stop light color)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        
        # Create a mask using the color threshold
        mask = cv2.inRange(hsv_frame, lower_red, upper_red)
        
        # Apply a bit of morphology to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw rectangles around detected stop lights
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Adjust the area threshold based on your scene
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(resized_frame, "STOP LIGHT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Show the result
        cv2.imshow("Stop Light Detection", resized_frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def stop_sign_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        
        resized_frame = cv2.resize(frame, (480, 320))
        
        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        
        # Define the lower and upper bounds for red color (stop sign color)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        
        # Create a binary mask for red color
        mask = cv2.inRange(hsv_frame, lower_red, upper_red)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if the polygon has 8 vertices (stop sign)
            if len(approx) == 8:
                cv2.drawContours(resized_frame, [approx], 0, (0, 255, 0), 2)
                cv2.putText(resized_frame, "STOP SIGN", (approx[0][0][0], approx[0][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Stop Sign Detection", resized_frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_paths = {
        "p": "pedestrian_video.mov",
        "l": "lane_video.mov",
        "sl": "stop_light_video.mov",
        "ss": "stop_sign_video.mov",
    }
    
    threads = []
    
    for key, video_path in video_paths.items():
        thread = threading.Thread(target=eval(key+"_detection"), args=(video_path,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()

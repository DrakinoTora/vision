import cv2
import numpy as np

def ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([1, 80, 20])
    upper_orange = np.array([25, 255, 255])

    mask_ball = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((15, 15), np.uint8)
    mask_ball = cv2.morphologyEx(mask_ball, cv2.MORPH_CLOSE, kernel)

    return mask_ball

def field(frame):
    low_green = np.array([30, 30, 45])
    up_green = np.array([85, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green_mask = cv2.inRange(hsv, low_green, up_green)
    green_mask1 = cv2.erode(green_mask, kernel, iterations=1)
    green_mask2 = cv2.dilate(green_mask1, kernel, iterations=6)

    contours, _ = cv2.findContours(green_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return frame, np.zeros_like(green_mask2)

    largest_contour = max(contours, key=cv2.contourArea)
    
    hull = cv2.convexHull(largest_contour)

    mask = np.zeros_like(green_mask2)
    cv2.drawContours(mask, [hull], -1, 255, thickness=-1)
    
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return roi_frame, mask

def find_first_last_orange(line_data, line_start):
    first_orange = None
    last_orange = None
    
    for idx in range(line_data.shape[0]): 
        if line_data[idx] == 255:
            first_orange = line_start + idx 
            break
    
    for idx in range(line_data.shape[0] - 1, -1, -1):  
        if line_data[idx] == 255:
            last_orange = line_start + idx 
            break
    
    return first_orange, last_orange

def find_top_bottom_orange(column_data, col_start):
    top_orange = None
    bot_orange = None
    
    for idy in range(column_data.shape[0]): 
        if column_data[idy] == 255:
            top_orange = col_start + idy 
            break
    
    for idy in range(column_data.shape[0] - 1, -1, -1):  
        if column_data[idy] == 255:
            bot_orange = col_start + idy 
            break
    
    return top_orange, bot_orange

def detect(mask_ball, mask_field, frame):
    blurred_ball_mask = cv2.GaussianBlur(mask_ball, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred_ball_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=1,
        maxRadius=1000
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            stroke = int(1.1 * r)
            
            line_y = y
            line_x_start = max(0, x - stroke)
            line_x_end = min(mask_ball.shape[1] - 1, x + stroke)
            orange_hline = mask_ball[line_y, line_x_start:line_x_end]
            first_orange, last_orange = find_first_last_orange(orange_hline, line_x_start)
            
            if first_orange is None or last_orange is None:
                continue
            
            total_x_pixel = last_orange - first_orange
            r_new = int(total_x_pixel / 2)
            x_new = first_orange + r_new
            
            line_x = x_new
            line_y_start = int(y) - int(stroke)
            line_y_end = int(y) + int(stroke)
            orange_vline = mask_ball[line_y_start:line_y_end, line_x]
            top_orange, bot_orange = find_top_bottom_orange(orange_vline, line_y_start)
            
            if top_orange is None or bot_orange is None:
                continue
            
            total_y_pixel = abs(top_orange - bot_orange)
            y_new = bot_orange - int(total_y_pixel / 2)
            
            if y_new != y:
                line_y = y_new
                orange_hline = mask_ball[line_y, line_x_start:line_x_end]
                first_orange, last_orange = find_first_last_orange(orange_hline, line_x_start)
                
                if first_orange is None or last_orange is None:
                    continue
                
                total_x_pixel = last_orange - first_orange
                r_new = int(total_x_pixel / 2)
                x_new = first_orange + r_new

            R = int(r_new * 1.5)
            
            x1, y1 = max(x_new - R, 0), max(y_new - R, 0)
            x2, y2 = min(x_new + R, frame.shape[1]), min(y_new + R, frame.shape[0])

            surrounding_field = mask_field[y1:y2, x1:x2]
            field_ratio = np.sum(surrounding_field == 255) / surrounding_field.size

            surrounding_ball = mask_ball[y1:y2, x1:x2]
            ball_ratio = np.sum(surrounding_ball == 255) / surrounding_ball.size

            if (field_ratio > 0.16 and ball_ratio < 0.47):
                cv2.line(frame, (x_new, y_new + r_new), (x_new, y_new - r_new), (0, 255, 0), 2)
                cv2.line(frame, (x_new - r_new, y_new), (x_new + r_new, y_new), (0, 255, 0), 2)
                cv2.circle(frame, (x_new, y_new), r_new, (0, 255, 0), 2)
            else:
                continue

            actual_diameter= 0.13 #meter
            focal_length= 714.1
            detected_diameter= total_x_pixel
            if detected_diameter==0:
                distance=0
            else:
                distance= (actual_diameter*focal_length)/detected_diameter
            text2 = f"Distance: {distance:.2f} meter"
            cv2.putText(frame, text2, (x_new - r_new, y_new + r_new + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            #print("distance= %.3f cm" %(distance*100))
            cv2.line(frame, (center_x, center_y), (x_new, y_new), (255, 255, 255), 2)

            break

    return frame

#cap = cv2.VideoCapture(r"D:\GMRT\Altair\vision\Program deteksi bola\sample3.mp4")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No Video Opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    y_start = 0
    y_end = height
    x_start = 0
    x_end = width

    cv2.line(frame, (center_x, y_start), (center_x, y_end), (255, 255, 255), 1)
    cv2.line(frame, (x_start, center_y), (x_end, center_y), (255, 255, 255), 1)

    seg_field, mask_field = field(frame)
    mask_ball = ball(seg_field)
    final_frame = detect(mask_ball, mask_field, frame)

    if final_frame is not None:
        cv2.imshow('ball detect', final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

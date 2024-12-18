import cv2
import numpy as np

def segment(frame):
    low_green = np.array([30, 100, 45])
    up_green = np.array([85, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green_mask = cv2.inRange(hsv, low_green, up_green)
    green_mask1 = cv2.erode(green_mask, kernel, iterations=1)
    green_mask2 = cv2.dilate(green_mask1, kernel, iterations=6)

    contours, _ = cv2.findContours(green_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)

    mask = np.zeros_like(green_mask2)
    cv2.drawContours(mask, [hull], -1, 255, thickness=-1)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return roi_frame

def calculate_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1 + 1e-6)

def is_parallel_and_close(line1, line2, slope_threshold=0.1, distance_threshold=10):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    slope1 = calculate_slope(x1, y1, x2, y2)
    slope2 = calculate_slope(x3, y3, x4, y4)

    if abs(slope1 - slope2) > slope_threshold:
        return False

    dist = abs((y3 - slope1 * x3 - (y1 - slope1 * x1)) / np.sqrt(1 + slope1**2))
    return dist < distance_threshold

def merge_lines(lines):
    merged_lines = []

    while lines:
        line = lines.pop(0)
        x1, y1, x2, y2 = line[0]
        max_line = line
        max_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        to_remove = []
        for other_line in lines:
            x3, y3, x4, y4 = other_line[0]
            if is_parallel_and_close((x1, y1, x2, y2), (x3, y3, x4, y4)):
                length = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
                if length > max_length:
                    max_length = length
                    max_line = other_line
                to_remove.append(other_line)

        for line_to_remove in to_remove:
            lines.remove(line_to_remove)

        merged_lines.append(max_line)
    return merged_lines

def intersect(frame, lines):
    i = 0
    for line in lines:
        i += 1
        L1, T1, L2, T2 = 0, 0, 0, 0

        x1, y1, x2, y2 = line[0]
        r = 8
        #cv2.putText(frame, "P1", (x1 + r + 20, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #cv2.putText(frame, "P2", (x2 - r - 20, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text = "line" + str(i)
        #cv2.putText(frame, text, (int(0.5*x1 + 0.5*x2) - 20, int(0.5*y1 + 0.5*y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
        cv2.circle(frame, (x1, y1), 3, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 3, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        x11, y11 = max(x1 - r, 0), max(y1 - r, 0)
        x12, y12 = min(x1 + r, frame.shape[1]), min(y1 + r, frame.shape[0])
        surrounding_point1 = frame[y11:y12, x11:x12]

        x21, y21 = max(x2 - r, 0), max(y2 - r, 0)
        x22, y22 = min(x2 + r, frame.shape[1]), min(y2 + r, frame.shape[0])
        surrounding_point2 = frame[y21:y22, x21:x22]

        cv2.rectangle(frame, (x11, y11), (x12, y12), (0, 255, 255), 1)
        cv2.rectangle(frame, (x21, y21), (x22, y22), (0, 255, 255), 1)

        is_point1 = cv2.inRange(surrounding_point1, (240, 0, 0), (255, 10, 10))
        is_line1 = cv2.inRange(surrounding_point1, (0, 0, 240), (10, 10, 255))
        L1 = np.sum(is_point1)
        T1 = np.sum(is_line1)
        
        th= 1200

        if L1 > th:
            cv2.putText(frame, "L1", (x1 - r, y1 + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif T1 >th:
            cv2.putText(frame, "T1", (x1 - r, y1 + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        is_point2 = cv2.inRange(surrounding_point2, (240, 0, 0), (255, 10, 10))
        is_line2 = cv2.inRange(surrounding_point2, (0, 0, 240), (10, 10, 255))
        L2 = np.sum(is_point2)
        T2 = np.sum(is_line2)
        
        if L2 > th:
            cv2.putText(frame, "L2", (x2 - r, y2 + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif T2 > th:
            cv2.putText(frame, "T2", (x2 - r, y2 + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print(text, "point 1 =", L1, "line 1 =", T1, "point 2 =", L2, "line 2 =", T2)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (x1, y1), 3, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 3, (255, 0, 0), -1)

img = cv2.imread('line1.png')
height, width, channels = img.shape
final = np.ones((height, width, 3), dtype=np.uint8) * 0
frame = segment(img)

low_white = np.array([0, 0, 165])
up_white = np.array([300, 60, 255])

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask_white = cv2.inRange(hsv, low_white, up_white)
blurred_bin = cv2.GaussianBlur(hsv, (5, 5), 2)
edges = cv2.Canny(blurred_bin, 50, 150)

lines = cv2.HoughLinesP(mask_white, 1, np.pi/180, threshold=200, minLineLength=50, maxLineGap=10)

if lines is not None:
    merged_lines = merge_lines(lines.tolist())
    

    for line in merged_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(final, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(final, (x1, y1), 3, (255, 0, 0), -1)
        cv2.circle(final, (x2, y2), 3, (255, 0, 0), -1)
    
    intersect(final, merged_lines)

cv2.imshow('Detected Lines', final)
#cv2.imshow('canny edge', edges)
#cv2.imshow('white', mask_white)
cv2.waitKey(0)
cv2.destroyAllWindows()

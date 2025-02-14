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
    cv2.drawContours(mask, [hull], -1, 255, thickness=-1)  # Isi area dalam hull
    
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return roi_frame, mask

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

        
            line_y = y
            line_x_start = 0
            line_x_end = frame.shape[1]

            orange_hline = mask_ball[line_y, line_x_start:line_x_end]

            first_orange = None
            last_orange = None

            # Mencari indeks pertama dari piksel oranye
            for idx in range(orange_hline.size):
                if orange_hline[idx] == 255:
                    first_orange = idx
                    break

            # Mencari indeks terakhir dari piksel oranye
            for idx in range(orange_hline.size - 1, -1, -1):
                if orange_hline[idx] == 255:
                    last_orange = idx
                    break

            if last_orange is not None and first_orange is not None:
                total_x_pixel = last_orange - first_orange
            else:
                continue

            r_new = int(total_x_pixel/2)
            x_new = first_orange + int(total_x_pixel/2)

            line_x = first_orange + int(r_new /4)
            line_y_start = 0
            line_y_end = frame.shape[0]

            orange_vline = mask_ball[line_y_start:line_y_end, line_x]

            top_orange = frame.shape[0]
            bot_orange = 0

            orange_col=0   #banyak pixel bola berturut turut
            void_col=0     #banyak pixel non bola berturut turut
            for idy in range(orange_vline.size - 1, -1, -1):
                if orange_col != 0 and void_col > int(0.5*orange_col):
                    break       #bola dianggap selesai
                if orange_vline[idy] == 255:
                    orange = idy
                    orange_col += 1 + void_col
                    void_col = 0
                    bot_orange = max(orange, bot_orange)
                    top_orange = min(orange, top_orange)
                elif orange_col > 0:
                    void_col += 1

            if top_orange is not None and bot_orange is not None:
                total_y_pixel = abs(top_orange -bot_orange)
            else:
                continue

            y_new= bot_orange - int(total_y_pixel/2)

            if y_new != y:
                line_y = y_new
                orange_hline = mask_ball[line_y, line_x_start:line_x_end]

                first_orange = None
                last_orange = None

                for idx in range(orange_hline.size):
                    if orange_hline[idx] == 255:
                        first_orange = idx
                        break

                for idx in range(orange_hline.size - 1, -1, -1):
                    if orange_hline[idx] == 255:
                        last_orange = idx
                        break

                if last_orange is not None and first_orange is not None:
                    total_x_pixel = last_orange - first_orange
                else:
                    continue   
                r_new = int(total_x_pixel/2)
                x_new = first_orange + int(total_x_pixel/2)

            R = int(r_new * 1.5)  #jarak deteksi
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
            focal_length= 71.41
            detected_diameter= total_x_pixel
            if detected_diameter==0:
                actual_distance=0
            else:
                actual_distance= (actual_diameter*focal_length)/detected_diameter
            h_robot= 0
            distance = np.sqrt(actual_distance**2 - h_robot**2)
            text2 = f"Distance: {distance:.2f} meter"
            cv2.putText(frame, text2, (x_new - r_new, y_new + r_new + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.line(frame, (center_x, center_y), (x_new, y_new), (255, 255, 255), 2)

    return frame

cap = cv2.VideoCapture(r"D:\GMRT\Altair\vision\Program deteksi bola mp4\sample2.mp4")
#cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No Video Opened")
    exit()

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#fps = cap.get(cv2.CAP_PROP_FPS)
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#out = cv2.VideoWriter('test2.mp4', fourcc, fps, (width, height))

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

    # Gambar garis vertikal di tengah
    cv2.line(frame, (center_x, y_start), (center_x, y_end), (255, 255, 255), 1)
    cv2.line(frame, (x_start, center_y), (x_end, center_y), (255, 255, 255), 1)

    seg_field, mask_field = field(frame)
    mask_ball = ball(seg_field)
    #mask_ball = ball(frame)
    #final_frame = detect(mask_ball, frame, frame)
    final_frame = detect(mask_ball, mask_field, frame)

    #cv2.imshow('mask field', mask_field)
    #cv2.imshow('mask ball', mask_ball)

    if final_frame is not None:
        #out.write(final_frame)
        cv2.imshow('Video Deteksi Bola', final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#out.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

#fungsi segmentasi lapangan
def segment(frame):
    low_green = np.array([30, 100, 45])
    up_green = np.array([85, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green_mask = cv2.inRange(hsv, low_green, up_green)
    green_mask1 = cv2.erode(green_mask, kernel, iterations=1)
    green_mask2 = cv2.dilate(green_mask1, kernel, iterations=6)

    #ambil bagian terbesar yang dianggap lapangan
    contours, _ = cv2.findContours(green_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)

    mask = np.zeros_like(green_mask2)
    cv2.drawContours(mask, [hull], -1, 255, thickness=-1)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return roi_frame

def slope(x1, y1, x2, y2):
    return (y1 - y2) / (x2 - x1 + 1e-6)

#jika garis relatif sejajar dan dekat = True
def is_parallel_and_close(line1, line2, slope_threshold=0.1, distance_threshold=10):    #threshold = toleransi perbedaan
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    slope1 = slope(x1, y1, x2, y2)
    slope2 = slope(x3, y3, x4, y4)

    if abs(slope1 - slope2) > slope_threshold:
        return False

    dist = abs((y3 - slope1 * x3 - (y1 - slope1 * x1)) / np.sqrt(1 + slope1**2))
    return dist < distance_threshold

#jika ada 2 garis sejajar dan dekat, buang 1
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

def intersect(frame, field, lines):
    i = 0
    for line in lines:
        i += 1
        L1, T1, L2, T2 = 0, 0, 0, 0     #banyak pixel yang terdeteksi potongan L atau T, 1: ujung garis kiri, 2: ujung garis kanan
        bL1, bT1, bL2, bT2 = False, False, False, False     #false = tidak ada potongan L/T

        x1, y1, x2, y2 = line[0]
        r = 8   #radius deteksi intersect
 
        #cv2.putText(frame, "P1", (x1 + r + 20, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #cv2.putText(frame, "P2", (x2 - r - 20, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #text = "line " + str(i)
        #cv2.putText(frame, text, (int(0.5*x1 + 0.5*x2) - 20, int(0.5*y1 + 0.5*y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #print(text, "grad = %.3f" %(slope(x1, y1, x2, y2)), x1, y1, x2, y2)

        #ubah warna agar tidak mendeteksi diri sendiri
        cv2.circle(frame, (x1, y1), 3, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 3, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        #kotak kuning untuk cari banyak pixel titik dan garis di sekitar titik
        x11, y11 = max(x1 - r, 0), max(y1 - r, 0)
        x12, y12 = min(x1 + r, frame.shape[1]), min(y1 + r, frame.shape[0])
        surrounding_point1 = frame[y11:y12, x11:x12]

        x21, y21 = max(x2 - r, 0), max(y2 - r, 0)
        x22, y22 = min(x2 + r, frame.shape[1]), min(y2 + r, frame.shape[0])
        surrounding_point2 = frame[y21:y22, x21:x22]

        cv2.rectangle(frame, (x11, y11), (x12, y12), (0, 255, 255), 1)
        cv2.rectangle(frame, (x21, y21), (x22, y22), (0, 255, 255), 1)

        is_point1 = cv2.inRange(surrounding_point1, (240, 0, 0), (255, 10, 10)) #pixel biru = titik
        is_line1 = cv2.inRange(surrounding_point1, (0, 0, 240), (10, 10, 255))  #pixel merah = garis
        L1 = np.sum(is_point1)
        T1 = np.sum(is_line1)
        
        th= 1200    #banyak pixel minimal yang dianggap L atau T

        if L1 > th:
            cv2.putText(frame, "L1", (x1 - r, y1 + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            bL1 = True
        elif T1 >th:
            cv2.putText(frame, "T1", (x1 - r, y1 + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            bT1 = True

        is_point2 = cv2.inRange(surrounding_point2, (240, 0, 0), (255, 10, 10))
        is_line2 = cv2.inRange(surrounding_point2, (0, 0, 240), (10, 10, 255))
        L2 = np.sum(is_point2)
        T2 = np.sum(is_line2)
        
        if L2 > th:
            cv2.putText(frame, "L2", (x2 - r, y2 + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            bL2 = True
        elif T2 > th:
            cv2.putText(frame, "T2", (x2 - r, y2 + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            bT2 = True

        #cek apakah titik ada di pojok, false = tidak di pojok
        P1edge, P2edge = False, False
        ed = 50     #toleransi dianggap pojok
        if y1 < ed or y1 > frame.shape[0] - ed:
            P1edge = True
        if x1 < ed or x1 > frame.shape[1] - ed:
            P1edge = True
        if y2 < ed or y2 > frame.shape[0] - ed:
            P2edge = True
        if x2 < ed or x2 > frame.shape[1] - ed:
            P2edge = True
        
        #garis tengah adalah garis yang diapit 2 titik, (T, T) (pojok, T) (T, pojok)
        if (bT1 and bT2) or (bT1 and P2edge) or (bT2 and P1edge):
            cv2.putText(frame, "garis tengah", (int(0.5*x1 + 0.5*x2), int(0.5*y1 + 0.5*y2) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            #jika garis tengah lurus, cari X intersect: dengan cek pixel putih satu satu hehe
            grad = slope(x1, y1, x2, y2)
            if grad > -0.1 and grad < 0.1:
                white = 10e6
                b = False
                if bT1:
                    for px in range (x1, x2):
                        white_before = white
                        white = 0
                        for py in range (int((y1 + y2)/2) - 30, int((y1 + y2)/2) + 30):
                            if field[py, px] == 255:
                                white += 1

                        #print("white before =", white_before, "white =", white)
                        if white > white_before + 7:
                            cv2.circle(frame, (px, int((y1 + y2)/2)), 4, (0, 255, 0), -1)
                            center_line = frame[int((y1 + y2)/2), x1:px]
                            actual_length= 1.5 #meter, adalah panjang asli dari T ke X
                            focal_length= 2817
                            detected_length= center_line.size
                            y_distance= (actual_length*focal_length)/detected_length
                            if y_distance <= 4.5:   #jarak dari tengah lapangan ke pojok
                                text2 = f"Distance: {y_distance:.2f} meter"
                                cv2.putText(frame, text2, (px, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                x_distance = (abs(x1 - frame.shape[1]//2) / (px - x1)) * actual_length
                                x_distance = -3 + x_distance    #posisi x robot dari pusat lapangan
                                y_distance = -y_distance        #posisi y robot dari pusat lapangan
                                print(x_distance, y_distance)
                                local_point(x_distance, y_distance, width, height, resolusi)
                            b = True
                            break
                    
                        if b:
                            break
                    
                elif bT2:
                    for px in range (x2, x1, -1):
                        white_before = white
                        white = 0
                        for py in range (int((y1 + y2)/2) - 30, int((y1 + y2)/2) + 30):
                            if field[py, px] == 255:
                                white += 1

                        #print("white =", white)
                        if white > white_before + 7:
                            cv2.circle(frame, (px, int((y1 + y2)/2)), 4, (0, 255, 0), -1)
                            center_line = frame[int((y1 + y2)/2), px:x2]
                            actual_length= 1.5 #meter
                            focal_length= 2817
                            detected_length= center_line.size
                            y_distance= (actual_length*focal_length)/detected_length
                            if y_distance <= 4.5:
                                text2 = f"Distance: {y_distance:.2f} meter"
                                cv2.putText(frame, text2, (px, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                x_distance = (abs(x2 - frame.shape[1]//2) / (x2 - px)) * actual_length
                                x_distance = 3 - x_distance
                                y_distance = -y_distance
                                print(x_distance, y_distance)
                                local_point(x_distance, y_distance, width, height, resolusi)
                            b = True
                            break
                        
                        if b:
                            break
        
        #garis samping adalah garis yang diapit 2 titik, (L, L) (L, pojok) (pojok, L)
        if (bL1 and bL2) or (bL1 and P2edge) or (bL2 and P1edge):
            cv2.putText(frame, "garis samping", (int(0.5*x1 + 0.5*x2), int(0.5*y1 + 0.5*y2) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        #print(text, "point 1 =", L1, "line 1 =", T1, "point 2 =", L2, "line 2 =", T2)p
        #print("P1 edge =", P1edge, "P2 edge =", P2edge)

        #kembalikan ke warna awal
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (x1, y1), 3, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 3, (255, 0, 0), -1)

#fungsi lokalisasi
def local_point(x, y, width, height, resolusi):
    center_x, center_y = width // 2, height // 2    #titik 0, 0 lapangan (tengah)

    cv2.rectangle(canvas, (0 + resolusi, resolusi), (width - resolusi, height - resolusi), (255, 255, 255), 2)  #garis pinggir
    cv2.line(canvas, (0 + resolusi, center_y), (width - resolusi, center_y), (255, 255, 255), 2)    #garis tengah
    cv2.circle(canvas, (center_x, center_y), int(1.5*resolusi), (255, 255, 255), 2) #lingkaran tengah
    cv2.rectangle(canvas, (int(0.5*resolusi) + resolusi, 2*resolusi), (width - int(1.5*resolusi), resolusi), (255, 255, 255), 2) #kotak gawang 1
    cv2.rectangle(canvas, (int(0.5*resolusi) + resolusi, height - 2*resolusi), (width - int(1.5*resolusi), height - resolusi), (255, 255, 255), 2) #kotak gawang 2

    #posisi robot dalam meter
    x = int(x*resolusi + center_x)
    y = int(-y*resolusi + center_y)
    cv2.circle(canvas, (x, y), int(0.2 * resolusi), (0, 0, 255), -1)

img = cv2.imread('sample3.png')
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

#create canvas, buat gambar lapangan
resolusi = 50
width, height = 8, 11
width *= resolusi
height *= resolusi
canvas = np.zeros((height, width, 3), dtype=np.uint8)

if lines is not None:
    merged_lines = merge_lines(lines.tolist())
    
    for line in merged_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(final, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(final, (x1, y1), 3, (255, 0, 0), -1)
        cv2.circle(final, (x2, y2), 3, (255, 0, 0), -1)
    
    intersect(final, mask_white, merged_lines)

cv2.imshow('Canvas', canvas)
cv2.imshow('Detected Lines', final)
cv2.imshow('frame', frame)
#cv2.imshow('white', mask_white)
cv2.waitKey(0)
cv2.destroyAllWindows()

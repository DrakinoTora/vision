#include <opencv2/opencv.hpp>
#include <iostream>
#include <utility>
#include <vector>

using namespace cv;
using namespace std;

// Global variables to mimic the Python global variables (center_x, center_y)
int center_x = 0;
int center_y = 0;

Mat ball(const Mat &frame) {
    // Convert to HSV color space
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // Define lower and upper bounds for orange color
    Scalar lower_orange(1, 80, 20);
    Scalar upper_orange(25, 255, 255);

    // Threshold the HSV image to get only orange colors
    Mat mask_ball;
    inRange(hsv, lower_orange, upper_orange, mask_ball);

    // Create a kernel of ones for morphological operations
    Mat kernel = Mat::ones(15, 15, CV_8U);
    morphologyEx(mask_ball, mask_ball, MORPH_CLOSE, kernel);

    return mask_ball;
}

pair<Mat, Mat> field(const Mat &frame) {
    // Define lower and upper bounds for green color
    Scalar low_green(30, 30, 45);
    Scalar up_green(85, 255, 255);

    // Convert the frame to HSV color space
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    
    // Create an elliptical structuring element (kernel) for morphological operations
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    
    // Threshold the HSV image to get only green colors
    Mat green_mask;
    inRange(hsv, low_green, up_green, green_mask);
    
    // Perform erosion followed by dilation
    Mat green_mask1, green_mask2;
    erode(green_mask, green_mask1, kernel, Point(-1,-1), 1);
    dilate(green_mask1, green_mask2, kernel, Point(-1,-1), 6);

    // Find contours in the mask
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(green_mask2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        // If no contours found, return the original frame and an empty mask of the same size as green_mask2
        return make_pair(frame, Mat::zeros(green_mask2.size(), green_mask2.type()));
    }
    
    // Find the largest contour based on area
    double max_area = 0;
    int max_index = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_index = static_cast<int>(i);
        }
    }
    
    vector<Point> largest_contour = contours[max_index];
    
    // Compute the convex hull of the largest contour
    vector<Point> hull;
    convexHull(largest_contour, hull);

    // Create a mask of zeros and draw the convex hull on it
    Mat mask = Mat::zeros(green_mask2.size(), green_mask2.type());
    vector<vector<Point>> hulls;
    hulls.push_back(hull);
    drawContours(mask, hulls, -1, Scalar(255), FILLED);
    
    // Create ROI by masking the original frame
    Mat roi_frame;
    bitwise_and(frame, frame, roi_frame, mask);
    
    return make_pair(roi_frame, mask);
}

pair<int, int> find_first_last_orange(const Mat &line_data, int line_start) {
    int first_orange = -1;
    int last_orange = -1;
    
    // Loop from beginning of the line_data row
    for (int idx = 0; idx < line_data.cols; idx++) {
        if (line_data.at<uchar>(0, idx) == 255) {
            first_orange = line_start + idx;
            break;
        }
    }
    
    // Loop from end of the line_data row backward
    for (int idx = line_data.cols - 1; idx >= 0; idx--) {
        if (line_data.at<uchar>(0, idx) == 255) {
            last_orange = line_start + idx;
            break;
        }
    }
    
    return make_pair(first_orange, last_orange);
}

pair<int, int> find_top_bottom_orange(const Mat &column_data, int col_start) {
    int top_orange = -1;
    int bot_orange = -1;
    
    // Loop from top of the column_data
    for (int idy = 0; idy < column_data.rows; idy++) {
        if (column_data.at<uchar>(idy, 0) == 255) {
            top_orange = col_start + idy;
            break;
        }
    }
    
    // Loop from bottom of the column_data upward
    for (int idy = column_data.rows - 1; idy >= 0; idy--) {
        if (column_data.at<uchar>(idy, 0) == 255) {
            bot_orange = col_start + idy;
            break;
        }
    }
    
    return make_pair(top_orange, bot_orange);
}

Mat detect(const Mat &mask_ball, const Mat &mask_field, Mat frame) {
    // Apply Gaussian Blur to the ball mask
    Mat blurred_ball_mask;
    GaussianBlur(mask_ball, blurred_ball_mask, Size(9, 9), 2);

    // Detect circles using HoughCircles
    vector<Vec3f> circles;
    HoughCircles(blurred_ball_mask, circles, HOUGH_GRADIENT, 1.2, 50, 50, 30, 1, 1000);

    if (!circles.empty()) {
        // Convert circle parameters to integer values (the rounding is achieved by casting after rounding)
        for (size_t i = 0; i < circles.size(); i++) {
            int x = cvRound(circles[i][0]);
            int y = cvRound(circles[i][1]);
            int r = cvRound(circles[i][2]);
            int stroke = static_cast<int>(1.1 * r);
            
            // Horizontal line processing
            int line_y = y;
            int line_x_start = max(0, x - stroke);
            int line_x_end = min(mask_ball.cols - 1, x + stroke);
            // Get a horizontal slice (row) from mask_ball with proper range
            Rect hRect(line_x_start, line_y, line_x_end - line_x_start, 1);
            if (hRect.width <= 0) {
                continue;
            }
            Mat orange_hline = mask_ball(hRect).clone();
            // Reshape to 1 row for processing
            orange_hline = orange_hline.reshape(0, 1);
            
            pair<int, int> first_last = find_first_last_orange(orange_hline, line_x_start);
            int first_orange = first_last.first;
            int last_orange = first_last.second;
            
            if (first_orange == -1 || last_orange == -1) {
                continue;
            }
            
            int total_x_pixel = last_orange - first_orange;
            int r_new = total_x_pixel / 2;
            int x_new = first_orange + r_new;
            
            // Vertical line processing
            int line_x = x_new;
            int line_y_start = y - stroke;
            int line_y_end = y + stroke;
            // Clamp vertical boundaries
            line_y_start = max(0, line_y_start);
            line_y_end = min(mask_ball.rows - 1, line_y_end);
            if (line_y_end - line_y_start <= 0) {
                continue;
            }
            Rect vRect(line_x, line_y_start, 1, line_y_end - line_y_start);
            Mat orange_vline = mask_ball(vRect).clone();
            // Reshape to single column
            orange_vline = orange_vline.reshape(0, orange_vline.rows);
            
            pair<int, int> top_bot = find_top_bottom_orange(orange_vline, line_y_start);
            int top_orange = top_bot.first;
            int bot_orange = top_bot.second;
            
            if (top_orange == -1 || bot_orange == -1) {
                continue;
            }
            
            int total_y_pixel = abs(top_orange - bot_orange);
            int y_new = bot_orange - (total_y_pixel / 2);
            
            if (y_new != y) {
                line_y = y_new;
                Rect hRect2(line_x_start, line_y, line_x_end - line_x_start, 1);
                if (hRect2.width <= 0) {
                    continue;
                }
                Mat orange_hline2 = mask_ball(hRect2).clone();
                orange_hline2 = orange_hline2.reshape(0, 1);
                pair<int, int> first_last2 = find_first_last_orange(orange_hline2, line_x_start);
                first_orange = first_last2.first;
                last_orange = first_last2.second;
                
                if (first_orange == -1 || last_orange == -1) {
                    continue;
                }
                
                total_x_pixel = last_orange - first_orange;
                r_new = total_x_pixel / 2;
                x_new = first_orange + r_new;
            }
            
            int R = static_cast<int>(r_new * 1.5);  // jarak deteksi
            
            int x1 = max(x_new - R, 0);
            int y1 = max(y_new - R, 0);
            int x2 = min(x_new + R, frame.cols);
            int y2 = min(y_new + R, frame.rows);
            // cv::rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

            // Compute the ratio of green in the surrounding field area
            Rect surroundingRect(x1, y1, x2 - x1, y2 - y1);
            Mat surrounding_field = mask_field(surroundingRect);
            double field_pixels = countNonZero(surrounding_field);
            double field_ratio = field_pixels / (surrounding_field.size().area());

            // Compute the ratio of orange in the surrounding ball area
            Mat surrounding_ball = mask_ball(surroundingRect);
            double ball_pixels = countNonZero(surrounding_ball);
            double ball_ratio = ball_pixels / (surrounding_ball.size().area());

            if ((field_ratio > 0.16) && (ball_ratio < 0.47)) {
                line(frame, Point(x_new, y_new + r_new), Point(x_new, y_new - r_new), Scalar(0, 255, 0), 2);
                line(frame, Point(x_new - r_new, y_new), Point(x_new + r_new, y_new), Scalar(0, 255, 0), 2);
                circle(frame, Point(x_new, y_new), r_new, Scalar(0, 255, 0), 2);
            } else {
                continue;
            }

            double actual_diameter = 0.13; // meter
            double focal_length = 714.1;
            int detected_diameter = total_x_pixel;
            double distance = 0;
            if (detected_diameter == 0) {
                distance = 0;
            } else {
                distance = (actual_diameter * focal_length) / detected_diameter;
            }
            // Create the text with distance value formatted to two decimals
            char text2[100];
            sprintf(text2, "Distance: %.2f meter", distance);
            putText(frame, text2, Point(x_new - r_new, y_new + r_new + 40), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

            // Print distance in cm (commented out as in original code)
            // printf("distance= %.3f cm\n", distance*100);
            line(frame, Point(center_x, center_y), Point(x_new, y_new), Scalar(255, 255, 255), 2);

            break;
        }
    }
    return frame;
}

int main() {
    // Open video file or camera stream
    VideoCapture cap("sample2.mp4");
    // VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "Error: No Video Opened" << endl;
        return -1;
    }

    // Uncommented VideoWriter code as in the original Python code
    // int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
    // double fps = cap.get(CAP_PROP_FPS);
    // int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    // int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    // VideoWriter out("test2.mp4", fourcc, fps, Size(width, height));

    while (true) {
        Mat frame;
        bool ret = cap.read(frame);
        if (!ret) {
            break;
        }

        int height = frame.rows;
        int width = frame.cols;

        center_x = width / 2;
        center_y = height / 2;

        int y_start = 0;
        int y_end = height;
        int x_start = 0;
        int x_end = width;

        // Draw vertical and horizontal lines on the frame
        line(frame, Point(center_x, y_start), Point(center_x, y_end), Scalar(255, 255, 255), 1);
        line(frame, Point(x_start, center_y), Point(x_end, center_y), Scalar(255, 255, 255), 1);

        // Process the field to get segmented field and its mask
        pair<Mat, Mat> fieldResult = field(frame);
        Mat seg_field = fieldResult.first;
        Mat mask_field = fieldResult.second;
        Mat mask_ball = ball(seg_field);
        // Mat mask_ball = ball(frame);
        // Mat final_frame = detect(mask_ball, frame, frame);
        Mat final_frame = detect(mask_ball, mask_field, frame);

        //imshow("mask field", mask_field);
        //imshow("mask ball", mask_ball);

        if (!final_frame.empty()) {
            // out.write(final_frame);
            imshow("ball detect", final_frame);
        }

        if (waitKey(1) == 'q') {
            break;
        }
    }

    // out.release();
    cap.release();
    destroyAllWindows();
    return 0;
}

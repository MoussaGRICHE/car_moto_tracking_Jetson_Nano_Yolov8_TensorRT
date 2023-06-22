
#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <vector>
#include <cmath>
#include <list>


const std::vector<std::string> CLASS_NAMES = {
	"car", "motorcycle"};

const std::vector<std::vector<unsigned int>> COLORS = {
	{ 0, 255, 0 }, { 0, 0, 255 }
};

// Define a vector of displayed class names
const std::vector<std::string> DISPALYED_CLASS_NAMES = {
	"car", "motorcycle" };

// Function to generate the GStreamer pipeline string
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {

    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

}


cv::Point crossingLine[4];
int clickCount = 0;


void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONUP) {
        if (clickCount < 2) {
            crossingLine[clickCount].x = x;
            crossingLine[clickCount].y = y;
            std::cout << "Click " << clickCount + 1 << ": (" << crossingLine[clickCount].x << ", " << crossingLine[clickCount].y << ")\n";
            clickCount++;
            // Destroy the window after getting user input
            if (clickCount == 2)
            destroyWindow("Get Crossing Line");
        }
    }
}


bool hasPassedLine(const cv::Point& lineStart, const cv::Point& lineEnd, const cv::Point& point)
{
    cv::Point lineVec(lineEnd.x - lineStart.x, lineEnd.y - lineStart.y);
    cv::Point pointVec(point.x - lineStart.x, point.y - lineStart.y);

    int crossProduct = lineVec.x * pointVec.y - lineVec.y * pointVec.x;
    
    // If the cross product is positive and the point is within the line segment, it has passed the line
    return crossProduct > 0 && point.x >= std::min(lineStart.x, lineEnd.x) && point.x <= std::max(lineStart.x, lineEnd.x) && point.y >= std::min(lineStart.y, lineEnd.y) && point.y <= std::max(lineStart.y, lineEnd.y);
}


#endif // UTIL_H
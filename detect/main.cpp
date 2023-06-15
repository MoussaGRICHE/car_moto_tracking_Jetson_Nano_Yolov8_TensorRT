//
// Created by ubuntu on 1/20/23.
//
#include "chrono"
#include "yolov8.hpp"
#include "opencv2/opencv.hpp"

const std::vector<std::string> CLASS_NAMES = {
	"car", "motorcycle"};

const std::vector<std::vector<unsigned int>> COLORS = {
	{ 0, 255, 0 }, { 0, 0, 255 }
};

// Function to generate the GStreamer pipeline string
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main(int argc, char** argv) {
    // Read command line arguments
    const std::string engine_file_path{argv[1]};
    const std::string input_type{argv[2]};
    std::string input_value;
    int infer_rate;
    std::string output_type;

    // Variables for video processing
    std::vector<std::string> imagePathList;
    bool isVideo{false};
    bool isCamera{false};

    // Create an instance of the YOLOv8 object detector
    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    // Process based on input type
    if (input_type == "video") {
        assert(argc == 6);
        input_value = argv[3];
        infer_rate = std::stoi(argv[4]);
        output_type = argv[5];
        if (IsFile(input_value)) {
            std::string suffix = input_value.substr(input_value.find_last_of('.') + 1);
            // Check if the input video file has a supported format
            if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv") {
                isVideo = true;
            } else {
                printf("suffix %s is wrong !!!\n", suffix.c_str());
                std::abort();
            }
        }
    } else if (input_type == "camera") {
        assert(argc == 5);
        infer_rate = std::stoi(argv[3]);
        output_type = argv[4];
        isCamera = true;
    }

    // Initialize OpenCV video capture and video writer
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    if (isVideo) {
        cap.open(input_value);
        if (!cap.isOpened()) {
            printf("can not open %s\n", input_value.c_str());
            return -1;
        }

        // Get video frame size
        cv::Size size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (output_type == "save") {
            // Generate a new filename for the output video
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
            auto str = oss.str();
            size_t lastindex = input_value.find_last_of(".");
            size_t lastSlash = input_value.find_last_of('/');
            size_t lastDot = input_value.find_last_of('.');
            std::string rawname = input_value.substr(lastSlash + 1, lastDot - lastSlash - 1);
            std::string new_filename = rawname + "_detection_" + str + ".mp4";
            writer.open(new_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, size);
        }
    } else {
        // Settings for camera input
        int capture_width = 1280;
        int capture_height = 720;
        int display_width = 1280;
        int display_height = 720;
        int framerate = 30;
        int flip_method = 2;

        // Generate the GStreamer pipeline string
        std::string pipeline = gstreamer_pipeline(capture_width, capture_height, display_width, display_height, framerate, flip_method);
        std::cout << "Using pipeline: \n\t" << pipeline << "\n";

        // Open the camera using the GStreamer pipeline
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            std::cout << "Failed to open camera." << std::endl;
            return -1;
        }

        // Get camera frame size
        cv::Size size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (output_type == "save") {
            // Generate a new filename for the output video
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
            auto str = oss.str();
            size_t lastindex = input_value.find_last_of(".");
            std::string rawname = input_value.substr(0, lastindex);
            std::string new_filename = "Camera_detection_" + str + ".mp4";
            writer.open(new_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, size);
        }
    }

    // Get the frame rate of the video
    int fps = cap.get(cv::CAP_PROP_FPS);

    // Variables for image processing
    cv::Mat res, image;
    cv::Size size = cv::Size{640, 640};
    std::vector<Object> objs;


    int total_ms = 0;

    int infer_frame_count = 0;

    // Main loop for processing video frames
    while (cap.read(image)) {
        if (infer_frame_count % infer_rate == 0) {
            auto start = std::chrono::system_clock::now();
            objs.clear();
            yolov8->copy_from_Mat(image, size);
            yolov8->infer();
            yolov8->postprocess(objs);

           

            // Draw bounding boxes and labels on the image
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);

            auto end = std::chrono::system_clock::now();
            double tc = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            double infer_fps = (1000.0 / tc) * infer_rate;
            printf("cost %2.4lf ms (%0.0lf fps, 1/ %d frame processed)\n", tc, std::round(infer_fps), infer_rate);

            // Draw the FPSon the image
            yolov8->draw_fps(image, res, infer_fps, infer_rate);

            if (output_type == "save") {
                writer.write(res);
            }

            if (output_type == "show") {
                // Show the result image
                cv::namedWindow("result", cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);
                cv::setWindowProperty("result", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
                cv::imshow("result", res);
                if (cv::waitKey(10) == 'q') {
                    break;
                }
            }
        }
        infer_frame_count++;
    }

    // Clean up resources
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}


#include "yolov8.hpp"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "BYTETracker.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "util.h"
#include <cstring>
#include <unistd.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // Read command line arguments
    const string engine_file_path{argv[1]};
    const string input_type{argv[2]};
    string input_value;
    int infer_rate;
    string output_type;
    string ssh;
    string pipeline;

    // Variables for video processing
    vector<string> imagePathList;
    bool isVideo{false};
    bool isCamera{false};
    string new_filename;

    int frame_count = 0;
    int infer_frame_count = 0;
    int total_ms = 0;
    map<string, int> classCounts_IN;
    map<string, int> classCounts_OUT;

    vector<int> crossedTrackerIds;

    // Create an instance of the YOLOv8 object detector
    YOLOv8* yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    // Process based on input type
    if (input_type == "video") {
        assert(argc == 6);
        input_value = argv[3];
        if (IsFile(input_value)) {
            string suffix = input_value.substr(input_value.find_last_of('.') + 1);
            // Check if the input video file has a supported format
            if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv") {
                isVideo = true;
            } else {
                printf("suffix %s is wrong !!!\n", suffix.c_str());
                abort();
            }
        }
        infer_rate = stoi(argv[4]);
        output_type = argv[5];
        ssh = (argc >= 7) ? argv[6] : "";

    } else if (input_type == "camera") {
        assert(argc == 5);
        infer_rate = stoi(argv[3]);
        output_type = argv[4];
        ssh = (argc >= 6) ? argv[5] : "";
        isCamera = true;

    }

    // Initialize OpenCV video capture and video writer
    VideoCapture cap;
    VideoWriter writer;
    if (isVideo) {
        cap.open(input_value);
        if (!cap.isOpened()) {
            printf("can not open %s\n", input_value.c_str());
            return -1;
        }

        // Get video frame size
        Size size = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));

        if (output_type == "save") {
            // Generate a new filename for the output video
            auto t = time(nullptr);
            auto tm = *localtime(&t);
            ostringstream oss;
            oss << put_time(&tm, "%Y-%m-%d_%H-%M-%S");
            auto str = oss.str();
            size_t lastindex = input_value.find_last_of(".");
            size_t lastSlash = input_value.find_last_of('/');
            size_t lastDot = input_value.find_last_of('.');
            string rawname = input_value.substr(lastSlash + 1, lastDot - lastSlash - 1);
            new_filename = rawname + "_detection_" + str + ".avi";
            writer.open(new_filename, VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, size);
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

        pipeline = gstreamer_pipeline(capture_width, capture_height, display_width, display_height, framerate, flip_method);         

        cout << "Using pipeline: \n\t" << pipeline << "\n";

        // Open the camera using the GStreamer pipeline
        cap.open(pipeline, CAP_GSTREAMER);
        if (!cap.isOpened()) {
            cout << "Failed to open camera." << endl;
            return -1;
        }

        // Get camera frame size
        Size size = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
        if (output_type == "save") {
            // Generate a new filename for the output video
            auto t = time(nullptr);
            auto tm = *localtime(&t);
            ostringstream oss;
            oss << put_time(&tm, "%Y-%m-%d_%H-%M-%S");
            auto str = oss.str();
            size_t lastindex = input_value.find_last_of(".");
            string rawname = input_value.substr(0, lastindex);
            new_filename = "Camera_detection_" + str + ".avi";
            writer.open(new_filename, VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, size);
        }
    }

    // Get the frame rate of the video
    int fps = cap.get(CAP_PROP_FPS);

    // Variables for image processing
    Mat res, image;
    Size size = Size{640, 640};
    vector<Object> objs;
    vector<Object> track_objs;

    // Create a tracker object for object tracking
    BYTETracker tracker(fps, 30);

    if (ssh == "ssh"){

        Mat frame;
        cap.read(frame);
        std::string filename = "frame_for_line.jpg";
        imwrite(filename, frame);

        char buffer[FILENAME_MAX];
        if (getcwd(buffer, FILENAME_MAX)) {
            std::cout << "Image saved to: " << buffer << "/" << filename << std::endl;
        } else {
            std::cerr << "Error getting current working directory: " << strerror(errno) << std::endl;
        }

        std::cout << "Enter the start point x-coordinate: ";
        std::cin >> crossingLine[0].x;
        std::cout << "Enter the start point y-coordinate: ";
        std::cin >> crossingLine[0].y;
        std::cout << "Enter the end point x-coordinate: ";
        std::cin >> crossingLine[1].x;
        std::cout << "Enter the end point y-coordinate: ";
        std::cin >> crossingLine[1].y;

    }

    else {
        // Create a window and set the mouse callback to get user input
        namedWindow("Get Crossing Line", WINDOW_NORMAL);
        setMouseCallback("Get Crossing Line", onMouse);

        // Get the first frame and display it to get user input
        Mat frame;
        cap.read(frame);
        imshow("Get Crossing Line", frame);

        while (true) {
            if (clickCount == 2)
                break;
            if (waitKey(10) == 27) // Wait for the Escape key (ASCII value 27) to be pressed
                break;
        }
    }

    for (const auto& className : DISPALYED_CLASS_NAMES) {
        classCounts_IN[className] = 0;
        classCounts_OUT[className] = 0;
    }

    // Main loop for processing video frames
    while (cap.read(image)) {
        if (infer_frame_count % infer_rate == 0) {
            auto start = chrono::system_clock::now();
            objs.clear();
            track_objs.clear();
            yolov8->copy_from_Mat(image, size);
            yolov8->infer();
            yolov8->postprocess(objs);
            vector<STrack> output_stracks = tracker.update(objs);

            // Process each detected object and its tracking information
            for (int i = 0; i < output_stracks.size(); i++) {
                vector<float> tlwh = output_stracks[i].tlwh;
                Scalar s = tracker.get_color(output_stracks[i].track_id);
                Object obj;
                obj.rect = Rect_<float>(
                    output_stracks[i].tlwh[0],
                    output_stracks[i].tlwh[1],
                    output_stracks[i].tlwh[2],
                    output_stracks[i].tlwh[3]
                );
                obj.label = output_stracks[i].label;
                obj.tracker_id = output_stracks[i].track_id;
                track_objs.push_back(obj);
            }

            // Check the line
            bool blnAtLeastOneObjCrossedTheLine = checkIfObjsCrossedTheLine(
                                                        track_objs, 
                                                        crossingLine, 
                                                        DISPALYED_CLASS_NAMES, 
                                                        CLASS_NAMES, 
                                                        classCounts_IN, 
                                                        classCounts_OUT, 
                                                        crossedTrackerIds
                                                        );

            // Draw the line
            Scalar lineColor = blnAtLeastOneObjCrossedTheLine ? Scalar(0.0, 200.0, 0.0) : Scalar(0.0, 0.0, 255.0);

            line(image, crossingLine[0], crossingLine[1], lineColor, 2);


            

            // Draw bounding boxes, labels, tracker_id on the image
            yolov8->draw_objects(image, res, track_objs, CLASS_NAMES, COLORS);

             // Draw the counting results on the image
            yolov8->drawCountingResults(image, res, CLASS_NAMES, classCounts_IN, classCounts_OUT); // "IN" counting results

            auto end = chrono::system_clock::now();
            double tc = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
            double infer_fps = (1000.0 / tc) * infer_rate;
            printf("cost %2.4lf ms (%0.0lf fps, 1/ %d frame processed)\n", tc, round(infer_fps), infer_rate);

            // Draw the FPSon the image
            yolov8->draw_fps(image, res, infer_fps, infer_rate);

            if (ssh == "ssh"){
                writer.write(res);
            }
            else {
                if (output_type == "save") {
                namedWindow("result", WINDOW_NORMAL | WINDOW_GUI_EXPANDED);
                setWindowProperty("result", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
                imshow("result", res);
                if (waitKey(10) == 'q' || waitKey(10) == 'Q') {
                    break;
                }
                writer.write(res); 
                }

                if (output_type == "show") {
                    // Show the result image
                    namedWindow("result", WINDOW_NORMAL | WINDOW_GUI_EXPANDED);
                    setWindowProperty("result", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
                    imshow("result", res);
                    if (waitKey(10) == 'q' || waitKey(10) == 'Q') {
                        break;
                    }
                }

            }
        }
        infer_frame_count++;
    }

    

    if (output_type == "save") {
        char buffer[FILENAME_MAX];
            if (getcwd(buffer, FILENAME_MAX)) {
                std::cout << "Video saved to: " << buffer << "/" << new_filename << std::endl;
            } else {
                std::cerr << "Error getting current working directory: " << strerror(errno) << std::endl;
            }
        }

    // Release camera and video resources
    cap.release();

    writer.release();

    // Close OpenCV windows
    destroyAllWindows();

    // Delete the YOLOv8 object
    delete yolov8;

    return 0;
}
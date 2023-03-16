#include <stdio.h>

#include <syslog.h>

#include <stdlib.h>

#include <iostream>

#include <syslog.h>

#include <pthread.h>

#include <sched.h>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>

#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;
#define ESCAPE_KEY (27)


void * ppl(void * arg) {
    VideoCapture cap((char * ) arg);
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    Mat frame, image;
    namedWindow("PEDESTRIAN_DETECTION", WINDOW_NORMAL);
    resizeWindow("PEDESTRIAN_DETECTION", 640, 480);
    while (1) {
        int64 t = getTickCount();
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        resize(frame, image, Size(480, 320), INTER_LINEAR);
	Mat mask(image.size().height, image.size().width, CV_8UC3, Scalar(0));
        Point p1 = Point(80, 230);
        Point p2 = Point(80, 80);
        Point p3 = Point(400,80);
        Point p4 = Point(400,230);
        Point vertices1[] = {p1,p2,p3,p4};
        vector < Point > vertices(vertices1, vertices1 + sizeof(vertices1) / sizeof(Point));
        vector < vector < Point > > verticesToFill;
        verticesToFill.push_back(vertices);
        fillPoly(mask, verticesToFill, Scalar(255, 255, 255));
        Mat maskedIm = image.clone();
        bitwise_and(image, mask, maskedIm);
        vector < Rect > found;
        vector < double > weights;
        hog.detectMultiScale(maskedIm, found, 0, Size(8, 8), Size(0, 0), 1, 0.8, false);
        for (size_t i = 0; i < found.size(); i++) {
            Rect r = found[i];
            rectangle(image, found[i], Scalar(0, 0, 255), 2);
            stringstream temp;
            putText(image, temp.str(), Point(found[i].x, found[i].y + 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
        }
        t = getTickCount() - t; {
            ostringstream buf;
            buf << "FPS: " << fixed << setprecision(1) << ((getTickFrequency() / (double) t));
            putText(image, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        }
	syslog(LOG_NOTICE, "PEDESTRIAN FPS: %f\n", (getTickFrequency() / double(t)));
        imshow("PEDESTRIAN_DETECTION", image);       
	char c = waitKey(1);
        if (c == 27) {

            break;
        }
    }
}

void * lane(void * arg) {
    VideoCapture cap((char * ) arg);
    Mat image, frame;
    bool right_flag = false;
    bool left_flag = false;
    double right_m;
    double left_m;
    Point right_b;
    Point left_b;
    double img_center;
    namedWindow("LANE_DETECTION", WINDOW_NORMAL);
    resizeWindow("LANE_DETECTION", 640, 480);

    while (1) {
        int64 t = getTickCount();
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        resize(frame, image, Size(640, 480), INTER_LINEAR);
        Mat imageGray;
	cvtColor(image, imageGray, COLOR_BGR2GRAY);
        Mat smoothedIm;
        GaussianBlur(imageGray, smoothedIm, Size(5, 5), 0);
        Mat edgesIm;
        Canny(smoothedIm, edgesIm, 20, 50);
        Mat mask(image.size().height, image.size().width, CV_8UC1, Scalar(0));
        Point p1 = Point(160, 376);
        Point p2 = Point(280, 280);
        Point p3 = Point(325, 280);
        Point p4 = Point(420, 377);
        Point vertices1[] = {p1,p2,p3,p4};
        vector < Point > vertices(vertices1, vertices1 + sizeof(vertices1) / sizeof(Point));
        vector < vector < Point > > verticesToFill;
        verticesToFill.push_back(vertices);
        fillPoly(mask, verticesToFill, Scalar(255, 255, 255));
        Mat maskedIm = edgesIm.clone();
        bitwise_and(edgesIm, mask, maskedIm);
        float pi = 3.14159265358979323846;
        float theta = pi / 180;
        int minLineLength = 10;
        int maxLineGap = 100;
        vector < Vec4i > lines;
        HoughLinesP(maskedIm, lines, 1, theta, 20, minLineLength, maxLineGap);
        vector < vector < Vec4i >> output(2);
        size_t j = 0;
        Point ini;
        Point fini;
        double slope_thresh = 0.3;
        vector < double > slopes;
        vector < Vec4i > selected_lines;
        vector < Vec4i > right_lines, left_lines;

        //Calculate the slope of all the detected lines
        for (auto i: lines) {
            ini = Point(i[0], i[1]);
            fini = Point(i[2], i[3]);

            //Basic algebra: slope = (y1 - y0)/(x1 - x0)
            double slope = (static_cast < double > (fini.y) - static_cast < double > (ini.y)) / (static_cast < double > (fini.x) - static_cast < double > (ini.x) + 0.00001);
            if (abs(slope) > slope_thresh) {
                slopes.push_back(slope);
                selected_lines.push_back(i);
            }
        }
        img_center = static_cast < double > ((edgesIm.cols / 2));
        while (j < selected_lines.size()) {
            ini = Point(selected_lines[j][0], selected_lines[j][1]);
            fini = Point(selected_lines[j][2], selected_lines[j][3]);
            if (slopes[j] > 0 && fini.x > img_center && ini.x > img_center) {
                right_lines.push_back(selected_lines[j]);
                right_flag = true;
            } else if (slopes[j] < 0 && fini.x < img_center && ini.x < img_center) {
                left_lines.push_back(selected_lines[j]);
                left_flag = true;
            }
            j++;
        }
        output[0] = right_lines;
        output[1] = left_lines;

        vector < Point > output1(4);
        Point x1;
        Point y1;
        Point x2;
        Point y2;
        Vec4d right_line;
        Vec4d left_line;
        vector < Point > right_pts;
        vector < Point > left_pts;
        if (right_flag == true) {
            for (auto i: output[0]) {
                x1 = Point(i[0], i[1]);
                y1 = Point(i[2], i[3]);

                right_pts.push_back(x1);
                right_pts.push_back(y1);
            }

            if (right_pts.size() > 0) {
                fitLine(right_pts, right_line, DIST_L2, 0, 0.01, 0.01);
                right_m = right_line[1] / right_line[0];
                right_b = Point(right_line[2], right_line[3]);
            }
        }
        if (left_flag == true) {
            for (auto j: output[1]) {
                x2 = Point(j[0], j[1]);
                y2 = Point(j[2], j[3]);

                left_pts.push_back(x2);
                left_pts.push_back(y2);
            }

            if (left_pts.size() > 0) {
                fitLine(left_pts, left_line, DIST_L2, 0, 0.01, 0.01);
                left_m = left_line[1] / left_line[0];
                left_b = Point(left_line[2], left_line[3]);
            }
        }
        int ini_y = 380;
        int fin_y = 310;

        double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
        double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;

        double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
        double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;

        output1[0] = Point(right_ini_x, ini_y);
        output1[1] = Point(right_fin_x, fin_y);
        output1[2] = Point(left_ini_x, ini_y);
        output1[3] = Point(left_fin_x, fin_y);

        vector < Point > poly_points;
        poly_points.push_back(output1[2]);
        poly_points.push_back(output1[0]);
        poly_points.push_back(output1[1]);
        poly_points.push_back(output1[3]);

        Mat allLinesIm(image.size().height, image.size().width, CV_8UC3, Scalar(0, 0, 0));
        for (int i = 0; i != output1.size(); ++i) {
            line(allLinesIm, output1[0], output1[1], Scalar(0, 0, 255), 5);
            line(allLinesIm, output1[2], output1[3], Scalar(0, 0, 255), 5);
            fillConvexPoly(allLinesIm, poly_points, Scalar(40, 0, 0), LINE_AA, 0);

        }
        Mat res;
        addWeighted(allLinesIm, 1, image, 0.9, 0, res);
        t = getTickCount() - t; {
            ostringstream buf;
            buf << "FPS: " << fixed << setprecision(1) << ((getTickFrequency() / (double) t));
            putText(res, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        }
	syslog(LOG_NOTICE, "LANE DETECTION FPS: %f\n", (getTickFrequency() / double(t)));        
	imshow("LANE_DETECTION", res);
        char c = waitKey(1);
        if (c == 27) {
            break;
        }

    }
}

void * stoplight(void * arg) {
    VideoCapture cap((char * ) arg);
    Mat frame, frame1, gray, hsv, red, bit1, blur;
    namedWindow("STOP_LIGHT_DETECTION", WINDOW_NORMAL);
    resizeWindow("STOP_LIGHT_DETECTION", 640, 480);

    while (1) {
        int64 t = getTickCount();
        cap >> frame1;
        if (frame1.empty()) {
            break;
        }
        resize(frame1, frame, Size(480, 320), INTER_LINEAR);
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(0, 210, 20), Scalar(179, 255, 255), red);
        bitwise_and(frame, hsv, bit1);
        medianBlur(red, blur, 5);
        vector < Vec3f > circles;
        Mat mask(frame.size().height, frame.size().width, CV_8UC1, Scalar(0));
        Point p1 = Point(20, 110);
        Point p2 = Point(20, 0);
        Point p3 = Point(440, 0);
        Point p4 = Point(440, 110);
        Point vertices1[] = {p1,p2,p3,p4};
        vector < Point > vertices(vertices1, vertices1 + sizeof(vertices1) / sizeof(Point));
        vector < vector < Point > > verticesToFill;
        verticesToFill.push_back(vertices);
        fillPoly(mask, verticesToFill, Scalar(255, 255, 255));
        Mat maskedIm = blur.clone();
        bitwise_and(blur, mask, maskedIm);
        HoughCircles(maskedIm, circles, HOUGH_GRADIENT, 1, 50, 425, 9, 4, 10);
        for (size_t i = 0; i < circles.size(); i++) {
            Vec3i c = circles[i];
            Point center = Point(c[0], c[1]);
            circle(frame, center, 10, Scalar(0, 0, 255), 3, LINE_AA);
            int radius = c[2];
            putText(frame, "STOP LIGHT", Point(120, 150), FONT_HERSHEY_PLAIN, 3.0, Scalar(0, 0, 255), 5, LINE_AA);
        }
        t = getTickCount() - t; {
            ostringstream buf;
            buf << "FPS: " << fixed << setprecision(1) << ((getTickFrequency() / (double) t));
		double a = (getTickFrequency() / (double) t);
            putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        }
	syslog(LOG_NOTICE, "STOP LIGHT DETECTION FPS: %f\n", (getTickFrequency() / double(t))); 
        imshow("STOP_LIGHT_DETECTION", frame);
        char c = waitKey(1);
        if (c == 27) {
      
	  break;
        }
    }
}


void * stopsign(void * arg)
{
 VideoCapture cap((char * ) arg);
    Mat frame, frame1, hsv, red, blur;
    namedWindow("STOP_SIGN_DETECTION", WINDOW_NORMAL);
    resizeWindow("STOP_SIGN_DETECTION", 640, 480);

    while (1) {
        int64 t = getTickCount();
        cap >> frame1;
        if (frame1.empty()) {
            break;
        }
        resize(frame1, frame, Size(480, 320), INTER_LINEAR);
        cvtColor(frame, hsv, COLOR_BGR2HSV);	
	inRange(hsv, Scalar(0, 0, 0), Scalar(255, 90, 255), red);	
        vector < Vec3f > circles;
        Mat mask(red.size().height, red.size().width, CV_8UC1, Scalar(0));
        Point p1 = Point(250, 160);
        Point p2 = Point(250, 80);
        Point p3 = Point(440, 80);
        Point p4 = Point(440, 160);
        Point vertices1[] = {p1,p2,p3,p4};
        vector < Point > vertices(vertices1, vertices1 + sizeof(vertices1) / sizeof(Point));
        vector < vector < Point > > verticesToFill;
        verticesToFill.push_back(vertices);
        fillPoly(mask, verticesToFill, Scalar(255, 255, 255));
        Mat maskedIm = frame.clone();
        bitwise_and(red, mask, maskedIm);       
	HoughCircles(maskedIm, circles, HOUGH_GRADIENT,1,100,425,9,11, 15);
        for (size_t i = 0; i < circles.size(); i++) {
            Vec3i c = circles[i];
            Point center = Point(c[0], c[1]);
            circle(frame, center, 20, Scalar(0, 0, 255), 2, LINE_AA);
            int radius = c[2];
            putText(frame, "STOP", Point(140, 180), FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 0, 255), 3, LINE_AA);
	}
        t = getTickCount() - t; {
            ostringstream buf;
            buf << "FPS: " << fixed << setprecision(1) << ((getTickFrequency() / (double) t));
		double a = (getTickFrequency() / (double) t);
            putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);	
        }
	syslog(LOG_NOTICE, "STOP SIGN DETECTION FPS: %f\n", (getTickFrequency() / double(t))); 
        imshow("STOP_SIGN_DETECTION", frame);
        char c = waitKey(1);
        if (c == 27) {
            break;
        }
    }
 
}



int main(int argc, char * argv[]) {
setlogmask(LOG_UPTO (LOG_NOTICE));
openlog("612_FINAL", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL1);
const char* keys = {
"{l|  |Lane detecion}"
"{p|  |Pedestrian detection}"
"{sl|  |Stop light detection}"
"{ss|  |Stop sign detection}"
};
CommandLineParser parser(argc, argv, keys);

    pthread_t th;
    pthread_t th1;
    pthread_t th2;
    pthread_t th3;

    int coreid1 = 0;
    int coreid2 = 1;
    int coreid3 = 2;
    int coreid4 = 3;

    cpu_set_t threadcpu1;
    cpu_set_t threadcpu2;
    cpu_set_t threadcpu3;
    cpu_set_t threadcpu4;

    CPU_ZERO( & threadcpu1);
    CPU_ZERO( & threadcpu2);
    CPU_ZERO( & threadcpu3);
    CPU_ZERO( & threadcpu4);

    CPU_SET(coreid1, & threadcpu1);
    CPU_SET(coreid2, & threadcpu2);
    CPU_SET(coreid3, & threadcpu3);
    CPU_SET(coreid4, & threadcpu4);

if(parser.has("ss"))   
{
    syslog(LOG_NOTICE, "Stop Sign Detection running on core #1");
    sched_setaffinity(0, sizeof(threadcpu1), & threadcpu1);
    pthread_create( &th3, NULL, stopsign, (void * )"trim_final.mov");
}

if(parser.has("p"))
{
    syslog(LOG_NOTICE, "Pedestrian Detection running on core #2");
    sched_setaffinity(0, sizeof(threadcpu2), & threadcpu2);
    pthread_create( &th1, NULL, ppl, (void * )"V002 (1).seq");
}

if(parser.has("l"))
{
    syslog(LOG_NOTICE, "Lane Detection running on core #3");
    sched_setaffinity(0, sizeof(threadcpu3), & threadcpu3);
    pthread_create( &th, NULL, lane, (void * )"12.mp4");
}

if(parser.has("sl"))   
{
    syslog(LOG_NOTICE, "Stop light Detection running on core #4");
    sched_setaffinity(0, sizeof(threadcpu4), & threadcpu4);
    pthread_create( &th2, NULL, stoplight, (void * )"V002 (1).seq");
}



if(parser.has("l"))
{
    pthread_join(th, NULL);
}
if(parser.has("p"))
{
    pthread_join(th1, NULL);
}
if(parser.has("sl"))
{
    pthread_join(th2, NULL);
}
if(parser.has("ss"))
{
    pthread_join(th3, NULL);
}
syslog(LOG_NOTICE, "Finished running");
closelog();
return 0;

}

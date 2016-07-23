#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

using namespace std;
using namespace cv;

const int marker_width = 200;

Scalar blue(255, 0, 0);
Scalar green(0, 255, 0);
Scalar red(0, 0, 255);

void drawQuad(Mat image, vector<Point2f> points, Scalar color) {
    line(image, points[0], points[1], color);
    line(image, points[1], points[2], color);
    line(image, points[2], points[3], color);
    line(image, points[4], points[4], color);
}


void clockwise(vector<Point2f>& square){
    Point2f v1 = square[1] - square[0];
    Point2f v2 = square[2] - square[0];

    double o = (v1.x * v2.y) - (v1.y * v2.x);

    if (o < 0.0){
        std::swap(square[1],square[3]);
    }
}

int main(int argc, char** argv) {

    Mat image;

    cv::namedWindow("image", CV_WINDOW_NORMAL);
    cv::setWindowProperty("image", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    VideoCapture cap("video.mp4");
    //VideoCapture cap(0);

    if(!cap.isOpened())
        return -1;

    while (cap.grab()) {
        cap.retrieve(image);

        Mat grayImage;
        cvtColor(image, grayImage, CV_RGB2GRAY);
        Mat blurredImage;
        blur(grayImage, blurredImage, Size(5, 5));
        Mat threshImage;
        threshold(blurredImage, threshImage, 128.0, 255.0, THRESH_OTSU);

        vector<vector<Point> > contours;
        findContours(threshImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

        vector<vector<Point2f> > squares;
        for (int i = 0; i < contours.size(); i++) {
            vector<Point> contour = contours[i];
            vector<Point> approx;
            approxPolyDP(contour, approx, arcLength(Mat(contour), true)*0.02, true);
            if( approx.size() == 4 &&
                fabs(contourArea(Mat(approx))) > 1000 &&
                isContourConvex(Mat(approx)) )
            {
                vector<Point2f> square;

                for (int i = 0; i < 4; ++i)
                {
                    square.push_back(Point2f(approx[i].x,approx[i].y));
                }
                squares.push_back(square);
            }
        }
        vector<Point2f> square = squares[0];
        drawQuad(image, square, green);

        clockwise(square);

        Mat marker;
        vector<Point2f> marker_square;

        marker_square.push_back(Point(0,0));
        marker_square.push_back(Point(marker_width-1, 0));
        marker_square.push_back(Point(marker_width-1,marker_width-1));
        marker_square.push_back(Point(0, marker_width-1));


        Mat transform = getPerspectiveTransform(square, marker_square);
        warpPerspective(grayImage, marker,transform, Size(marker_width,marker_width));
        threshold(marker, marker, 125, 255, THRESH_BINARY|THRESH_OTSU);

        vector<Point> direction_point = {{50, 50} ,{150, 50},{150, 150},{50,150}};
        int direction;
        for (int i = 0; i < 4; ++i){
            Point p = direction_point[i];
            if (countNonZero(marker(Rect(p.x-25,p.y-25,marker_width/4,marker_width/4))) >20){
                direction = i;
                break;
            }
        }
        for (int i = 0; i < direction; ++i){
            rotate(square.begin(), square.begin() + 1, square.end());
        }

        circle(image, square[0], 5, red);

        FileStorage fs("calibrate/out_camera_data.xml", FileStorage::READ);
        Mat intrinsics, distortion;
        fs["Camera_Matrix"] >> intrinsics;
        fs["Distortion_Coefficients"] >> distortion;

        vector<Point3f> objectPoints;
        objectPoints.push_back(Point3f(-1, 1, 0));
        objectPoints.push_back(Point3f(1, 1, 0));
        objectPoints.push_back(Point3f(1, -1, 0));
        objectPoints.push_back(Point3f(-1, -1, 0));
        Mat objectPointsMat(objectPoints);

        Mat rvec;
        Mat tvec;

        solvePnP(objectPointsMat, square, intrinsics, distortion, rvec, tvec);

        cout << "rvec: " << rvec << endl;
        cout << "tvec: " << tvec << endl;

        vector<Point3f> line3dx = {{0, 0, 0}, {1, 0, 0}};
        vector<Point3f> line3dy = {{0, 0, 0}, {0, 1, 0}};
        vector<Point3f> line3dz = {{0, 0, 0}, {0, 0, 1}};

        vector<Point2f> line2dx;
        vector<Point2f> line2dy;
        vector<Point2f> line2dz;
        projectPoints(line3dx, rvec, tvec, intrinsics, distortion, line2dx);
        projectPoints(line3dy, rvec, tvec, intrinsics, distortion, line2dy);
        projectPoints(line3dz, rvec, tvec, intrinsics, distortion, line2dz);


        line(image, line2dx[0], line2dx[1], red);
        line(image, line2dy[0], line2dy[1], blue);
        line(image, line2dz[0], line2dz[1], green);


        cv::imshow("image", image);
        cv::waitKey(100);
    }
    return 0;
}

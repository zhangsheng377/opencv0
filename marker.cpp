#include <opencv2/core/core.hpp>    //opencv核心库
#include <opencv2/imgproc/imgproc.hpp>  //opencv图像处理库
#include <opencv2/highgui/highgui.hpp>  //用户界面库，播放视频用
#include <opencv2/calib3d/calib3d.hpp>  //相机校准，姿态估计用

#include <iostream>

using namespace std;
using namespace cv;

//图像的宽度
const int marker_width = 200;

//Scalar是一个四元组，这里对应b，g，r，a
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
    //由于opengl中的y轴是向下生长的，所以o < 0 为逆时针的情况
    if (o < 0.0){
        //逆时针时交换点1与点3使其成为顺时针
        std::swap(square[1],square[3]);
    }
}

int main(int argc, char** argv) {

    //Mat即矩阵，常用于存储图像
    Mat image;

    VideoCapture cap("video2.mp4");
    //VideoCapture cap(0);//第0号设备，常是摄像头

    if(!cap.isOpened())
        return -1;

    //cap.grab()即抓取下一帧
    while (cap.grab()) {
        //解码抓取的帧，并放入image中
        cap.retrieve(image);
        
        //转为灰度图
        Mat grayImage;
        cvtColor(image, grayImage, CV_RGB2GRAY);
        //进行模糊化
        Mat blurredImage;
        blur(grayImage, blurredImage, Size(5, 5));
        //使用大津算法，将图像进行二分
        Mat threshImage;
        threshold(blurredImage, threshImage, 128.0, 255.0, THRESH_OTSU);

        vector<vector<Point> > contours;
        findContours(threshImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

        vector<vector<Point2f> > squares;
        for (int i = 0; i < contours.size(); i++) {
            vector<Point> contour = contours[i];
            vector<Point> approx;
            //使用多边形近似将轮廓近似为更简单的多边形
            approxPolyDP(contour, approx, arcLength(Mat(contour), true)*0.02, true);

            //取面积足够大且为凸包的四边形作为我们的Marker
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

		if(squares.size()<1) continue;
        vector<Point2f> square = squares[0];
        drawQuad(image, square, green);

		for(int i=1;i<squares.size();i++){	//zsd
			vector<Point2f> square_temp = squares[i];
        	drawQuad(image, square_temp, green);
		}

        //旋转square使其变为顺时针方向
        clockwise(square);

        Mat marker; //存储变换后的marker图
        vector<Point2f> marker_square;  //目标形状

        //从左上点顺时针走一遍,opengl的y轴向下生长
        marker_square.push_back(Point(0,0));
        marker_square.push_back(Point(marker_width-1, 0));
        marker_square.push_back(Point(marker_width-1,marker_width-1));
        marker_square.push_back(Point(0, marker_width-1));

		//获取透视变换用矩阵
        Mat transform = getPerspectiveTransform(square, marker_square);
		//应用变换矩阵得到marker
        warpPerspective(grayImage, marker,transform, Size(marker_width,marker_width));
		//再对转换后的marker做一次二值化
        threshold(marker, marker, 125, 255, THRESH_BINARY|THRESH_OTSU);

		//跟之前一样从左上点顺时针走一遍
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
        	//根据方位左旋数组
            rotate(square.begin(), square.begin() + 1, square.end());
        }

        circle(image, square[0], 5, red);

		//加载摄像机内参数
        FileStorage fs("calibrate/out_camera_data.xml", FileStorage::READ);
        Mat intrinsics, distortion;
        //内参数(焦距、光心)矩阵
        fs["Camera_Matrix"] >> intrinsics;
        //畸变五元组
        fs["Distortion_Coefficients"] >> distortion;

		//四顶点对应的3d点集
        vector<Point3f> objectPoints;
        //注意：因为opencv中是y轴向下（x轴在y轴前），为了我们所要的视觉效果（y轴在x轴前），这里的y都取了负。
        objectPoints.push_back(Point3f(-1, 1, 0));
        objectPoints.push_back(Point3f(1, 1, 0));
        objectPoints.push_back(Point3f(1, -1, 0));
        objectPoints.push_back(Point3f(-1, -1, 0));
        Mat objectPointsMat(objectPoints);

        Mat rvec;
        Mat tvec;
		//获取外参数
        solvePnP(objectPointsMat, square, intrinsics, distortion, rvec, tvec);
		//rvec：输出变量，指相机分别绕x,y,z轴的旋转量所组成的向量
		//tcev：输出变量，指相机在世界坐标系中的坐标
        cout << "rvec: " << rvec << endl;
        cout << "tvec: " << tvec << endl;

		//比如第一个就是原点，坐标（1,0,0）的点
        vector<Point3f> line3dx = {{0, 0, 0}, {1, 0, 0}};
        vector<Point3f> line3dy = {{0, 0, 0}, {0, 1, 0}};
        vector<Point3f> line3dz = {{0, 0, 0}, {0, 0, 1}};
		//映射后的点坐标
        vector<Point2f> line2dx;
        vector<Point2f> line2dy;
        vector<Point2f> line2dz;
        //做点映射
        projectPoints(line3dx, rvec, tvec, intrinsics, distortion, line2dx);
        projectPoints(line3dy, rvec, tvec, intrinsics, distortion, line2dy);
        projectPoints(line3dz, rvec, tvec, intrinsics, distortion, line2dz);
		//绘制轴线
        line(image, line2dx[0], line2dx[1], red);
        line(image, line2dy[0], line2dy[1], blue);
        line(image, line2dz[0], line2dz[1], green);

        //新建窗口，并显示image
        cv::imshow("image窗口名", image);
        //延迟 x ms
        cv::waitKey(100);
    }
    return 0;
}

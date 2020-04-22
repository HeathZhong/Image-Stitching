// SfitMatcher.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

// void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);
// 
// typedef struct
// {
// 	Point2f left_top;
// 	Point2f left_bottom;
// 	Point2f right_top;
// 	Point2f right_bottom;
// }four_corners_t;
// 
// four_corners_t corners;
// 
// void CalcCorners(const Mat& H, const Mat& src)
// {
// 	double v2[] = { 0, 0, 1 };//左上角
// 	double v1[3];//变换后的坐标值
// 	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
// 	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
// 
// 	V1 = H * V2;
// 	//左上角(0,0,1)
// 	cout << "V2: " << V2 << endl;
// 	cout << "V1: " << V1 << endl;
// 	corners.left_top.x = v1[0] / v1[2];
// 	corners.left_top.y = v1[1] / v1[2];
// 
// 	//左下角(0,src.rows,1)
// 	v2[0] = 0;
// 	v2[1] = src.rows;
// 	v2[2] = 1;
// 	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
// 	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
// 	V1 = H * V2;
// 	corners.left_bottom.x = v1[0] / v1[2];
// 	corners.left_bottom.y = v1[1] / v1[2];
// 
// 	//右上角(src.cols,0,1)
// 	v2[0] = src.cols;
// 	v2[1] = 0;
// 	v2[2] = 1;
// 	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
// 	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
// 	V1 = H * V2;
// 	corners.right_top.x = v1[0] / v1[2];
// 	corners.right_top.y = v1[1] / v1[2];
// 
// 	//右下角(src.cols,src.rows,1)
// 	v2[0] = src.cols;
// 	v2[1] = src.rows;
// 	v2[2] = 1;
// 	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
// 	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
// 	V1 = H * V2;
// 	corners.right_bottom.x = v1[0] / v1[2];
// 	corners.right_bottom.y = v1[1] / v1[2];
// 
// }
// 
// int main(int argc, char *argv[])
// {
// 	Mat image01 = imread("Picture1.png", 1);    //右图
// 	Mat image02 = imread("Picture2.png", 1);    //左图
// 	imshow("p2", image01);
// 	imshow("p1", image02);
// 
// 	//灰度图转换  
// 	Mat image1, image2;
// 	cvtColor(image01, image1, COLOR_RGB2GRAY);
// 	cvtColor(image02, image2, COLOR_RGB2GRAY);
// 
// 	//SURF
// 	Ptr<xfeatures2d::SURF> surf;
// 	surf = xfeatures2d::SURF::create(2500);
// 	vector<KeyPoint> KeyPoint1, KeyPoint2;
// 	//提取特征点并计算描述子
// 	Mat Descript1, Descript2;
// 	surf->detectAndCompute(image1, Mat(), KeyPoint1, Descript1);
// 	surf->detectAndCompute(image2, Mat(), KeyPoint2, Descript2);
// 	vector<DMatch>matches;
// 	if (Descript1.type() != CV_32F || Descript2.type() != CV_32F)
// 	{
// 		Descript1.convertTo(Descript1, CV_32F);
// 		Descript2.convertTo(Descript2, CV_32F);
// 	}
// 	Ptr<DescriptorMatcher>matcher = DescriptorMatcher::create("FlannBased");
// 	matcher->match(Descript1, Descript2, matches);
// 	//计算特征点距离的最大值
// 	double MaxDistance = 0;
// 	for (int i = 0; i < Descript1.rows; i++)
// 	{
// 		double Dist = matches[i].distance;
// 		if (Dist > MaxDistance)
// 		{
// 			MaxDistance = Dist;
// 		}
// 	}
// 	//挑选好的匹配点
// 	vector<DMatch>OK_matches(500);
// 	for (int j = 0; j < Descript1.rows; j++)
// 	{
// 		if (matches[j].distance < 0.5*MaxDistance)
// 		{
// 			OK_matches.push_back(matches[j]);
// 		}
// 	}
// 
// // 	Mat first_match;
// // 	drawMatches(image02, KeyPoint2, image01, KeyPoint1, OK_matches, first_match);
// // 	imshow("first_match ", first_match);
// 
// 	vector<Point2f> imagePoints1(200), imagePoints2(200);
// 	int a = OK_matches.size();
// 	for (int i = 0; i < a; i++)
// 	{
// 		if (i >= OK_matches.size() || i < 0) { cout << "vetcor下标越界" << endl; break; }
// 		imagePoints2.push_back(KeyPoint2[OK_matches[i].queryIdx].pt);
// 		imagePoints1.push_back(KeyPoint1[OK_matches[i].trainIdx].pt);
// 	}
// 
// 
// 
// 	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
// 	Mat homo = findHomography(imagePoints1, imagePoints2, RANSAC);
// 	////也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
// 	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
// 	cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵      
// 
//    //计算配准图的四个顶点坐标
// 	CalcCorners(homo, image01);
// 	cout << "left_top:" << corners.left_top << endl;
// 	cout << "left_bottom:" << corners.left_bottom << endl;
// 	cout << "right_top:" << corners.right_top << endl;
// 	cout << "right_bottom:" << corners.right_bottom << endl;
// 
// 	//图像配准  
// 	Mat imageTransform1, imageTransform2;
// 	warpPerspective(image01, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
// 	//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
// 	imshow("直接经过透视矩阵变换", imageTransform1);
// 	imwrite("trans1.jpg", imageTransform1);
// 
// 
// 	//创建拼接后的图,需提前计算图的大小
// 	int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
// 	int dst_height = image02.rows;
// 
// 	Mat dst(dst_height, dst_width, CV_8UC3);
// 	dst.setTo(0);
// 
// 	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
// 	image02.copyTo(dst(Rect(0, 0, image02.cols, image02.rows)));
// 
// 	imshow("b_dst", dst);
// 
// 
// 	OptimizeSeam(image02, imageTransform1, dst);
// 
// 
// 	imshow("dst", dst);
// 	imwrite("dst.jpg", dst);
// 
// 	waitKey();
// 
// 	return 0;
// }
// 
// 
// //优化两图的连接处，使得拼接自然
// void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
// {
// 	int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  
// 
// 	double processWidth = img1.cols - start;//重叠区域的宽度  
// 	int rows = dst.rows;
// 	int cols = img1.cols; //注意，是列数*通道数
// 	double alpha = 1;//img1中像素的权重  
// 	for (int i = 0; i < rows; i++)
// 	{
// 		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
// 		uchar* t = trans.ptr<uchar>(i);
// 		uchar* d = dst.ptr<uchar>(i);
// 		for (int j = start; j < cols; j++)
// 		{
// 			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
// 			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
// 			{
// 				alpha = 1;
// 			}
// 			else
// 			{
// 				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
// 				alpha = (processWidth - (j - start)) / processWidth;
// 			}
// 
// 			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
// 			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
// 			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
// 
// 		}
// 	}
// 
// }


typedef struct  
{
	Point2f Left_Top;
	Point2f Left_Bottom;
	Point2f Right_Top;
	Point2f Right_Bottom;
}Four_Corners;
Four_Corners Corners;

void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	Corners.Left_Top.x = v1[0] / v1[2];
	Corners.Left_Top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	Corners.Left_Bottom.x = v1[0] / v1[2];
	Corners.Left_Bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	Corners.Right_Top.x = v1[0] / v1[2];
	Corners.Right_Top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	Corners.Right_Bottom.x = v1[0] / v1[2];
	Corners.Right_Bottom.y = v1[1] / v1[2];

}

//优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(Corners.Left_Top.x, Corners.Left_Bottom.x);//开始位置，即重叠区域的左边界  

	double processWidth = img1.cols - start;//重叠区域的宽度  
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}

int main()
{
	Mat SrcImage1, SrcImage2;
	Mat GaussBlurImage1, GaussBlurImage2;
	SrcImage1 = imread("Picture1.png", IMREAD_COLOR);
	SrcImage2 = imread("Picture2.png", IMREAD_COLOR);
	if (SrcImage1.data == nullptr || SrcImage2.data == nullptr)
	{
		cout << "Image load error..." << endl;
		return 0;
	}

	Ptr<xfeatures2d::SIFT> sift;
	sift = xfeatures2d::SIFT::create(200);
	//ORB
	// 	Ptr<ORB> orb;
	// 	orb = ORB::create();
	//SURF
	//Ptr<xfeatures2d::SURF> surf;
	//surf = xfeatures2d::SURF::create(1500);
	//提取特征点
	vector<KeyPoint> KeyPoint1, KeyPoint2;
// 	sift->detect(SrcImage1, KeyPoint1);
// 	sift->detect(SrcImage2, KeyPoint2);
// 	//画特征点
// 	Mat KeyPointImage1, KeyPointImage2;
// 	drawKeypoints(SrcImage1, KeyPoint1, KeyPointImage1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
// 	drawKeypoints(SrcImage2, KeyPoint2, KeyPointImage2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
// 	//显示特征点
// 	imshow("KeyPoint1", KeyPointImage1);
// 	imshow("Keypoint2", KeyPointImage2);
	//提取特征点并计算描述子
	Mat Descript1, Descript2;
	sift->detectAndCompute(SrcImage1, Mat(), KeyPoint1, Descript1);
	sift->detectAndCompute(SrcImage2, Mat(), KeyPoint2, Descript2);
	vector<DMatch>matches;
	if (Descript1.type() != CV_32F || Descript2.type() != CV_32F)
	{
		Descript1.convertTo(Descript1, CV_32F);
		Descript2.convertTo(Descript2, CV_32F);
	}
	Ptr<DescriptorMatcher>matcher = DescriptorMatcher::create("FlannBased");
	matcher->match(Descript1, Descript2, matches);
	//计算特征点距离的最大值
	double MaxDistance = 0;
	for (int i = 0; i < Descript1.rows; i++)
	{
		double Dist = matches[i].distance;
		if (Dist > MaxDistance)
		{
			MaxDistance = Dist;
		}
	}
	//挑选好的匹配点
	vector<DMatch>OK_matches;
	for (int j = 0; j < Descript1.rows; j++)
	{
		if (matches[j].distance < 0.5*MaxDistance)
		{
			OK_matches.push_back(matches[j]);
		}
	}
	//图像配准
	//转换类型为Point2f
	vector<Point2f>ImagePoint1, ImagePoint2;
	for (int i = 0; i < OK_matches.size(); i++)
	{
		ImagePoint1.push_back(KeyPoint1[OK_matches[i].trainIdx].pt);
		ImagePoint2.push_back(KeyPoint2[OK_matches[i].queryIdx].pt);
	}
	//转换图像1到图像2的映射矩阵
	Mat Homo = findHomography(ImagePoint1, ImagePoint2, RANSAC);
	cout << "Homo = \n" << Homo << endl;
	//图像配准  
	Mat ImageTransform1, ImageTransform2;
	warpPerspective(SrcImage1, ImageTransform1, Homo, Size(MAX(Corners.Right_Top.x, Corners.Right_Bottom.x), SrcImage2.rows));
	CalcCorners(Homo, SrcImage1);
	cout << "left_top:" << Corners.Left_Top << endl;
	cout << "left_bottom:" << Corners.Left_Bottom << endl;
	cout << "right_top:" << Corners.Right_Top << endl;
	cout << "right_bottom:" << Corners.Right_Bottom << endl;
	imshow("直接经过透视矩阵变换", ImageTransform1);
	//imwrite("trans1.jpg", imageTransform1);
	//创建拼接后的图,需提前计算图的大小
	int dst_width = ImageTransform1.cols;  //取最右点的长度为拼接图的长度
	int dst_height = SrcImage2.rows;
	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);
	ImageTransform1.copyTo(dst(Rect(0, 0, ImageTransform1.cols, ImageTransform1.rows)));
	SrcImage2.copyTo(dst(Rect(0, 0, SrcImage2.cols, SrcImage2.rows)));
	imshow("b_dst", dst);
	OptimizeSeam(SrcImage2, ImageTransform1, dst);
	imshow("dst", dst);
	Mat OutputImage;
	drawMatches(SrcImage1, KeyPoint1, SrcImage2, KeyPoint2, OK_matches, OutputImage);
	imshow("OutPutImage", OutputImage);
	waitKey(0);

}




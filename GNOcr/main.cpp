/**************************************************************
* 燃气表数字识别程序
* Filename:main.cpp
* @author Sunie
* Date:2016.06.08
*Description:主要实现自然场景下的燃气表的数字识别
*主要使用KNN算法
***************************************************************/
#include<opencv.hpp>
#include <iostream>
#include<stdio.h>
#include<ml.hpp>
#include<ml.h>

using namespace cv;
using namespace std;
using namespace ml;


Ptr<TrainData> prepare_train_data();//测试数据的准备
Mat g_grayImage;//灰度图
Mat g_srcImage;//原图
Mat g_bilateralImage;//双边滤波图
Mat g_blackImage;//二值化图
Mat g_dstImage;
RNG g_rng(12345);
vector<vector<Point>> g_vContours;//边框存储
vector<Vec4i> g_vHierarchy;
RotatedRect g_mr;


//Sobel边缘检测相关变量
Mat g_sobelGradient_X, g_sobelGradient_Y;
Mat g_sobelAbsGradient_X, g_sobelAbsGradient_Y;
int g_sobelKernelSize = 1;//TrackBar位置参数  

						  //训练数据相关
Mat t_trainData = Mat(70, 323, CV_32FC1);//训练数据
Mat_<int>t_trainclass;//训练数据标签


bool verifySizes(RotatedRect mr);//判断是否为数字
void bubbleSort(int arr1[], int arr2[]);
void ShowHelpText();

int main() {

	ShowHelpText();

	//载入原图
	g_srcImage = imread("01781-2.jpg");

	//设置掩膜
	g_blackImage.create(g_srcImage.rows, g_srcImage.cols, g_srcImage.type());
	//灰度化
	if (g_srcImage.channels() == 3)
		cvtColor(g_srcImage, g_grayImage, COLOR_RGB2GRAY);
	else
		g_grayImage = g_srcImage;

	bilateralFilter(g_grayImage, g_bilateralImage, 4, 8, 2);//双边滤波
	imshow("双边滤波", g_bilateralImage);//双边滤波显示
	threshold(g_bilateralImage, g_blackImage, 115, 255, THRESH_BINARY);//二值化

	namedWindow("gray_mat");
	imshow("gray_mat", g_grayImage);
	imshow("black_mat", g_blackImage);


	//=============================【查找轮廓】=================================
	Mat g_result;
	g_blackImage.copyTo(g_result);
	g_blackImage.copyTo(g_dstImage);
	findContours(g_blackImage, g_vContours, g_vHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//寻找轮廓(外轮廓)
																							  //==========================================================================


																							  //=============================【绘制轮廓】=================================
	Mat drawing = Mat::zeros(g_blackImage.size(), CV_8UC3);
	for (int i = 0; i < g_vContours.size(); i++)
	{
		Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//任意值
		drawContours(drawing, g_vContours, i, color, 1, 8, g_vHierarchy, 0, Point());
	}
	//显示效果图
	imshow("轮廓检测", drawing);
	//=============================================================================


	//迭代寻找出的轮廓
	vector<vector<Point> >::iterator itc = g_vContours.begin();
	vector<RotatedRect> rects;


	//=============================【筛选不符要求的轮廓】=================================
	int t = 0;
	while (itc != g_vContours.end())
	{
		//创建最小外接矩形
		RotatedRect mr = minAreaRect(Mat(*itc));
		//测试用代码
		/*	 int area = mr.size.height * mr.size.width;
		float r = (float)mr.size.width / (float)mr.size.height;
		cout << "宽" << mr.size.width << "\n";
		cout << "高" << mr.size.height << "\n";
		cout << "面积" << area << "\n";
		cout << "比例" << r << "\n";*/

		//取出外接矩形轮廓信息
		if (!verifySizes(mr))
		{
			itc = g_vContours.erase(itc);
		}
		else
		{
			++itc;
			rects.push_back(mr);
		}
	}
	//===========================================================================


	//处理轮廓最小外接矩形位置信息
	vector<vector<Point> > g_contours_poly(g_vContours.size());
	vector<Rect> g_boundRect(g_vContours.size());

	//为了顺序输出数字的位置信息
	int position[6];
	int positionNumber[6] = { 0, 1, 2, 3, 4, 5 };


	//=============================【遍历存储位置信息的容器，并分割图像】=================================
	for (int i = 0; i < g_vContours.size(); i++)
	{

		g_boundRect[i] = boundingRect(Mat(g_vContours[i]));//获取区域的边界左上和右下坐标

		Rect g_box;

		//指定分割处理区域范围
		g_box.x = g_boundRect[i].tl().x;
		g_box.y = g_boundRect[i].tl().y;
		g_box.width = g_boundRect[i].br().x - g_boundRect[i].tl().x;
		g_box.height = g_boundRect[i].br().y - g_boundRect[i].tl().y;
		position[i] = g_box.x;//获取左上角x坐标
							  //分割指定区域，在指定区域内对图像进行处理
		Mat g_imageROI(g_dstImage, g_box);
		//归一化大小为17*19
		Mat g_imageROIResize;
		resize(g_imageROI, g_imageROIResize, cvSize(17, 19));
		Mat g_imageROIReshape = g_imageROIResize.reshape(0, 1);
		//cout << g_imageROIReshape << "\n";
		/*float response = model->predict(g_imageROIReshape);
		cout << response << "\n";*/

		char g_chFile_name[200];//归一化输出变量
								//归一化输出内容
		sprintf(g_chFile_name, "%s%d%s", "ddd_", i, ".bmp");

		String g_file_name = g_chFile_name;//数组类型转换为字符类型
		imwrite(g_file_name, g_imageROIResize);//写入文件到文件夹

	}
	//==========================================================================


	//冒泡排序得出正确的排序
	bubbleSort(position, positionNumber);

	//=============================【KNN】=================================
	Ptr<KNearest> model = KNearest::create();//创建KNN
	Ptr<TrainData> tData = prepare_train_data();//获得训练数据
	model->setDefaultK(3);    //设定k值
	model->setIsClassifier(true);
	model->train(tData);//训练
	cout << "燃气表字符为" << endl;

	//根据位置信息顺序读取上面写入的图片
	for (int i = 0; i < 6; i++)
	{
		//归一化路径地址
		char g_chtestfile_name[255];
		sprintf(g_chtestfile_name, "%s%d%s", "ddd_", positionNumber[i], ".bmp");
		String g_testfile_name = g_chtestfile_name;
		//读取测试图片(灰度图)
		Mat g_testfile1 = Mat(19, 17, CV_32FC1);
		g_testfile1 = imread(g_testfile_name, 0);
		//二值化测试图片
		Mat g_testfileTh = Mat(19, 17, CV_32FC1);
		threshold(g_testfile1, g_testfileTh, 115, 255, THRESH_BINARY);
		//归一化测试数据
		Mat g_testfileRe = Mat(1, 323, CV_32FC1);
		g_testfileRe = g_testfileTh.reshape(0, 1);
		Mat_<float>g_testfile(1, 323);
		g_testfileRe.copyTo(g_testfile);
		//检测
		float response = model->predict(g_testfile);
		cout << response << endl;
	}
	//========================================================================

	waitKey(0);
	return 0;
}

//------------------------------------【verifySizes( )函数】------------------------------------
//		 描述：判断是否为正确的矩阵
//----------------------------------------------------------------------------------------------
bool verifySizes(RotatedRect mr)
{
	int min = 120;//最小面积限制
	int max = 2000;//最大面积限制
	float rmin = 0.15;//最小高度和宽度的比例限制
	float rmax = 1.3;//最大高度和宽度的比例限制
	int area = mr.size.height * mr.size.width;//矩形框面积大小
	float r = (float)mr.size.width / (float)mr.size.height;//矩形框高度和宽度的比例

														   /*条件判断，小于最小面积限制、大于最大面积限制、
														   小于最小比例限制和大于最大比例限制，
														   四个条件只要满足一个就返回false*/
	if ((area < min || area > max) || (r < rmin || r > rmax))
	{
		return false;
	}
	else
	{
		return true;
	}
}

//--------------------------------【prepare_train_data( )函数】---------------------------------
//		 描述：准备训练数据
//----------------------------------------------------------------------------------------------
Ptr<TrainData> prepare_train_data()
{
	char t_file_path[] = "train/";//地址码头部
	int t_train_samples = 10;//文件数量
	int t_classes = 10;//文件夹数量

	int t_trainclassLabels[70];//训练数据标签
							   //基本数据
	Mat t_srcImage;//原图
	Mat t_blackImage;//二值化图
	Mat t_blackImageResize;
	Mat t_reshapeImage;//归一化数据
	char t_file[255];
	int i, j;
	int t = 0;

	for (i = 0; i < t_classes; i++)
	{
		if (i == 2 || i == 5 || i == 6) {
			continue;
		}
		else {
			for (j = 0; j < t_train_samples; j++)
			{
				//读取文件
				sprintf(t_file, "%s%d/0%d.bmp", t_file_path, i, j);//归一化输出地址
				string t_file_name = t_file;//输出地址转化为字符串类型
				t_srcImage = imread(t_file_name, 0);


				//二值化
				threshold(t_srcImage, t_blackImage, 115, 255, THRESH_BINARY);
				//归一化训练数据
				t_reshapeImage = t_blackImage.reshape(0, 1);
				//将数据赋给相应行
				t_reshapeImage.row(0).copyTo(t_trainData.row(t));
				t_trainclassLabels[t] = i;//训练标签数组设置
				t++;

			}
		}
	}
	//cout << t << "\n";
	t_trainclass = Mat(70, 1, CV_32SC1, t_trainclassLabels);//训练标签数组转换成矩阵
	Ptr<TrainData> tData = TrainData::create(t_trainData, ROW_SAMPLE, t_trainclass);//训练举证规范化

	return tData;//返回训练矩阵
}

//-------------------------------------【bubbleSort( )函数】------------------------------------
//		 描述：冒泡排序更迭位置信息
//----------------------------------------------------------------------------------------------
void bubbleSort(int arr1[], int arr2[])
{
	for (int i = 0; i < 6; i++) {
		//比较两个相邻的元素   
		for (int j = 0; j < 6 - i - 1; j++) {
			if (arr1[j] > arr1[j + 1]) {
				int t = arr1[j];
				int t2 = arr2[j];
				arr1[j] = arr1[j + 1];
				arr2[j] = arr2[j + 1];
				arr1[j + 1] = t;
				arr2[j + 1] = t2;
			}
		}
	}
}

//--------------------------------------【ShowHelpText( )函数】---------------------------------
//		 描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t毕设题目：自然场景下的文字识别\n");
	printf("\n\n\t\t\t姓名：李阳\n");
	printf("\n\n\t\t\t学号：12122326\n");
	printf("\n\n\t\t\t指导老师：沈为\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
	printf("\n\n  ----------------------------------------------------------------------------\n");
}
// QuickshiftSegmentation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>


using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
	Mat image = imread("D:\\nihalsid\\Label23D-PreprocessingScripts\\files\\segmentation\\input\\frame-000005.color.jpg", IMREAD_COLOR);
	//Mat image = imread("D:\\nihalsid\\Label23D-PreprocessingScripts\\files\\segmentation\\input-min\\test.png", IMREAD_COLOR);
	Mat image_as_float(image.rows, image.cols, CV_32FC3);
	double *image_float_data = (double*) malloc(sizeof(double) * image.rows * image.cols * image.channels());
	double *distances = nullptr;
	int *parents = nullptr;
	image.convertTo(image_as_float, CV_32FC3);
	for (int c = 0; c < image.channels(); c++) {
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				image_float_data[(image.rows * image.cols) * c + i * image.cols + j] = (double)image_as_float.at<Vec3f>(i, j)[c];
			}
		}
	}
	return 0;
}

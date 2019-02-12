// QuickshiftSegmentation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include "quickshift_common.h"
#include <assert.h>

using namespace cv;
using namespace std;

void image_from_data(image_t & im, const Mat& image) {
	im.N1 = image.rows;
	im.N2 = image.cols;
	im.K = image.channels();

	im.I = (float *)calloc(im.N1*im.N2*im.K, sizeof(float));
	for (int k = 0; k < im.K; k++)
		for (int col = 0; col < im.N2; col++)
			for (int row = 0; row < im.N1; row++)
			{
				im.I[row + col * im.N1 + k * im.N1*im.N2] = max(0., 32. * image.at<Vec3b>(row, col)[k] / 255. + double(rand()) / RAND_MAX / 10 - 0.1); // Scale 0-32
			}
}

int * map_to_flatmap(float * map, unsigned int size)
{
	/********** Flatmap **********/
	int *flatmap = (int *)malloc(size * sizeof(int));
	for (unsigned int p = 0; p < size; p++)
	{
		flatmap[p] = map[p];
	}

	bool changed = true;
	while (changed)
	{
		changed = false;
		for (unsigned int p = 0; p < size; p++)
		{
			changed = changed || (flatmap[p] != flatmap[flatmap[p]]);
			flatmap[p] = flatmap[flatmap[p]];
		}
	}

	/* Consistency check */
	for (unsigned int p = 0; p < size; p++)
		assert(flatmap[p] == flatmap[flatmap[p]]);

	return flatmap;
}

Mat imseg_assignments(image_t im, int* flatmap) {
	Mat outImage(im.N1, im.N2, CV_16UC1);
	int ctr = 0;
	for (int r = 0; r < im.N1; r++) {
		for (int c = 0; c < im.N2; c++) {
			int p = r + c * im.N1;
			if (flatmap[p] == p)
				outImage.at<uint16_t>(r, c) = ctr++;
		}
	}

	for (int r = 0; r < im.N1; r++) {
		for (int c = 0; c < im.N2; c++) {
			int p = r + c * im.N1;
			int r2 = flatmap[p] % im.N1;
			int c2 = flatmap[p] / im.N1;
			outImage.at<uint16_t>(r, c) = outImage.at<uint16_t>(r2, c2);
		}
	}
	return outImage;
}

image_t imseg(image_t im, int * flatmap)
{
	/********** Mean Color **********/
	float * meancolor = (float *)calloc(im.N1*im.N2*im.K, sizeof(float));
	float * counts = (float *)calloc(im.N1*im.N2, sizeof(float));

	for (int p = 0; p < im.N1*im.N2; p++)
	{
		counts[flatmap[p]]++;
		for (int k = 0; k < im.K; k++)
			meancolor[flatmap[p] + k * im.N1*im.N2] += im.I[p + k * im.N1*im.N2];
	}

	int roots = 0;
	for (int p = 0; p < im.N1*im.N2; p++)
	{
		if (flatmap[p] == p)
			roots++;
	}
	//printf("Roots: %d\n", roots);

	int nonzero = 0;
	for (int p = 0; p < im.N1*im.N2; p++)
	{
		if (counts[p] > 0)
		{
			nonzero++;
			for (int k = 0; k < im.K; k++)
				meancolor[p + k * im.N1*im.N2] /= counts[p];
		}
	}
	printf("Nonzero: %d\n", nonzero);
	assert(roots == nonzero);


	/********** Create output image **********/
	image_t imout = im;
	imout.I = (float *)calloc(im.N1*im.N2*im.K, sizeof(float));
	for (int p = 0; p < im.N1*im.N2; p++)
		for (int k = 0; k < im.K; k++)
			imout.I[p + k * im.N1*im.N2] = meancolor[flatmap[p] + k * im.N1*im.N2];

	free(meancolor);
	free(counts);

	return imout;
}


void quickshift_wrapper(Mat& image, float tau, float sigma) {
	float *map, *E, *gaps;
	int * flatmap;
	image_t imout;
	unsigned int totaltimer;

	int dims[3] = {image.rows, image.cols, image.channels()};

	map = (float *)calloc(dims[0] * dims[1], sizeof(float));
	gaps = (float *)calloc(dims[0] * dims[1], sizeof(float));
	E = (float *)calloc(dims[0] * dims[1], sizeof(float));

	image_t im;
	image_from_data(im, image);
	quickshift_gpu(im, sigma, tau, map, gaps, E);
	for (int p = 0; p < im.N1*im.N2; p++)
		if (map[p] == p) {
			//cout << "Map[" << p << "]: " << map[p] << " == " << p << " but gaps[" << p << "] = "<< gaps[p] <<" != Inf"   << endl;
		}

	flatmap = map_to_flatmap(map, im.N1*im.N2);
	imout = imseg(im, flatmap);

	Mat label_image(dims[0], dims[1], CV_8UC3);
	for (int k = 0; k < im.K; k++)
		for (int col = 0; col < im.N2; col++)
			for (int row = 0; row < im.N1; row++)
			{
				label_image.at<Vec3b>(row, col)[k] = uchar (imout.I[row + col * im.N1 + k * im.N1*im.N2] * 255./32.);
			}

	//namedWindow("Labels");
	//imshow("Labels", label_image);
	//waitKey(0);

	Mat lbl_idx_image = imseg_assignments(im, flatmap);
	for (int r = 0; r < im.N1 - 1; r++) {
		for (int c = 0; c < im.N2 - 1; c++) {
			if ((lbl_idx_image.at<uint16_t>(r, c) != lbl_idx_image.at<uint16_t>(r, c + 1)) || (lbl_idx_image.at<uint16_t>(r, c) != lbl_idx_image.at<uint16_t>(r + 1, c))) {
				image.at<Vec3b>(r, c)[0] = 0;
				image.at<Vec3b>(r, c)[1] = 255;
				image.at<Vec3b>(r, c)[2] = 255;
			}
		}
	}
	namedWindow("Seg");
	imshow("Seg", image);
	waitKey(0);

	assert(imout.N1 == dims[0]);
	assert(imout.N2 == dims[1]);
	assert(imout.K == 3);
}

int main(int argc, const char * argv[]) {
	Mat image = imread("D:\\nihalsid\\Label23D-PreprocessingScripts\\files\\segmentation\\input\\frame-000005.color.jpg", IMREAD_COLOR);
	//Mat image = imread("D:\\nihalsid\\Label23D-PreprocessingScripts\\files\\segmentation\\input-min\\test.png", IMREAD_COLOR);
	quickshift_wrapper(image, 5.f, 3.f);
	return 0;
}

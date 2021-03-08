// VectorOperations.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <cstdlib>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

void ComputeMaskWithIntrinsics256(std::vector<float>& image);
void ComputeMask();

int main()
{
	// check if using optimized opencv
	bool usesOptimized = cv::useOptimized();
	std::cout << std::boolalpha << usesOptimized << std::endl;

	ComputeMask();

	system("pause");
}

void ComputeMask()
{
	srand(time(NULL));

	//
	// arrays initialization
	//
	int size = std::pow(2, 14); // 2^13 = 8192
	int N = size * size;		// 8192 x 8192
	std::cout << "matrix size: " << size << " x " << size << std::endl;

	std::vector<float> imageArr(N);
	for (int i = 0; i < N; i++)
	{
		imageArr[i] = (rand() % 100) / 100.0;
	}

	cv::Mat image(imageArr);
	std::vector<float> imageVector(imageArr);
	std::vector<float> intrnsVector(imageArr);
	cv::Mat mask1;
	double threshold = 0.5;
	int const maxBinaryValue = 1;
	int const thresholdType = 1; // Threshold Binary
	int64 start1, start2, start3, end1, end2, end3 = 0;
	double time1 = 0, time2 = 0, time3 = 0;

	//
	// operation on image (get binary mask with threshold)
	//
	int iterations = 30;
	for (int i = 0; i < iterations; i++) {

		std::cout << i << std::endl;

		// ------------- OpenCV --------------------------------------------------------- //
		start1 = cv::getTickCount();
		cv::threshold(image, mask1, threshold, maxBinaryValue, thresholdType);
		end1 = cv::getTickCount();

		// ------------- With Intrinsics ------------------------------------------------ //
		start2 = cv::getTickCount();
		ComputeMaskWithIntrinsics256(intrnsVector);
		end2 = cv::getTickCount();

		// ------------- Without OpenCV ------------------------------------------------- //
		start3 = cv::getTickCount();
		for (int i = 0; i < N; i++) {
			imageVector[i] = (imageVector[i] > threshold) ? 1.0 : 0.0;
		}
		end3 = cv::getTickCount();
		// ------------------------------------------------------------------------------ //

		// first is warmup
		if (i > 0) {
			time1 += end1 - start1;
			time2 += end2 - start2;
			time3 += end3 - start3;
		}
	}

	std::cout << "OpenCV Threshold binary mask tick count: " << time1 / cv::getTickFrequency() << std::endl;
	std::cout << "Intrinsics Threshold binary mask tick count: " << time2 / cv::getTickFrequency() << std::endl;
	std::cout << "No OpenCV Threshold binary mask tick count: " << time3 / cv::getTickFrequency() << std::endl;

	//
	// show result
	//
	//cv::resize(mask1, mask1, cv::Size(size, size));
	//cv::imshow("Mask OpenCV", mask1);
	//cv::waitKey(0);

	//cv::Mat mask2(imageVector);
	//cv::resize(mask2, mask2, cv::Size(size, size));
	//cv::imshow("Mask No OpenCV", mask2);
	//cv::waitKey(0);

	//cv::Mat mask3(intrnsVector);
	//cv::resize(mask3, mask3, cv::Size(size, size));
	//cv::imshow("Intrinsics Mask", mask3);
	//cv::waitKey(0);

	//cv::destroyAllWindows();
}

// Use AVX2 Vector co-processor to handle 4 locations at once
void ComputeMaskWithIntrinsics256(std::vector<float>& image)
{
	// 64-bit "double" registers
	__m256 _threshold, _mask, _input, _c, _one;
	_one = _mm256_set1_ps(1);

	// Expand constants into vectors of constants
	// one_dot_five =	|0.5|0.5|0.5|0.5|		
	_threshold = _mm256_set1_ps(0.5);

	// load last four elements for later use
	for (size_t i = 0; i < image.size() - 8; i += 8)
	{
		// load values into registers
		_input = _mm256_load_ps(image.data() + i);

		// compare with 0.5
		_mask = _mm256_cmp_ps(_input, _threshold, _CMP_GT_OQ);

		// replace NaNs with 1
		_c = _mm256_and_ps(_one, _mask);

		// save result
		image[i + 0] = _c.m256_f32[0];
		image[i + 1] = _c.m256_f32[1];
		image[i + 2] = _c.m256_f32[2];
		image[i + 3] = _c.m256_f32[3];
		image[i + 4] = _c.m256_f32[4];
		image[i + 5] = _c.m256_f32[5];
		image[i + 6] = _c.m256_f32[6];
		image[i + 7] = _c.m256_f32[7];
	}
}



#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
using namespace cv;

typedef std::vector<std::vector<float>> matrixF; 
typedef std::vector<std::vector<int>> matrixI;

void imshowHSV(cv::Mat& image);
void printArr(matrixF arr);
int getLowThresh(matrixF arr);
std::vector<float> getPixelThresh(matrixF normImg);
std::vector<float> getPixelSTD(matrixF normImg, std::vector<float> pixelAvg);
matrixF normalize(matrixI arr);
void displayOneFile();
std::vector<float> getSuperPixelMean(matrixF normImg);
std::vector<float> getSuperPixelSTD(matrixF normImg, std::vector<float> meanImg);
matrixF upScale(matrixF arr);
Mat hystThreshold(Mat image);
RNG rng(12345);
Mat enhanceHeat(Mat heatImg);
Mat loadDataToMat(matrixF dataArr, int iFrame);
void sortRect(std::vector<Rect> boundRect, std::vector<Rect>* onePerson, std::vector<Rect>* twoPerson, int meeting);
void getMeanRect(std::vector<Rect> onePerson, std::vector<Rect> twoPerson);
int getParticipants(std::vector<Rect> boundRect);
void loadData(std::vector<matrixI>* inData, std::vector<matrixF>* normData, std::vector<std::string> fileNames);
void initialThresholdImage(std::vector<Mat>* heatFrames, std::vector<matrixF> normData, int meeting);

int main()
{
	//displayOneFile();

	namedWindow("Tracker", CV_WINDOW_NORMAL);
	namedWindow("Original", CV_WINDOW_NORMAL);

	// All meetings files from:
	// https://github.com/bsirmacek/heat-sensor-data
	std::vector<std::string> fileNames = { "no_present_before.txt", "present_at_1.txt", "present_at_1_2.txt",
	"present_at_1_2_3.txt", "present_at_1_2_3_4.txt", "present_at_2_3_4.txt", "present_at_3_4.txt",
	"present_at_4.txt", "no_present_after.txt", "2_no_present.txt", "2_present_at_1_2.txt", "2_present_at_2_3.txt",
	"2_present_at_3_4.txt", "2_present_at_3.txt"};

	// Holds raw data straight from file
	std::vector<matrixI> input;

	// Holds normalized data from input
	std::vector<matrixF> normData;

	// Holds heat data from the files during calculations
	std::vector<Mat> heatFrames;

	// Holds heat data from the files without calculations, for comparison
	std::vector<Mat> originalFrames;

	// Holds frames when all calculations are done to show the end result
	std::vector<Mat> showFrames;
	
	// Holds information of the tracking rectangle size for each frame. 
	// Used for calculating the mean of the rectangles. 
	std::vector<Rect> onePersonRectSize;
	std::vector<Rect> twoPersonRectSize;

	input.resize(fileNames.size());
	normData.resize(fileNames.size());

	// Load from files
	loadData(&input, &normData, fileNames);

	// Each meeting is one file from above
	for (int meeting = 0; meeting < fileNames.size(); meeting++)
	{

		// Takes the normalized data and loads it into cv::Mat format after a thresholding has been performed
		initialThresholdImage(&heatFrames, normData, meeting);
		

		for (int iFrame = 0; iFrame < heatFrames.size(); iFrame++)
		{
			Mat previousFrame, currentFrame, nextFrame, originalFrame;
			Mat newFrame = Mat::zeros(heatFrames[iFrame].rows, heatFrames[iFrame].cols, CV_8UC1);
			currentFrame = heatFrames[iFrame];
			if (iFrame != 0)
				previousFrame = heatFrames[iFrame - (int)1];
			if (iFrame != heatFrames.size() - 1)
				nextFrame = heatFrames[iFrame + 1];

			double minThreshold;
			// Get the minimum value in the frame to "zero" the background so that only relevant heatsignatures is shown
			minMaxLoc(currentFrame, &minThreshold);
			minThreshold += 20;

			// Aditional filtering, if heat signature appears in less than 3 frames, disregard it.
			// Also "zero" the background
			for (int y = 0; y < currentFrame.rows; y++)
			{
				for (int x = 0; x < currentFrame.cols; x++)
				{

					Vec<uchar, 1> currentPixel = currentFrame.at<Vec<uchar, 1>>(Point(x, y));
					Vec<uchar, 1> newPixel = newFrame.at<Vec<uchar, 1>>(Point(x, y));
					Vec<uchar, 1> previousPixel;
					Vec<uchar, 1> nextPixel;

					if (!previousFrame.empty())
						previousPixel = previousFrame.at<Vec<uchar, 1>>(Point(x, y));
					if (!nextFrame.empty())
						nextPixel = nextFrame.at<Vec<uchar, 1>>(Point(x, y));

					if (iFrame == 0)
					{
						if (currentPixel[0] > minThreshold&& nextPixel[0] > minThreshold)
							newPixel[0] = currentPixel[0];
						else
							newPixel[0] = 0;
					}
					else if (iFrame == heatFrames.size() - 1)
					{
						if (currentPixel[0] > minThreshold&& previousPixel[0] > minThreshold)
							newPixel[0] = currentPixel[0];
						else
							newPixel[0] = 0;
					}
					else
					{
						if ((currentPixel[0] > minThreshold&& nextPixel[0] > minThreshold&& previousPixel[0] > minThreshold) ||
							currentPixel[0] < minThreshold && nextPixel[0] > minThreshold&& previousPixel[0] > minThreshold)
							newPixel[0] = currentPixel[0];
						else
							newPixel[0] = 0;
					}

					newFrame.at<Vec<uchar, 1>>(Point(x, y)) = newPixel;
				}
			}



			// Additional thesholding for the edge detection, converts gradient pixels to binary for easier edge detection
			Mat currentFrameHyst = hystThreshold(newFrame);

			Mat cannyOutput;
			Canny(currentFrameHyst, cannyOutput, 240, 255);

			std::vector<std::vector<Point>> contours;
			Mat hierarchy;
			findContours(cannyOutput, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			std::vector<std::vector<Point>> contoursPoly(contours.size());
			std::vector<Rect> boundRect(contours.size());
			std::vector<Point2f> centers(contours.size());
			std::vector<float> radius(contours.size());

			Mat trackingRectangles = Mat::zeros(cannyOutput.size(), CV_8UC3);

			// Calculate bounding rectangles for each heat signature and their centers.
			for (size_t i = 0; i < contours.size(); i++)
			{
				approxPolyDP(contours[i], contoursPoly[i], 1, true);
				boundRect[i] = boundingRect(contoursPoly[i]);
				minEnclosingCircle(contoursPoly[i], centers[i], radius[i]);
			}

			// Visually create tracking rectangles
			for (size_t i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(0, 0, 200);
				rectangle(trackingRectangles, boundRect[i].tl(), boundRect[i].br(), color, 1);
			}


			// Used forgetting the mean size of rectangles (only needs to be run once)
			//sortRect(boundRect, &onePersonRectSize, &twoPersonRectSize, meeting);

			std::string participants = std::to_string(getParticipants(boundRect));
			currentFrame = enhanceHeat(currentFrame);

			Mat showFrame;
			cvtColor(currentFrame, currentFrame, CV_GRAY2BGR);
			//cvtColor(originalFrame, originalFrame, CV_GRAY2BGR);
			addWeighted(currentFrame, 1.0, trackingRectangles, 0.5, 0.0, showFrame);
			//hconcat(showFrame, originalFrame, showFrame);
			resize(showFrame, showFrame, Size(showFrame.rows * 4, showFrame.cols * 4), 0, 0, INTER_CUBIC);
			putText(showFrame, participants, Point(80, 30), 0, FONT_HERSHEY_PLAIN, Scalar(0, 0, 200));

			// Store showFrame, which contains heat signatures (before hystThreshold is done), tracking rectangles and
			// text with how many participants.
			showFrames.push_back(showFrame);
			originalFrames.push_back(loadDataToMat(normData[meeting], iFrame));

		}
		heatFrames.clear();
	}

	// Saves mean rectangle size to a file, (only needs to be run once)
	//getMeanRect(onePersonRectSize, twoPersonRectSize);

	// Show final heat images and original as comparison
	for (int frame = 0; frame < showFrames.size(); frame++)
	{
		imshow("Tracker", showFrames[frame]);
		imshow("Original", originalFrames[frame]);
		waitKey(100);
	}

	return 0;
}


/* ==================================================================================*/
/* ================================== Functions =====================================*/
/* ==================================================================================*/

int getParticipants(std::vector<Rect> boundRect)
{
	int participants = 0;
	int meanOnePerson = 170;
	int meanTwoPerson = 200;

	for (int i = 0; i < boundRect.size(); i++)
	{
		int rectSize = boundRect[i].height * boundRect[i].width;
		if (rectSize < (((meanTwoPerson - meanOnePerson)/2)+meanOnePerson+10))
			participants += 1;
		else if (rectSize >= (meanTwoPerson - meanOnePerson))
			participants += 2;
	}

	return participants;
}

void getMeanRect(std::vector<Rect> onePerson, std::vector<Rect> twoPerson)
{
	int meanOneRectSize = 0, meanTwoRectSize = 0;
	for (int i = 0; i < onePerson.size(); i++)
	{
		meanOneRectSize += onePerson[i].height * onePerson[i].width;
	}

	for (int i = 0; i < twoPerson.size(); i++)
	{
		meanTwoRectSize += twoPerson[i].height * twoPerson[i].width;
	}
	meanOneRectSize = meanOneRectSize / onePerson.size();
	meanTwoRectSize = meanTwoRectSize / twoPerson.size();

	std::ofstream myfile("rectMean.txt");
	if (myfile.is_open())
	{
		myfile << meanOneRectSize << "," << meanTwoRectSize << "\n";
		myfile.close();
	}
	else std::cout << "Unable to open file";
}

void sortRect(std::vector<Rect> boundRect, std::vector<Rect>* onePerson, std::vector<Rect>* twoPerson, int meeting)
{

	if (!boundRect.empty())
	{
		if (meeting == 1 || meeting == 2 || meeting == 6 || meeting == 7 || meeting == 10 || meeting == 11 || meeting == 12)
		{
			for (int rect = 0; rect < boundRect.size(); rect++)
			{
				onePerson->push_back(boundRect[rect]);
			}

		}
		else if (meeting == 3 || meeting == 5)
		{
			int largest = boundRect[0].height * boundRect[0].width, secondLargest = 0;
			Rect tempRect = boundRect[0];
			for (int rect = 1; rect < boundRect.size(); rect++)
			{
				int temp = boundRect[rect].height * boundRect[rect].width;

				if (largest < temp)
				{
					onePerson->push_back(tempRect);
					twoPerson->push_back(boundRect[rect]);
				}
				else
				{
					onePerson->push_back(boundRect[rect]);
					twoPerson->push_back(tempRect);
				}
			}
		}
		else if (meeting == 4)
		{
			for (int rect = 0; rect < boundRect.size(); rect++)
			{
				twoPerson->push_back(boundRect[rect]);
			}
		}
	}
}

 /*
	Increases contrast on pixels so it's nicer to look at image
 */
Mat enhanceHeat(Mat heatImg)
{
	double min, max;
	minMaxLoc(heatImg, &min, &max);

	for (int y = 0; y < heatImg.rows; y++)
	{
		for (int x = 0; x < heatImg.cols; x++)
		{

			Vec<uchar, 1> pixel = heatImg.at<Vec<uchar, 1>>(Point(x, y));

			if(pixel[0] > (min+50))
				pixel[0] += (200 - max);

			heatImg.at<Vec<uchar, 1>>(Point(x, y)) = pixel;
		}
	}
	return heatImg;
}

Mat hystThreshold(Mat image)
{
	double min, max;
	minMaxLoc(image, &min, &max);
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			Vec<uchar, 1> pixel = image.at<Vec<uchar, 1>>(Point(x, y));

			if (pixel[0] > (2 * (float)max / (float)3))
			{
				pixel[0] = 255;
			}
			else if (pixel[0] > ((float)max / (float)3))
			{
				pixel[0] = 255;
			}
			else
			{
				pixel[0] = 0;
			}

			image.at<Vec<uchar, 1>>(Point(x, y)) = pixel;
		}
	}
	return image;
}

matrixF upScale(matrixF arr)
{
	matrixF tempMatrix;
	int doubleSize = arr[0].size() * 4;

	for (int frame = 0; frame < arr.size(); frame++)
	{
		Mat tempImg = Mat::zeros(8, 8, CV_32FC1);;

		int element = 0;
		for (int y = 0; y < tempImg.rows; y++)
		{
			for (int x = 0; x < tempImg.cols; x++)
			{
				Vec<float, 1> pixel = tempImg.at<Vec<float, 1>>(Point(x, y));
				pixel[0] = arr[frame][element]*1000;

				tempImg.at<Vec<float, 1>>(Point(x, y)) = pixel;
				element++;
			}
		}

		Mat upscaledTempImg;
		pyrUp(tempImg, upscaledTempImg, Size(tempImg.cols * 2, tempImg.rows * 2));
		std::vector<float> temp(doubleSize, 0);

		temp.assign(upscaledTempImg.data, upscaledTempImg.data + upscaledTempImg.total());

		for (int i = 0; i < temp.size(); i++)
		{
			temp[i] /= 1000;
		}
		tempMatrix.push_back(temp);
	}
	return tempMatrix;
}

std::vector<float> getSuperPixelMean(matrixF normImg)
{

	std::vector<float> sum(normImg[0].size(), 0);

	for (int frame = 0; frame < normImg.size(); frame++)
	{
		Mat tempImg = Mat::zeros(8, 8, CV_32FC1);;

		int element = 0;
		for (int y = 0; y < tempImg.rows; y++)
		{
			for (int x = 0; x < tempImg.cols; x++)
			{
				Vec<float, 1> pixel = tempImg.at<Vec<float, 1>>(Point(x, y));
				pixel[0] = normImg[frame][element]*1000;

				tempImg.at<Vec<float, 1>>(Point(x, y)) = pixel;
				element++;
			}
		}
	}
	int n = sum.size();

	for (int i = 0; i < n; i++)
	{
		sum[i] = sum[i] / n;
	}
		
	return sum;
}

/*
	Calculates mean for each pixel. 
	Each pixel's value is sumed through each frame, and then divided by number of frames.
*/
std::vector<float> getPixelThresh(matrixF normImg)
{
	std::vector<float> temp (normImg[0].size(), 0);

	for (int j = 0; j < normImg[0].size(); j++)
	{
		for (int i = 0; i < normImg.size(); i++)
		{
			temp[j] += normImg[i][j];
		}
	}

	int n = normImg.size();
	for (int i = 0; i < temp.size(); i++)
	{
		temp[i] = temp[i] / n;
	}

	return temp;
}


/*
	Calculates standard deviation from each pixel mean value
*/
std::vector<float> getPixelSTD(matrixF normImg, std::vector<float> pixelAvg)
{
	std::vector<float> temp(normImg[0].size(), 0);
	
	for (int j = 0; j < normImg[0].size(); j++)
	{
		for (int i = 0; i < normImg.size(); i++)
		{
			temp[j] += (pixelAvg[j] - normImg[i][j])*(pixelAvg[j] - normImg[i][j]);
		}
	}

	int n = temp.size();
	for (int i = 0; i < n; i++)
	{
		temp[i] = sqrt(temp[i] / (n - 1));
	}

	return temp;
}

void displayOneFile()
{
	std::ifstream data("no_present_before.txt");
	std::string line, val;
	matrixI noPresentArrBefore;
	matrixF normArr;
	int lowThreshold = 0;

	while (std::getline(data, line, '\r'))
	{
		int element = 0;
		std::vector<int> parsedRow;
		std::stringstream lineStream(line);
		while (std::getline(lineStream, val, ','))
		{
			if (element != 0)
				parsedRow.push_back(std::stoi(val));

			element++;
		}
		noPresentArrBefore.push_back(parsedRow);
	}
	normArr = normalize(noPresentArrBefore);

	Mat imgV = Mat::zeros(8, 8, CV_8UC1);

	namedWindow("One file", CV_WINDOW_NORMAL);

	//lowThreshold = getLowThresh(normData[0][0]) + 70;
	int meeting = 0;

	for (int iFrame = 0; iFrame < normArr.size(); iFrame++)
	{
		std::vector<Mat> channels;
		Mat image;
		int element = 0;
		for (int y = 0; y < imgV.rows; y++)
		{
			for (int x = 0; x < imgV.cols; x++)
			{

				Vec<uchar, 1> pixelV = imgV.at<Vec<uchar, 1>>(Point(x, y));

				int temp = normArr[iFrame][element] * 255;
				if (temp < lowThreshold)
					pixelV[0] = 0;
				else
					pixelV[0] = temp;

				imgV.at<Vec<uchar, 1>>(Point(x, y)) = pixelV;

				element++;

			}
		}

		Mat upscaledImg;
		pyrUp(imgV, upscaledImg, Size(imgV.cols * 2, imgV.rows * 2));

		//cvtColor(image, image, CV_BGR2HSV);
		Mat imgSobel, abs, absMag;
		Sobel(upscaledImg, imgSobel, CV_8UC1, 1, 1, 1);
		convertScaleAbs(imgSobel, abs);
		addWeighted(abs, 0.5, abs, 0.5, 0, absMag);

		imshow("One file", upscaledImg);
		waitKey(0);
	}
}

void imshowHSV(cv::Mat& image)
{
	cv::Mat hsv;
	cv:cvtColor(image, hsv, CV_HSV2BGR);
	cv::imshow("Window", hsv);
}

void printArr(matrixF arr)
{
	for (auto& row : arr) {					   /* iterate over rows */
		for (auto& value : row)               /* iterate over vals */
			std::cout << value << ",";       /* output value      */
		std::cout << "\n";                  /* tidy up with '\n' */
	}
}

matrixF normalize(matrixI arr)
{
	matrixF temp;
	int min = 300, max = 0;
	temp.resize(arr.size());
	for (int i = 0; i < arr.size(); i++)
	{
		temp[i].resize(arr[i].size());
		for (int j = 0; j < arr[i].size(); j++)
		{
			if (min > arr[i][j])
				min = arr[i][j];
			if (max < arr[i][j])
				max = arr[i][j];
		}
	}

	for (int i = 0; i < arr.size(); i++)
	{
		for (int j = 0; j < arr[i].size(); j++)
		{
			temp[i][j] = (float)(arr[i][j] - min) / (float)(max - min);
		}
	}

	return temp;
}

int getLowThresh(matrixF arr)
{
	int sum = 0;
	for (int i = 0; i < arr.size(); i++)
	{
		sum += arr[0][i]*255;
	}
	return sum / 64;
}

Mat loadDataToMat(matrixF dataArr, int iFrame)
{
	int element = 0;
	Mat temp = Mat::zeros(8, 8, CV_8UC1);
	for (int y = 0; y < temp.rows; y++)
	{
		for (int x = 0; x < temp.cols; x++)
		{
			Vec<uchar, 1> pixel = temp.at<Vec<uchar, 1>>(Point(x, y));
			pixel[0] = dataArr[iFrame][element] * 255;

			temp.at<Vec<uchar, 1>>(Point(x, y)) = pixel;

			element++;
		}
	}
	return temp;
}


/*
	Loads data from files, will also normalize input data. 
*/
void loadData(std::vector<matrixI>* inData, std::vector<matrixF>* normData, std::vector<std::string> fileNames)
{
	std::string line, val;
	for (int file = 0; file < fileNames.size(); file++)
	{
		std::ifstream data(fileNames[file]);
		while (std::getline(data, line, '\r'))
		{
			int element = 0;
			std::vector<int> parsedRow;
			std::stringstream lineStream(line);
			while (std::getline(lineStream, val, ','))
			{
				if (element != 0)
					parsedRow.push_back(std::stoi(val));

				element++;
			}
			(*inData)[file].push_back(parsedRow);
		}
		(*normData)[file] = normalize((*inData)[file]);
	}
}

/*
	Loads normalized data into cv::Mat.
	Also thresholds and upscales the data.

	Thresholds are calculated by getting the mean value and standard deviation
	of each pixel from one of the "no participant" meetings,
	these pixel mean values function as a filter to remove noise. 
	Each pixel of normalize raw data is then compared to the mean value and standard deviation
	of the threshold for that pixel. 
*/
void initialThresholdImage(std::vector<Mat>* heatFrames, std::vector<matrixF> normData, int meeting)
{
	Mat imgV = Mat::zeros(8, 8, CV_8UC1);
	Mat original = Mat::zeros(8, 8, CV_8UC1);
	std::vector<float> pixelThresholds = getPixelThresh(normData[0]);
	std::vector<float> pixelSTD = getPixelSTD(normData[0], pixelThresholds);

	for (int iFrame = 0; iFrame < normData[meeting].size(); iFrame++)
	{

		int element = 0;
		for (int y = 0; y < imgV.rows; y++)
		{
			for (int x = 0; x < imgV.cols; x++)
			{

				Vec<uchar, 1> pixelV = imgV.at<Vec<uchar, 1>>(Point(x, y));
				Vec<uchar, 1> pixelO = original.at<Vec<uchar, 1>>(Point(x, y));


				int temp = normData[meeting][iFrame][element] * 255;
				int pixelThreshold = (pixelThresholds[element] + pixelSTD[element] * 2) * 255;

				if (temp < pixelThreshold)
					pixelV[0] = 0;
				else
					pixelV[0] = temp;

				pixelO[0] = temp;

				imgV.at<Vec<uchar, 1>>(Point(x, y)) = pixelV;
				original.at<Vec<uchar, 1>>(Point(x, y)) = pixelO;

				element++;
			}
		}

		// Scale up the image using image pyramids
		Mat upscaleTemp, upscaledImg;
		pyrUp(imgV, upscaleTemp, Size(imgV.cols * 2, imgV.rows * 2));
		pyrUp(upscaleTemp, upscaledImg, Size(upscaleTemp.cols * 2, upscaleTemp.rows * 2));

		// Store all thresholded and upscaled frames here
		(*heatFrames).push_back(upscaledImg);
	}

}
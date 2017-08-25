#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <windows.h>
#include "HogParam.h"
#include "util.h"

using namespace std;
using namespace cv;

#define HIST_SIZE	80
#define HIST_SEG	16
#define ID	7

typedef Ptr<ml::SVM> MySVM;

void getHogFeature(Mat& image, HogParam& param, vector<float>& hogFeat, bool cvt);

void getMSHogFeature(Mat& image, vector<HogParam>& params, vector<float>& msHogFeat);

void getHistogramFeature(Mat& image, float*& feat);

void trainHog(vector<string>& file_names, vector<int>& labels, vector<HogParam>& params, int iter);

int detectHog(Mat& image, MySVM& svm, vector<HogParam>& params);

void trainHist(vector<string>& file_names, vector<int>& labels, int iter);

int detectHist(Mat& image, MySVM& svm);

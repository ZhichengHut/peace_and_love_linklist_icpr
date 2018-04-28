#ifndef TLBO_H
#define TLBO_H

#include <iostream>
#include <string>  
#include <vector>
#include <opencv.hpp>
#include <opencv2/opencv.hpp>

#include <time.h>

using namespace std;
using namespace cv;

void getMask(vector<Mat> &imgList, vector<int> &labelList, int pos_num, int neg_num, int pop_num, int iteration_num, Mat &mask_result, float &threshold);

#endif//TLBO_H
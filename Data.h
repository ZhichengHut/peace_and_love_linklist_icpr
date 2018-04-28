#ifndef DATA_H
#define DATA_H

#include<vector>
#include <opencv.hpp>

using namespace cv;

class Data{
private:
	Mat img;
	int label;

public:
	Data(Mat I, int L);
	Mat get_Img();
	int get_Lab();
};

#endif//DATA_H
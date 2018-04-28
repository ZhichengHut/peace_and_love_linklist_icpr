#ifndef EXTRACTDATA_H
#define EXTRACTDATA_H

#include <iostream>
#include <fstream>  
#include <sstream>  
#include <string>  
#include <vector>
#include <opencv.hpp>
#include <opencv2/opencv.hpp>

#include <typeinfo>
#include <io.h>
#include <direct.h>
#include <windows.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

using namespace std;
using namespace cv;

vector<int> readCSV(string csvFile);
Mat preProcess(Mat img, float thresh);
vector<Point2i> getCenter(Mat img, int R);
void clearFold(string out_fold);

void saveTrainData(string img_file, string csv_file, string out_fold, float thresh, int width, int R, int rand_num);
void saveTrainData(string img_file, string out_fold, float thresh, int width, int R, int rand_num);
void getTrainingSet(string train_fold, string out_fold, float thresh, int width, int R, int rand_num);

//for the second filter
void saveTrainData(string img_file, string csv_file, string out_fold, float thresh, int width, int R);
void saveTrainData(string img_file, string out_fold, float thresh, int width, int R);
void getTrainingSet(string train_fold, string out_fold, float thresh, int width, int R);
void extractData(string train_fold, string out_fold, float train_thresh, bool get_train, int width, int R);

void getTestImg(string curDir, string img_name, float thresh, int width, int R, int rand_num);
void getTestingSet(string test_fold, float thresh, int width, int R, int rand_num);
void extractData(string train_fold, string test_fold, string out_fold, float train_thresh, float test_thresh, bool get_train, bool get_test, int width, int R, int rand_num);



#endif//EXTRACTDATA_H
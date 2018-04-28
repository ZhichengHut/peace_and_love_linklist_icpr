#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "Node.h"

class RandomForest{
private:
	int window_width;
	int tree_num;
	int sample_num;
	int maxDepth;
	int minLeafSample;
	float minInfoGain;

	vector<Mat> imgData;
	vector<int> LabelData;

	Node **root_list;

public:
	RandomForest(vector<Mat> &img = vector<Mat>(), vector<int> &label = vector<int>(), int w_w=0, int t_n=0, int s_n=0, int maxD=0, int minL=0, float minInfo=0);
	~RandomForest();

	void train();
	void save(ofstream &fout);
	void load(ifstream &fin);
	
	float predict(Mat test_img);
	vector<float> predict(vector<Mat> &test_img);
};

#endif//RANDOMFOREST_H
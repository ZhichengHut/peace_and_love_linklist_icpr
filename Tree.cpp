#include "Tree.h"

Tree::Tree(vector<Mat> &SP, vector<int> &LB, int w_w, int maxD, int minL, float minInfo){
	window_width = w_w;

	//sample.assign(SP.begin(), SP.end());
	//label.assign(LB.begin(), LB.end());
	sample = SP;
	label = LB;
	num_1 = accumulate(label.begin(), label.end(),0);
	num_0 = label.size() - num_1;


	maxDepth = maxD;
	minLeafSample = minL;
	minInfoGain = minInfo;
	root = new Node();
}
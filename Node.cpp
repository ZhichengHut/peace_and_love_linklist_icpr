#include "Node.h"

Node::Node(vector<Mat> &sample, vector<int> &label, int curr_depth, int w_w, int maxD, int minL, float minInfo){
	//data
	imgList = sample;
	imgLabel = label;

	//parameter of model
	maxDepth = maxD;
	minLeafSample = minL;
	minInfoGain = minInfo;

	x1 = 0;
	x2 = 0;
	y1 = 0;
	y2 = 0;
	theta = 0;
	d = w_w;
	voting = 2;

	//status of node
	LeafFlag = false;
	sample_num = imgList.size();
	positive_num = 1.0 * accumulate(imgLabel.begin(), imgLabel.end(),0);
	current_depth = curr_depth;
	infoGain = 0;
	Entro = calculate_entropy(sample_num, positive_num);

	leftchild = NULL;
	rightchild = NULL;
	//cout << "Entro = " << Entro << endl;
}

Node::~Node(){
	imgList.clear();
	imgLabel.clear();
	vector<Mat>().swap(imgList);
	vector<int>().swap(imgLabel);

	if(!LeafFlag){
		delete leftchild;
		leftchild = NULL;
		delete rightchild;
		rightchild = NULL;
	}
}

void Node::setLeaf(){
	//set the flag
	LeafFlag = true;
	
	//calculate the vote
	int p_1 = 1.0 * accumulate(imgLabel.begin(), imgLabel.end(),0);
	int p_2 = imgLabel.size() - p_1;

	if(p_2 > p_1)
		voting = 0;
	else
		voting = 1;

	//release the space
	imgList.clear();
	imgLabel.clear();
	vector<Mat>().swap(imgList);
	vector<int>().swap(imgLabel);

	//cout << "This node is setted to be leaf, depth = " << current_depth << endl;
	//cin.get();
}

float Node::calculate_entropy(int sample_num, int positive_num){
	float entropy = 0;

	if(sample_num != 0){
		float pp = 1.0 * positive_num / sample_num;						//positive%
		float np = 1.0 * (sample_num - positive_num) / sample_num;		//negtive%

		if(pp!=0 && np !=0)
			entropy = -1.0*pp*log(1.0*pp)/log(2.0) - 1.0*np*log(1.0*np)/log(2.0);
	}
	return entropy;
}

void Node::split_Node(){
	/*if(Entro==0 || current_depth>maxDepth || sample_num<minLeafSample){
		setLeaf();
		return;
	}*/

	if(Entro==0 || sample_num<minLeafSample){
		setLeaf();
		return;
	}

	if(current_depth>maxDepth){
		cout << "insufficient depth" << endl;
		setLeaf();
		return;
	}

	srand (time(NULL));

	int r = imgList[0].rows;
	int c = imgList[0].cols;

	for(int i=0; i<100; i++){
		//randomly choose the parameter
		int d_tmp = rand() % (min(imgList[0].cols, imgList[0].rows)) + 1;
		int x1_tmp = rand() % (c-d_tmp+1);
		int y1_tmp = rand() % (r-d_tmp+1);
		int x2_tmp = rand() % (c-d_tmp+1);
		int y2_tmp = rand() % (r-d_tmp+1);

		//randomly choose one positive and one negative sample to calculate the theta
		int ss_index1 = rand() % imgLabel.size();
		//float theta_tmp = mean(imgList[ss_index1](Rect(x1_tmp,y1_tmp,d_tmp,d_tmp)))[0] - mean(imgList[ss_index1](Rect(x2_tmp,y2_tmp,d_tmp,d_tmp)))[0];
		//int theta_tmp = get_Sum(imgList[ss_index1], x1_tmp, y1_tmp, d_tmp) - get_Sum(imgList[ss_index1], x2_tmp, y2_tmp, d_tmp);
		int theta_tmp = get_Sum(imgList[ss_index1], x1_tmp, y1_tmp, d_tmp);
		while(true){
			int ss_index2 = rand() % imgLabel.size();
			if(imgLabel[ss_index1] + imgLabel[ss_index2] == 1){
				//theta_tmp += (mean(imgList[ss_index2](Rect(x1_tmp,y1_tmp,d_tmp,d_tmp)))[0] - mean(imgList[ss_index2](Rect(x2_tmp,y2_tmp,d_tmp,d_tmp)))[0]);
				//theta_tmp += (get_Sum(imgList[ss_index2], x1_tmp, y1_tmp, d_tmp) - get_Sum(imgList[ss_index2], x2_tmp, y2_tmp, d_tmp));
				theta_tmp += get_Sum(imgList[ss_index2], x1_tmp, y1_tmp, d_tmp);
				theta_tmp /= 2;
				break;
			}
		}

		//split the node under this set of parameter
		int left_num = 0;
		int left_positive = 0;
		int right_num = 0;
		int right_positive = 0;

		for(int p=0; p<sample_num; p++){
			//cout << "img size " << imgList[p].cols <<" " << imgList[p].rows << endl;

			//float mean1 = mean(imgList[p](Rect(x1_tmp,y1_tmp,d_tmp,d_tmp)))[0];
			int sum1 = get_Sum(imgList[p], x1_tmp,y1_tmp,d_tmp);
			//cout << 11 << endl;
			//float mean2 = mean(imgList[p](Rect(x2_tmp,y2_tmp,d_tmp,d_tmp)))[0];
			//int sum2 = get_Sum(imgList[p], x2_tmp,y2_tmp,d_tmp);
			int sum2 = 0;

			//if(mean1-mean2>theta_tmp){
			if(sum1-sum2>theta_tmp){
				//leftImg_tmp.push_back(imgList[p]);
				//leftLabel_tmp.push_back(imgLabel[p]);
				left_num++;
				left_positive += imgLabel[p];
			}
			else{
				//rightImg_tmp.push_back(imgList[p]);
				//rightLabel_tmp.push_back(imgLabel[p]);
				right_num++;
				right_positive += imgLabel[p];
			}
		}

		//calculate the current information gain
		float infoGain_new = Entro - (left_num*calculate_entropy(left_num, left_positive) + right_num*calculate_entropy(right_num, right_positive))/sample_num;
		//cout << "new gain = " << infoGain_new << endl;
		//cin.get();
		
		if(infoGain_new > infoGain){	
			//cout << "tmp : " << x1_tmp << " " << y1_tmp << " "  << x2_tmp << " "  << y2_tmp << " "  << " " << d << endl;
			infoGain = infoGain_new;
			x1 = x1_tmp;
			x2 = x2_tmp;
			y1 = y1_tmp;
			y2 = y2_tmp;
			d = d_tmp;
			theta = theta_tmp;
			//cout << "new gain = " << infoGain << endl;
		}
	}

	if(infoGain<minInfoGain){
		setLeaf();
		return;
	}

	vector<Mat> left_img;
	vector<int> left_label;
	vector<Mat> right_img;
	vector<int> right_label;

	for(int p=0; p<sample_num; p++){
		//cout << "location: " << x1 << " " << y1 << " " << x2 << " " << y2 << " " << d << endl;
		//float mean1 = mean(imgList[p](Rect(x1,y1,d,d)))[0];
		//float mean2 = mean(imgList[p](Rect(x2,y2,d,d)))[0];

		int sum1 = get_Sum(imgList[p], x1,y1,d);
		//int sum2 = get_Sum(imgList[p], x2,y2,d);
		int sum2 = 0;

		if(sum1-sum2>theta){
			left_img.push_back(imgList[p]);
			left_label.push_back(imgLabel[p]);
		}
		else{
			right_img.push_back(imgList[p]);
			right_label.push_back(imgLabel[p]);
		}
	}

	//create the left and right child node
	leftchild = new Node(left_img, left_label, current_depth+1, d, maxDepth, minLeafSample, minInfoGain);
	rightchild = new Node(right_img, right_label, current_depth+1, d, maxDepth, minLeafSample, minInfoGain);

	//recycle space
	imgList.clear();
	vector<Mat>().swap(imgList);
	imgLabel.clear();
	vector<int>().swap(imgLabel);
	left_img.clear();
	vector<Mat>().swap(left_img);
	left_label.clear();
	vector<int>().swap(left_label);
	right_img.clear();
	vector<Mat>().swap(right_img);
	right_label.clear();
	vector<int>().swap(right_label);

	//split the child node recursively
	leftchild->split_Node();
	rightchild->split_Node();
}

void Node::save(ofstream &fout){
	fout << x1 << " " << x2 << " " << y1 << " " << y2 << " " << d << " " << theta << " " << voting << " " << LeafFlag << endl;
	if(!LeafFlag){
		leftchild->save(fout);
		rightchild->save(fout);
	}
	return;
}

void Node::load(ifstream &fin){
	fin >> x1 >> x2 >> y1 >> y2 >> d >> theta >> voting >> LeafFlag;
	//cout << "x1:" << x1 << " x2:" << x2 << " y1:" << y1 << " y2:" << y2 << " d:" << d << " theta:" << theta << " voting:" << voting << " LeafFlag:" << LeafFlag << endl;
	//cin.get();
	
	if(!LeafFlag){
		leftchild = new Node();
		leftchild->load(fin);
		rightchild = new Node();
		rightchild->load(fin);
	}
}

int Node::predict(Mat &test_img){
	while(!LeafFlag){
		//float mean1 = mean(test_img(Rect(x1,y1,d,d)))[0];
		//float mean2 = mean(test_img(Rect(x2,y2,d,d)))[0];
		//if(mean1-mean2>theta)
		//cout << "x1 = " << x1 << " y1 = " << y1 << " x2 = " << x2 << " y2 = " << y2 << " d = " << d << " col = " << test_img.cols << " row = " << test_img.rows << endl;
		//cin.get();
		int sum1 = get_Sum(test_img, x1,y1,d);
		//int sum2 = get_Sum(test_img, x2,y2,d);
		int sum2 = 0;

		if(sum1-sum2>theta)
			return leftchild->predict(test_img);
		else
			return rightchild->predict(test_img);
	}

	if(voting != 0 && voting != 1){
		cout << "vote error" << endl;
		cin.get();
	}
	else{
		//cout << "one prediction ended" << endl;
		return voting;
	}
}
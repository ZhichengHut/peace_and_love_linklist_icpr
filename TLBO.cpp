#include "TLBO.h"

void getMask(vector<Mat> &imgList, vector<int> &labelList, int pos_num, int neg_num, int pop_num, int iteration_num, Mat &mask_result, float &threshold){
	int sample_num = pos_num + neg_num;

	srand((unsigned)time(NULL));

	Mat his_pos = Mat::zeros(256, pos_num, CV_32FC1);
	Mat his_neg = Mat::zeros(256, neg_num, CV_32FC1);

	int histSize = 256;
	float range[] = {0, 256} ;
	const float* histRange = {range};
	
	bool uniform = true; 
	bool accumulate = false;
	
	Mat hist;
	Mat his_scale = Mat::zeros(256,1,CV_32SC1);
	for(int i=0; i<256; i++)
		his_scale.at<int>(i,0) = i;

	int curr_pos = 0;
	int curr_neg = 0;

	for(int i=0; i<sample_num; i++){
		calcHist(&imgList[i], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
		hist.convertTo(hist,his_scale.type());
		hist = hist.mul(his_scale);
		if(labelList[i]){
			hist.copyTo(his_pos.col(curr_pos++));
		}
		else{
			hist.copyTo(his_neg.col(curr_neg++));
		}
	}

	RNG rnger(getTickCount());
    int width = 256, height = pop_num;
    Mat mask;
    //cv::Scalar mm, ss;
    // CV_8UC3 uniform distribution
    mask.create(height, width, CV_32FC1);
    rnger.fill(mask, RNG::UNIFORM, Scalar::all(0.0), Scalar::all(1.0));

	Mat mask_tmp = Mat::zeros(height, width, CV_32FC1);
	Mat mean_D = Mat::zeros(1,256, CV_32FC1);
	his_pos.convertTo(his_pos,mask.type());
	his_neg.convertTo(his_neg,mask.type());

	//////////////////  initialization  //////////////////
	Mat reject_num =  Mat::zeros(1,pop_num, CV_32SC1);
	double max, min;
	Point min_loc, max_loc;
	
	Mat pos_student = mask*his_pos;
	Mat neg_student = mask*his_neg;
	
	for(int i=0; i<pop_num; i++){
		minMaxLoc(pos_student.row(i), &min, &max, &min_loc, &max_loc);
		reject_num.at<int>(0,i) = sum(neg_student.row(i)>max)[0]/255;
	}
	minMaxLoc(reject_num, &min, &max, &min_loc, &max_loc);
	cout << "first max rejection num = " << max << endl;
	cout << "original mask = " << endl;
	cout << mask.row(max_loc.y) << endl;

	for(int i=0; i<iteration_num; i++){
		//////////////////  teaching phase  //////////////////
		minMaxLoc(reject_num.row(0), &min, &max, &min_loc, &max_loc);

		Mat mask_teacher = mask.row(max_loc.y);

		for(int j=0; j<256; j++)
			mean_D.at<float>(0,j) = mean(mask.col(j))[0];

		Mat Diff_d = (rand()/(RAND_MAX+0.0))*(mask_teacher - ((rand()%2)+1)*mean_D);
		
		for(int j=0; j<pop_num; j++){
			Mat X_new_tmp = mask.row(j)+Diff_d;
			normalize(X_new_tmp, X_new_tmp, 1.0, 0.0, CV_MINMAX);
			X_new_tmp.copyTo(mask_tmp.row(j));
		}

		//normalize(mask_tmp, mask_tmp, 1.0, 0.0, CV_MINMAX);

		pos_student = mask_tmp*his_pos;
		neg_student = mask_tmp*his_neg;

		int re_num = 0;

		for(int j=0; j<pop_num; j++){
			minMaxLoc(pos_student.row(j), &min, &max, &min_loc, &max_loc);
			//if(sum(neg_student.row(j)>max)[0]/255 > reject_num.at<int>(0,j) && sum(mask_tmp.row(j)>=0)[0]/255 == mask_tmp.cols){
			if(sum(neg_student.row(j)>max)[0]/255 > reject_num.at<int>(0,j)){
				re_num++;
				mask_tmp.row(j).copyTo(mask.row(j));
				reject_num.at<int>(0,j) = sum(neg_student.row(j)>max)[0]/255;
			}
		}

		//cout << "teaching phase, improve num = " << re_num << endl;
		//minMaxLoc(reject_num, &min, &max, &min_loc, &max_loc);
		//cout << "max rejection num = " << max << endl;

		//////////////////  learning phase  //////////////////
		for(int j=0; j<pop_num; j++){
			int k = rand()%pop_num;
			while(k==j)
				k = rand()%pop_num;

			if(reject_num.at<int>(0,j)>reject_num.at<int>(0,k)){
				Mat X_new_tmp = mask.row(j)+(rand()/(RAND_MAX+0.0))*(mask.row(j)-mask.row(k));
				normalize(X_new_tmp, X_new_tmp, 1.0, 0.0, CV_MINMAX);
				X_new_tmp.copyTo(mask_tmp.row(j));
			}
			else{
				Mat X_new_tmp = mask.row(j)+(rand()/(RAND_MAX+0.0))*(mask.row(k)-mask.row(j));
				normalize(X_new_tmp, X_new_tmp, 1.0, 0.0, CV_MINMAX);
				X_new_tmp.copyTo(mask_tmp.row(j));
			}
		}

		//normalize(mask_tmp, mask_tmp, 1.0, 0.0, CV_MINMAX);
		
		pos_student = mask_tmp*his_pos;
		neg_student = mask_tmp*his_neg;
		
		re_num = 0;
		
		for(int j=0; j<pop_num; j++){
			minMaxLoc(pos_student.row(j), &min, &max, &min_loc, &max_loc);
			//if(sum(neg_student.row(j)>max)[0]/255 > reject_num.at<int>(0,j) && sum(mask_tmp.row(j)>=0)[0]/255 == mask_tmp.cols){
			if(sum(neg_student.row(j)>max)[0]/255 > reject_num.at<int>(0,j)){
				re_num++;
				mask_tmp.row(j).copyTo(mask.row(j));
				reject_num.at<int>(0,j) = sum(neg_student.row(j)>max)[0]/255;
			}
		}
		//cout << "learning phase, improve num = " << re_num << endl;
		//minMaxLoc(reject_num, &min, &max, &min_loc, &max_loc);
		//cout << "max rejection num = " << max << endl;
		//cin.get();
	}

	minMaxLoc(reject_num, &min, &max, &min_loc, &max_loc);
	cout << "remove num = " << max << endl;
	mask.row(max_loc.y).copyTo(mask_result);

	minMaxLoc(pos_student.row(max_loc.y), &min, &max, &min_loc, &max_loc);
	cout << "threshold = " << max << endl;
	threshold = max;

	return;
}
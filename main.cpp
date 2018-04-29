#include "ExtractData.h"
#include "ReadData.h"
#include "Node.h"
#include "Data.h"
#include "RandomForest.h"
#include "Evaluate.h"
#include "TLBO.h"

#include <time.h>
#include <iomanip>  

int main(){
	string train_fold = "C:/45 Thesis/data_icpr/train_A";
	string test_fold = "C:/45 Thesis/data_icpr/test_A";
	string out_fold = "C:/45 Thesis/data_icpr/train_A/extracted/";
	string model_address = "E:/45 Thesis/result_icpr/model.txt";
	string mask_address = "E:/45 Thesis/result_icpr/mask.txt";

	bool generate_mask_flag;
	Mat mask_result = Mat::zeros(1,256,CV_32FC1);
	float mask_threshold = 0;

	bool load_model_flag;
	cout << "Do you want to load currently existed model? (0/1)" << endl;
	cin >> load_model_flag;

	if(load_model_flag){
		ifstream fin_model(model_address);
		ifstream fin_mask(mask_address);

		if(!fin_model.is_open()){
			cout << "No model exists" << endl;
			return 1;
		}
		else if(!fin_mask.is_open()){
			cout << "No mask exists" << endl;
			return 1;
		}

		cout << "*****************Start to load the mask*****************" << endl;
		fin_mask >> mask_threshold;
		for(int k=0; k<256; k++)
			fin_mask >> mask_result.at<float>(0,k);

		fin_mask.close();
		cout << "*****************Loading completed*****************" << endl << endl;

		RandomForest *RF = new RandomForest();
		RF->load(fin_model);
		fin_model.close();

		cout << "*****************Start to evaluate the performance*****************" << endl;
		double start,end;
		start=clock();
		//get_predict_result(RF, test_fold);
		//int sample_interval = 5;
		//float prob_threshold = 0.4;
		//get_predict_result(RF, test_fold, patch_width, sample_interval, prob_threshold);
		get_predict_result(RF, test_fold, mask_result, mask_threshold);
		end=clock();
		double test_t = (end - start) / CLOCKS_PER_SEC ;
		cout << "*****************Evaluation completed*****************" << endl << endl;
		
		cout << "*****************Start to calculate F1 score*****************" << endl;
		float F1_score = get_F1_score(test_fold);
		cout << "*****************Calculation completed*****************" << endl << endl;
		
		ofstream fin("E:/45 Thesis/result/result.csv",ios::app);
		if(!fin){
			cout << "open file error" <<endl; 
			cin.get();
			return 0;
		}
		
		fin << ",test time," << test_t << endl;
		fin.close();
		
		delete RF;
		RF = NULL;
	}

	//cin.get();

	if(!load_model_flag){
		cout << "*****************Start to extract sub-image*****************" << endl;
		float train_thresh = 0.25;
		float test_thresh = 0.15;

		bool get_train = false;
		bool get_test = false;

		int patch_width = 35;
		int core_R = 4;
		int ran_point = 20;

		extractData(train_fold, test_fold, out_fold, train_thresh, test_thresh, get_train, get_test, patch_width, core_R, ran_point);
		cout << "*****************Extraction completed*****************" << endl << endl;

		cout << "*****************Start to read training data*****************" << endl;
		vector<Mat> imgTrain;
		//vector<Mat> integral_img_list;
		vector<int> labelTrain;
		int pos_num = 0;
		int neg_num = 0;

		readTrainData(out_fold, imgTrain, labelTrain, pos_num, neg_num);

		cout << "Sample number = " << pos_num+neg_num << endl;
		cout << "Positive sampe number = " << pos_num << endl;
		cout << "*****************Reading completed*****************" << endl << endl;

		cout << "Do you want to generate a mask? (0/1)" << endl;
		//cin >> generate_mask_flag;
		generate_mask_flag = false;

		if(generate_mask_flag){
			cout << "*****************Start to generate mask*****************" << endl;
			int pop_num = 50;
			int iteration_num = 200;
			
			getMask(imgTrain, labelTrain, pos_num, neg_num, pop_num, iteration_num, mask_result, mask_threshold);

			ofstream fout_mask(mask_address);
			fout_mask << mask_threshold << endl;
			for(int k=0; k<256; k++)
				fout_mask << setprecision(8) << mask_result.at<float>(0,k) << endl;
			fout_mask.close();
			cout << "*****************Generation ended*****************" << endl << endl;
		}
		else{
			ifstream fin_mask(mask_address);
			fin_mask >> mask_threshold;
			for(int k=0; k<256; k++)
				fin_mask >> mask_result.at<float>(0,k);
		}

		double start,end;

		for(int i=1; i<=20; i+=2){
			int window_width = 10;

			//int tree_num = 3;
			int tree_num = 30;
			int sample_num = 10000;
			int maxDepth = 50;
			//int minLeafSample = 10;
			int minLeafSample = i;
			float minInfo = 0;

			cout << "*****************Start to train the model*****************" << endl;
			start=clock();
			RandomForest *RF = new RandomForest(imgTrain, labelTrain, window_width, tree_num, sample_num, maxDepth, minLeafSample, minInfo);
			RF->train();
			end = clock();
			double train_t = (end - start) / CLOCKS_PER_SEC ;
			cout << "*****************Training completed*****************" << endl << endl;

			cout << "Do you want to save the model? (0/1)" << endl;
			bool save_flag;
			//cin >> save_flag;
			save_flag = false;
			if(save_flag){
				ofstream fout_model(model_address);
				RF->save(fout_model);
				fout_model.close();
			}


			cout << "*****************Start to evaluate the performance*****************" << endl;
			start=clock();
			//get_predict_result(RF, test_fold);
			int sample_interval = 8;
			float prob_threshold = 0.5;
			get_predict_result(RF, test_fold, patch_width, sample_interval, prob_threshold);
			//get_predict_result(RF, test_fold, mask_result, mask_threshold);
			end=clock();
			double test_t = (end - start) / CLOCKS_PER_SEC ;
			cout << "*****************Evaluation completed*****************" << endl << endl;

			cout << "*****************Start to calculate F1 score*****************" << endl;
			float F1_score = get_F1_score(test_fold);
			cout << "*****************Calculation completed*****************" << endl << endl;

			ofstream fin("E:/45 Thesis/result_icpr/result.csv",ios::app);
			if(!fin){
				cout << "open file error" <<endl; 
				cin.get();
				return 0;
			}

			fin <<",tree num," <<  tree_num << ",sumple num," << sample_num << ",maxDepth," << maxDepth << ",minLeafSample," << minLeafSample << ",minInfo," << minInfo <<",train time," << train_t << ",test time," << test_t <<",window width," << window_width << endl;;
			fin.close();

			delete RF;
			RF = NULL;
		}
		imgTrain.clear();
		vector<Mat>().swap(imgTrain);
		labelTrain.clear();
		vector<int>().swap(labelTrain);
	}

	cout << "*****************Benchmark completed*****************" << endl;
	cin.get();
}
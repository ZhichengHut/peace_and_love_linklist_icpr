#include "Evaluate.h"


void get_predict_result(RandomForest *RF, string test_fold, Mat &mask, float threshold){

	cout << 111 << endl;
    char curDir[100];
	
	sprintf(curDir, "%s", test_fold.c_str());

	DIR* pDIR;
	struct dirent *entry;
	struct stat s;
	
	stat(curDir,&s);
	
	// if path is a directory
	if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
		if(pDIR=opendir(curDir)){
			//for all entries in directory
			while(entry = readdir(pDIR)){
				//cout << entry->d_name << endl;
				stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
				if (((s.st_mode & S_IFMT ) != S_IFDIR ) && ((s.st_mode & S_IFMT) == S_IFREG )){
					if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "bmp"){
						//find the corresponding fold according to the image name
						string sub_curDIR = string(curDir) + "/" + string(entry->d_name).substr(0,6);
						cout << sub_curDIR << endl;

						vector<Mat> imgTest;
						vector<int> X,Y;
						
						DIR* subDIR;
						struct dirent *sub_entry;
						struct stat sub_s;
						
						stat(sub_curDIR.c_str(), &sub_s);
						
						/////////extract the test image and their position
						if((sub_s.st_mode & S_IFMT) == S_IFDIR ){
							if(subDIR=opendir(sub_curDIR.c_str())){
								while(sub_entry = readdir(subDIR)){
									stat((sub_curDIR + string("/") + string(sub_entry->d_name)).c_str(),&sub_s);
									if (((sub_s.st_mode & S_IFMT ) != S_IFDIR ) && ((sub_s.st_mode & S_IFMT) == S_IFREG )){
										if(string(sub_entry->d_name).substr(string(sub_entry->d_name).find_last_of('.') + 1) == "png"){
											Mat img_tmp = imread(sub_curDIR + string("/") + string(sub_entry->d_name), 0);
											if(TLBO_test(img_tmp, mask, threshold)){
												integral(img_tmp, img_tmp);	
												int x = atoi(string(sub_entry->d_name).substr(0,4).c_str());
												int y = atoi(string(sub_entry->d_name).substr(5,4).c_str());
												imgTest.push_back(img_tmp);
												X.push_back(x);
												Y.push_back(y);
											}
										}
									}
								}
							}
						}
						vector<float> result = RF->predict(imgTest);
						
						imgTest.clear();
						vector<Mat>().swap(imgTest);
						
						string csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,6) + "_predict.csv";
						ofstream fout(csv_name);
						for(int i=0; i<result.size(); i++){
							if(result[i] >= 0.6)
								fout << Y[i] << "," << X[i] << endl;
						}
						
						result.clear();
						vector<float>().swap(result);
						X.clear();
						vector<int>().swap(X);
						Y.clear();
						vector<int>().swap(Y);
						
						fout.close();
					}
				}
			}
		}
	}
}

void get_predict_result(RandomForest *RF, string test_fold, int width, int sample_interval, float prob_threshold){
    char curDir[100];

	sprintf(curDir, "%s", test_fold.c_str());
	cout << curDir << endl;
	
	DIR* pDIR;
	struct dirent *entry;
	struct stat s;
	
	stat(curDir,&s);
	
	// if path is a directory
	if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
		if(pDIR=opendir(curDir)){
			//for all entries in directory
			while(entry = readdir(pDIR)){
				stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
				if (((s.st_mode & S_IFMT ) != S_IFDIR ) && ((s.st_mode & S_IFMT) == S_IFREG )){
					if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "bmp"){
						//find the corresponding fold according to the image name
						string cur_img = string(curDir) + "/" + string(entry->d_name);
						cout << "current img: " << cur_img << endl;
						
						Mat imgTest = imread(cur_img,0);
						
						vector<float> result;
						Mat test_tmp;
						
						for(int x=0; x<imgTest.cols-2*width; x+=sample_interval){
							for(int y=0; y<imgTest.rows-2*width; y+=sample_interval){
								integral(imgTest(Rect(x,y,2*width,2*width)), test_tmp);		
								result.push_back(RF->predict(test_tmp));
							}
						}
						
						int m = sqrt(result.size()*1.0);
						//cout << "test size: " << result.size() << endl;
						//cout << "m: " << m << endl;
						
						Mat heat_map_tmp = Mat::zeros(m,m,CV_32FC1);
						for(int j=0; j<m; j++){
							for(int i=0; i<m; i++){
								heat_map_tmp.at<float>(i,j) = result[j*m+i];
							}
						}

						//imshow("heat_map_orginal", heat_map_tmp);
						
						Mat heat_map_tmp2 = Mat::zeros(imgTest.cols-2*width,imgTest.rows-2*width,CV_32FC1);
						resize(heat_map_tmp,heat_map_tmp2,Size(imgTest.cols-2*width,imgTest.rows-2*width),0,0,INTER_LINEAR);
						
						Mat heat_map = Mat::zeros(imgTest.cols,imgTest.rows,CV_32FC1);
						heat_map_tmp2.copyTo(heat_map(Rect(width,width,heat_map_tmp2.cols,heat_map_tmp2.rows)));

						Mat kernel = Mat::ones(3,3,CV_32FC1);
						int iteration = 5;
						
						for(float p_t=0.3; p_t<1; p_t+=0.05){
							Mat hh_tmp;
							threshold(heat_map, hh_tmp, p_t,255 ,THRESH_BINARY);

							/// Apply the open operation
							morphologyEx(heat_map,heat_map, MORPH_OPEN, kernel, Point(-1,-1), iteration);
							
							//Mat kernel = Mat::ones(3,3,CV_32FC1);
							//int iteration = 3;
							
							/// Apply the close operation
							//morphologyEx(heat_map,heat_map, MORPH_CLOSE, kernel, Point(-1,-1), iteration);
							
							imwrite(curDir + string("/heat.png"),hh_tmp);
							//imshow("heat map", heat_map);
							//waitKey(0);
							hh_tmp = imread(curDir + string("/heat.png"),0);
							remove((curDir + string("/heat.png")).c_str());
							
							for(int R = 1; R<20; R++){
								vector<Point2i> center = getCenter(hh_tmp, R);
								int cell_num = center.size();
								cout << "mitosis number: " << cell_num << endl;
								
								stringstream stream_pt, stream_R;  
								stream_pt << p_t;
								stream_R << R;

								string csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,6) + "_predict_" + stream_pt.str() + "_" + stream_R.str() + ".csv";
								ofstream fout(csv_name);
								for(int i=0; i< cell_num; i++)		
									fout << center[i].y << "," <<  center[i].x << endl;
								
								center.clear();
								vector<Point2i>().swap(center);
								result.clear();
								vector<float>().swap(result);
								fout.close();

							}
						}
					}
				}
			}
		}
	}
}

float get_F1_score(string test_fold){
	for(float p_t=0.3; p_t<1; p_t+=0.05){
		for(int R = 1; R<20; R++){
			int TP = 0;
			int FP = 0;
			int FN = 0;
			
			float F1_score = 0.0;

			char curDir[100];
			
			sprintf(curDir, "%s", test_fold.c_str());
			
			DIR* pDIR;
			struct dirent *entry;
			struct stat s;
			
			stat(curDir,&s);

			// if path is a directory
			if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
				if(pDIR=opendir(curDir)){
					//for all entries in directory
					while(entry = readdir(pDIR)){
						stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
						if (((s.st_mode & S_IFMT ) != S_IFDIR ) && ((s.st_mode & S_IFMT) == S_IFREG )){
							if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "bmp"){
								vector<int> ground_truth;
								vector<int> prediction;

								string csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,6) + ".csv";
								ifstream fin1(csv_name);
								if(fin1){
									//cout << csv_name << endl;
									ground_truth = readCSV(csv_name);
								}
								fin1.close();
								
								stringstream stream_pt, stream_R;  
								stream_pt << p_t;
								stream_R << R;

								csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,6) + "_predict_" + stream_pt.str() + "_" + stream_R.str() + ".csv";
								ifstream fin2(csv_name);
								if(fin2){
									cout << csv_name << endl;
									prediction = readCSV(csv_name);
								}
								fin2.close();

								int num1 = ground_truth.size() / 2;
								int num2 = prediction.size() / 2;
								FN += num1;
								FP += num2;

									/*for(int i=0; i<ground_truth.size(); i+=2){
										for(int j=0; j<prediction.size(); j+=2){
											float distance = sqrt(pow(ground_truth[i]-prediction[j],2.0)+pow(ground_truth[i+1]-prediction[j+1],2.0));
											if(distance <= 30){
												TP++;
												FP--;
												FN--;
											}
										}
									}*/


								for(vector<int>::iterator i=ground_truth.begin(); i!=ground_truth.end();){
									bool detected_flag = false;
									for(vector<int>::iterator j=prediction.begin(); j!=prediction.end();){
										 float distance = sqrt(pow(*i-*j,2.0)+pow(*(i+1)-*(j+1),2.0));
										 if(distance <= 30){
											TP++;
											FP--;
											FN--;
											i = ground_truth.erase(i);
											i = ground_truth.erase(i);
											j = prediction.erase(j);
											j = prediction.erase(j);
											detected_flag = true;
											break;
										 }
										 else
											 j+=2;
									}
									if(!detected_flag)
										i+=2;
								}
							}
						}
					}
				}
			}

			float Pr = 1.0*TP/(TP+FP);
			float Re = 1.0*TP/(TP+FN);
			F1_score = 2*Pr*Re/(Pr+Re);
			cout << "Pr = " << Pr << ", Re = " << Re << ", F1 score = " << F1_score << endl;
			cout << "TP = " << TP << ", FP = " << FP << ", FN = " << FN << endl;

			ofstream fin("E:/45 Thesis/result_icpr/result.csv",ios::app);
			if(!fin){
				cout << "open file error" <<endl; 
				cin.get();
				return 0;
			}
			
			fin << "TP," << TP << ",FP," << FP << ",FN," << FN << ",Pr," << Pr << ",Re," << Re << ",F1 score," << F1_score << ",prob," << p_t << ",R," << R << endl;
			fin.close();
		}
	}

	//return F1_score;
	return 0.0;
}


bool TLBO_test(Mat &img, Mat &mask, float threshold){
	int histSize = 256;
	float range[] = {0, 256} ;
	const float* histRange = {range};
	
	bool uniform = true; 
	bool accumulate = false;
	
	Mat hist;
	calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	Mat value = mask*hist;

	if(value.at<float>(0,0) > threshold)
		return false;
	else
		return true;
}
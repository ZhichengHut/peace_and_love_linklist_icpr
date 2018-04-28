#include "ExtractData.h"

int index = 0;

int total_c = 0;
int total_miss = 0;
int total_mitosis = 0;

vector<int> readCSV(string csvFile){
	ifstream fin(csvFile); //打开文件流操作  
    string line;   
	vector<int> csvData;

	while (getline(fin, line))   //整行读取，换行符“\n”区分，遇到文件尾标志eof终止读取  
    {
		stringstream line_tmp(line);
		string num;
		while(getline(line_tmp,num,',')){
			stringstream num_tmp(num);
			int position;
			num_tmp >> position;
			csvData.push_back(position);
		}
    }
	return csvData;
}

Mat preProcess(Mat img, float thresh){
	Mat out[3];
    split(img, out);

	Mat r = out[2];
	Mat g = out[1];
	Mat b = out[0];

	r.convertTo(r,CV_32FC1);
	g.convertTo(g,CV_32FC1);
	b.convertTo(b,CV_32FC1);

	Mat blue_ratio = 100*b/(1+r+g)*256/(1+r+g+b);

	blue_ratio.convertTo(blue_ratio,CV_8UC1);
	//imshow("blue ratio", blue_ratio);
	//waitKey(0);
	//cout << "blue ratio" << endl;
	//cout << blue_ratio(Rect(0,0,10,10)) << endl;

	GaussianBlur( blue_ratio, blue_ratio, Size(3,3), 0, 0, BORDER_DEFAULT );
	//imshow("GaussianBlur", blue_ratio);
	//waitKey(0);
	//cout << "br after gaussian" << endl;
	//cout << blue_ratio(Rect(0,0,10,10)) << endl;

	int kernel_size = 3;
	int scale = 1;
	int delta = 0.5;
	int ddepth = CV_16U;

	Laplacian( blue_ratio, blue_ratio, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
	convertScaleAbs(blue_ratio, blue_ratio);
	//imshow("Laplacian", blue_ratio);
	//waitKey(0);

	//cout << "after laplace" << endl;
	//cout << blue_ratio(Rect(0,0,10,10)) << endl;

	threshold(blue_ratio, blue_ratio, thresh*255,255 ,THRESH_BINARY);
	//imshow("threshold", blue_ratio);
	//waitKey(0);
	//cout << "after threshold" << endl;
	//cout << blue_ratio(Rect(0,0,10,10)) << endl;
	//cin.get();

	//imshow("brgl" , blue_ratio);
	//waitKey();

	Mat kernel = Mat::ones(3,3,CV_32FC1);
	int iteration = 5;
	
	/// Apply the close operation
	morphologyEx(blue_ratio,blue_ratio, MORPH_CLOSE, kernel, Point(-1,-1), iteration);
	//imshow("close", blue_ratio);
	//waitKey(0);

	return blue_ratio;
}

vector<Point2i> getCenter(Mat img, int R){
	vector<vector<Point>> contours;
	vector<Vec4i> heirarchy;
	vector<Point2i> center;
	//vector<int> radius;

	//find contours
	findContours(img.clone(), contours, heirarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

	size_t count = contours.size();

	/*for(int i=0; i<count; i++){
		if(contourArea(contours[i])>5){
			Point2f c;
			float r;
			minEnclosingCircle(contours[i], c, r);
			center.push_back(c);
		}
	}*/
	for( int i=0; i<count; i++)
	{
		Point2f c;
		float r;
		minEnclosingCircle(contours[i], c, r);

		if(r>R){
			center.push_back(c);
			//radius.push_back(r);
		}
	}

	contours.clear();
	vector<vector<Point>>().swap(contours);
	heirarchy.clear();
	vector<Vec4i>().swap(heirarchy);

	return center;
}

void clearFold(string out_fold){
	if(!access(out_fold.c_str(), 0)){
		char curDir[100];
		sprintf(curDir,out_fold.c_str());
		
		DIR* pDIR;
		struct dirent *entry;
		struct stat s;
		stat(curDir,&s);
		
		pDIR=opendir(curDir);
		int i = 0;
		
		while(entry = readdir(pDIR)){
			stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
			if (((s.st_mode & S_IFMT) != S_IFDIR) && ((s.st_mode & S_IFMT) == S_IFREG)){
				remove((curDir + string(entry->d_name)).c_str());	
				/*cout << string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) << endl;
				cout << (string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "png") << endl;
				cout << (string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "csv") << endl;
				cin.get();*/
			}
		}
	}
	else
		mkdir(out_fold.c_str());

	//cout << "dir complete" << endl;
}


void saveTrainData(string img_file, string csv_file, string out_fold, float thresh, int width, int R, int rand_num){
	Mat img = imread(img_file,1);
	Mat blue_ratio = preProcess(img, thresh);

	//int R = 10;
	vector<Point2i> center = getCenter(blue_ratio, R);
	int cell_num = center.size();
	for(int i=0; i<rand_num; i++){
		int x = rand() % (img.cols-width);
		int y = rand() % (img.rows-width);
		//cout << x << " " << y << endl; 
		center.push_back(Point2i(x,y));
	}

	vector<int> csvData = readCSV(csv_file);
	int mitosis_num = csvData.size()/2;

	//int width = 30;

	int candidate_count = 0; 

	for(int i=0; i<center.size(); i++){
		int x = center[i].x;
		int y = center[i].y;

		//whether the position is out of bound
		if(x < width + 1)
			x = width + 1;
		else if(x > img.cols - width)
			x = img.cols - width;

		if(y < width + 1)
            y = width + 1;
        else if(y > img.rows - width)
            y = img.rows - width;

		bool label = false;
		for(int j=0; j<mitosis_num; j++){
			//if(((x-width)<csvData[2*j+1]) && ((x+width)>csvData[2*j+1]) && ((y-width)<csvData[2*j]) && ((y+width)>csvData[2*j])){
			if(sqrt(pow(x-csvData[2*j+1],2.0)+pow(y-csvData[2*j],2.0))<30){
				label = true;
				candidate_count++;
				break;
			}
		}

		if(label)
			continue;
		else{
			char img_name[100];
			sprintf(img_name, "%s%04i_0.png", out_fold.c_str(), index);
			imwrite(img_name, img(Rect(x-width,y-width,2*width,2*width)));
			index ++;
		}
	}

	cout << "cell: " << cell_num << " candidate: " << candidate_count << ", mitosis_num: " << mitosis_num << endl;
	total_c += cell_num;
	total_miss += max(0, mitosis_num-candidate_count);
	total_mitosis += mitosis_num;

	for(int i=0; i<mitosis_num; i++){
		bool x_label = true;
		bool y_label = true;
		for(int p=0; p<3; p++){
		//for(int p=0; p<1; p++){
			int x = csvData[2*i+1]+5*(p-1);
			//int x = csvData[2*i+1];
			if(x < width + 1 && x_label){
				x = width + 1;
				x_label = false;
			}
			else if(x < width + 1 && !x_label)
				continue;

			if(x > img.cols - width && x_label){
				x = img.cols - width;
				x_label = false;
			}
			else if(x > img.cols - width && !x_label)
				continue;

			for(int q=0; q<3; q++){
			//for(int q=0; q<1; q++){
				int y = csvData[2*i]+5*(q-1);
				//int y = csvData[2*i];
				if(y < width + 1 && y_label){
					y = width + 1;
					y_label = false;
				}
				else if(y < width + 1 && !y_label)
					continue;
				
				if(y > img.rows - width && y_label){
					y = img.rows - width;
					y_label = false;
				}
				else if(y > img.rows - width && !y_label)
					continue;
				
				char img_name[100];
				sprintf(img_name, "%s%04i_1.png", out_fold.c_str(), index);
				imwrite(img_name, img(Rect(x-width,y-width,2*width,2*width)));
				index ++;

				//transpose
				Mat img_t;
				transpose(img(Rect(x-width,y-width,2*width,2*width)), img_t);
				sprintf(img_name, "%s%04i_1.png", out_fold.c_str(), index);
				imwrite(img_name, img_t);
				index ++;

				//rotate 90 degree
				flip(img_t, img_t, 1);
				sprintf(img_name, "%s%04i_1.png", out_fold.c_str(), index);
				imwrite(img_name, img_t);
				index ++;

				//rotate 180 degree
				transpose(img_t, img_t);
				flip(img_t, img_t, 0);
				sprintf(img_name, "%s%04i_1.png", out_fold.c_str(), index);
				imwrite(img_name, img_t);
				index ++;

				//rotate 270 degree
				transpose(img_t, img_t);
				flip(img_t, img_t, 0);
				sprintf(img_name, "%s%04i_1.png", out_fold.c_str(), index);
				imwrite(img_name, img_t);
				index ++;
			}
		}
	}
}

void saveTrainData(string img_file, string out_fold, float thresh, int width, int R, int rand_num){
	Mat img = imread(img_file,1);
	Mat blue_ratio = preProcess(img, thresh);

	//int R = 10;
	vector<Point2i> center = getCenter(blue_ratio, R);
	int cell_num = center.size();
	for(int i=0; i<rand_num; i++){
		int x = rand() % (img.cols-width);
		int y = rand() % (img.rows-width);
		//cout << x << " " << y << endl; 
		center.push_back(Point2i(x,y));
	}

	//int width = 30;

	for(int i=0; i< center.size(); i++){
		int x = center[i].x;
		int y = center[i].y;

		if(x < width + 1)
			x = width + 1;
		else if(x > img.cols - width)
			x = img.cols - width;

		if(y < width + 1)
            y = width + 1;
        else if(y > img.rows - width)
            y = img.rows - width;
		
		char img_name[100];
		sprintf(img_name, "%s%04i_0.png", out_fold.c_str(), index);
		imwrite(img_name, img(Rect(x-width,y-width,2*width,2*width)));
		index ++;
	}
}

void getTrainingSet(string train_fold, string out_fold, float thresh, int width, int R, int rand_num){
	vector<string> tif_set, csv_set;
	clearFold(out_fold);

	char delim = '/';

    char curDir[100];
    
    for(int c=1; c<=9; c++){
		//sprintf(curDir, "%s%c%i%c", dataPath.c_str(), delim, c, delim);
		sprintf(curDir, "%s%02i", train_fold.c_str(), c);
		//cout << curDir << endl;

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
						//cout << string(entry->d_name) << endl;
						if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "csv")
							csv_set.push_back(curDir + string("/") + string(entry->d_name));
						else
							tif_set.push_back(curDir + string("/") + string(entry->d_name));
					}
				}
			}
		}
	}

	int j = 0;
	for(int i=0; i<tif_set.size(); i++){
		if(j<csv_set.size() && tif_set[i].substr(train_fold.length(),5) == csv_set[j].substr(train_fold.length(),5)){
			cout << tif_set[i] << endl;
			//cout << csv_set[j] << endl;
			saveTrainData(tif_set[i], csv_set[j++], out_fold, thresh, width, R, rand_num);
		}
		else{
			cout << tif_set[i] << endl;
			saveTrainData(tif_set[i], out_fold, thresh, width, R, rand_num);
		}
	}
}

//for the second filter
void saveTrainData(string img_file, string csv_file, string out_fold, float thresh, int width, int R){
	Mat img = imread(img_file,1);
	Mat blue_ratio = preProcess(img, thresh);

	//int R = 10;
	vector<Point2i> center = getCenter(blue_ratio, R);
	int cell_num = center.size();

	vector<int> csvData = readCSV(csv_file);
	int mitosis_num = csvData.size()/2;

	//int width = 30;

	int candidate_count = 0; 

	for(int i=0; i<center.size(); i++){
		int x = center[i].x;
		int y = center[i].y;

		//whether the position is out of bound
		if(x < width + 1)
			x = width + 1;
		else if(x > img.cols - width)
			x = img.cols - width;

		if(y < width + 1)
            y = width + 1;
        else if(y > img.rows - width)
            y = img.rows - width;

		bool label = false;
		for(int j=0; j<mitosis_num; j++){
			//if(((x-width)<csvData[2*j+1]) && ((x+width)>csvData[2*j+1]) && ((y-width)<csvData[2*j]) && ((y+width)>csvData[2*j])){
			if(sqrt(pow(x-csvData[2*j+1],2.0)+pow(y-csvData[2*j],2.0))<30){
				label = true;
				candidate_count++;
				break;
			}
		}

		if(label)
			continue;
		else{
			char img_name[100];
			sprintf(img_name, "%s%04i_0.png", out_fold.c_str(), index);
			imwrite(img_name, img(Rect(x-width,y-width,2*width,2*width)));
			index ++;
		}
	}

	cout << "cell: " << cell_num << " candidate: " << candidate_count << ", mitosis_num: " << mitosis_num << endl;
	total_c += cell_num;
	total_miss += max(0, mitosis_num-candidate_count);
	total_mitosis += mitosis_num;

	for(int i=0; i<mitosis_num; i++){
		bool x_label = true;
		bool y_label = true;
		for(int p=0; p<3; p++){
		//for(int p=0; p<1; p++){
			int x = csvData[2*i+1]+5*(p-1);
			//int x = csvData[2*i+1];
			if(x < width + 1 && x_label){
				x = width + 1;
				x_label = false;
			}
			else if(x < width + 1 && !x_label)
				continue;

			if(x > img.cols - width && x_label){
				x = img.cols - width;
				x_label = false;
			}
			else if(x > img.cols - width && !x_label)
				continue;

			for(int q=0; q<3; q++){
			//for(int q=0; q<1; q++){
				//int y = csvData[2*i]+5*(q-1);
				int y = csvData[2*i];
				if(y < width + 1 && y_label){
					y = width + 1;
					y_label = false;
				}
				else if(y < width + 1 && !y_label)
					continue;
				
				if(y > img.rows - width && y_label){
					y = img.rows - width;
					y_label = false;
				}
				else if(y > img.rows - width && !y_label)
					continue;
				
				char img_name[100];
				sprintf(img_name, "%s%04i_1.png", out_fold.c_str(), index);
				imwrite(img_name, img(Rect(x-width,y-width,2*width,2*width)));
				index ++;
			}
		}
	}
}

void saveTrainData(string img_file, string out_fold, float thresh, int width, int R){
	Mat img = imread(img_file,1);
	Mat blue_ratio = preProcess(img, thresh);

	//int R = 10;
	vector<Point2i> center = getCenter(blue_ratio, R);
	int cell_num = center.size();

	//int width = 30;

	for(int i=0; i< center.size(); i++){
		int x = center[i].x;
		int y = center[i].y;

		if(x < width + 1)
			x = width + 1;
		else if(x > img.cols - width)
			x = img.cols - width;

		if(y < width + 1)
            y = width + 1;
        else if(y > img.rows - width)
            y = img.rows - width;
		
		char img_name[100];
		sprintf(img_name, "%s%04i_0.png", out_fold.c_str(), index);
		imwrite(img_name, img(Rect(x-width,y-width,2*width,2*width)));
		index ++;
	}
}

void getTrainingSet(string train_fold, string out_fold, float thresh, int width, int R){
	vector<string> tif_set, csv_set;
	clearFold(out_fold);

	char delim = '/';

    char curDir[100];
    
    for(int c=1; c<=9; c++){
		//sprintf(curDir, "%s%c%i%c", dataPath.c_str(), delim, c, delim);
		sprintf(curDir, "%s%02i", train_fold.c_str(), c);
		//cout << curDir << endl;

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
						//cout << string(entry->d_name) << endl;
						if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "csv")
							csv_set.push_back(curDir + string("/") + string(entry->d_name));
						else
							tif_set.push_back(curDir + string("/") + string(entry->d_name));
					}
				}
			}
		}
	}

	int j = 0;
	for(int i=0; i<tif_set.size(); i++){
		if(j<csv_set.size() && tif_set[i].substr(train_fold.length(),5) == csv_set[j].substr(train_fold.length(),5)){
			cout << tif_set[i] << endl;
			//cout << csv_set[j] << endl;
			saveTrainData(tif_set[i], csv_set[j++], out_fold, thresh, width, R);
		}
		else{
			cout << tif_set[i] << endl;
			saveTrainData(tif_set[i], out_fold, thresh, width, R);
		}
	}
}

void getTestImg(string curDir, string img_name, float thresh, int width, int R){
	string out_fold = curDir + "/" + img_name.substr(0,2) + "/";
	string img_file = curDir + "/" + img_name;

	clearFold(out_fold);

	Mat img = imread(img_file,1);
	Mat blue_ratio = preProcess(img, thresh);

	//int R = 10;
	vector<Point2i> center = getCenter(blue_ratio, R);
	int cell_num = center.size();

	for(int i=0; i< cell_num; i++){
		int x = center[i].x;
		int y = center[i].y;

		if(x < width + 1)
			x = width + 1;
		else if(x > img.cols - width)
			x = img.cols - width;

		if(y < width + 1)
            y = width + 1;
        else if(y > img.rows - width)
            y = img.rows - width;
		
		char img_name[100];
		sprintf(img_name, "%s%04i_%04i.png", out_fold.c_str(), x,y);
		imwrite(img_name, img(Rect(x-width,y-width,2*width,2*width)));
	}
}

void getTestingSet(string test_fold, float thresh, int width, int R){
	vector<string> tif_set, csv_set;

	char delim = '/';

    char curDir[100];
    
    for(int c=10; c<=12; c++){
		list<Mat> imgList;
		sprintf(curDir, "%s%02i", test_fold.c_str(), c);
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
						if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "tif")
							getTestImg(curDir, entry->d_name, thresh, width, R);
					}
				}
			}
		}
	}
}

void extractData(string train_fold, string test_fold, string out_fold, float train_thresh, float test_thresh, bool get_train, bool get_test, int width, int R, int rand_num){
	if(get_train){
		getTrainingSet(train_fold, out_fold, train_thresh, width, R, rand_num);
		cout << "Extracted training set completed" << endl;
		cout << "total cell: " << total_c << ", total miss: " << total_miss << ", total mitosis: " << total_mitosis << endl;
		//cin.get();
	}

	if(get_test){
		getTestingSet(test_fold, test_thresh, width, R);
		cout << "Extracted testing set completed" << endl;
	}
};

void extractData(string train_fold, string out_fold, float train_thresh, bool get_train, int width, int R){
	if(get_train){
		getTrainingSet(train_fold, out_fold, train_thresh, width, R);
			cout << "Extracted training set completed" << endl;
			cout << "total cell: " << total_c << ", total miss: " << total_miss << ", total mitosis: " << total_mitosis << endl;
	}
};
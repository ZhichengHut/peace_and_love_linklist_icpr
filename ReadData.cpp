#include "ReadData.h"

void readTrainData(string path, vector<Mat>& imgList, vector<int>& labelList, int &pos_num, int &neg_num){
	printf("Read training data starts\n");
	//string path = "D:/Bachelor Thesis/version1.0/dataset/";
	
	char curDir[100];
	sprintf(curDir,path.c_str());

	DIR* pDIR;
	struct dirent *entry;
	struct stat s;
	stat(curDir,&s);

	pDIR=opendir(curDir);
	int i = 0;

	while(entry = readdir(pDIR)){
		stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
		if (((s.st_mode & S_IFMT) != S_IFDIR) && ((s.st_mode & S_IFMT) == S_IFREG)){
			Mat img = imread((curDir + string(entry->d_name)).c_str(),0);
			//img.convertTo(img, CV_32FC1);
			//integral(img,img);
			imgList.push_back(img);
			labelList.push_back(entry->d_name[strlen(entry->d_name)-5]-'0');
			pos_num += (entry->d_name[strlen(entry->d_name)-5]-'0');
			neg_num += ('1'-entry->d_name[strlen(entry->d_name)-5]);
		}
	}
	printf("Read training data ends\n");
}



/*void readTestData(string path, vector<Mat>& imgList, vector<int>& X, vector<int>& Y){
	printf("Read training data starts\n");
	DIR* pDIR;
	
	struct dirent *entry;
	struct stat s;
	
	stat(path.c_str(),&s);
	
	// if path is a directory
	if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
		if(pDIR=opendir(path.c_str())){
			//for all entries in directory
			while(entry = readdir(pDIR)){
				stat((path.c_str() + string("/") + string(entry->d_name)).c_str(),&s);
				if (((s.st_mode & S_IFMT ) != S_IFDIR ) && ((s.st_mode & S_IFMT) == S_IFREG )){
					if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "tif"){
						string sub_curDIR = string(path.c_str()) + "/" + string(entry->d_name).substr(0,2);
						cout << "sub_curDIR = " << sub_curDIR << endl;
						
						DIR* subDIR;
						struct dirent *sub_entry;
						struct stat sub_s;
						
						stat(sub_curDIR.c_str(), &sub_s);
						
						if((sub_s.st_mode & S_IFMT) == S_IFDIR ){
							if(subDIR=opendir(sub_curDIR.c_str())){
								while(sub_entry = readdir(subDIR)){
									stat((sub_curDIR + string("/") + string(sub_entry->d_name)).c_str(),&sub_s);
									if (((sub_s.st_mode & S_IFMT ) != S_IFDIR ) && ((sub_s.st_mode & S_IFMT) == S_IFREG )){
										if(string(sub_entry->d_name).substr(string(sub_entry->d_name).find_last_of('.') + 1) == "png"){
											cout << "sub_entry->d_name = " << sub_entry->d_name << endl;
											Mat img_tmp = imread(sub_curDIR + string("/") + string(sub_entry->d_name), 1);
											int x = atoi(string(sub_entry->d_name).substr(0,4).c_str());
											int y = atoi(string(sub_entry->d_name).substr(5,4).c_str());
											imgList.push_back(img_tmp);
											X.push_back(x);
											Y.push_back(y);
										}
									}
								}
							}
						}
					}
				}
			}		
		}
	}
	printf("Read training data ends\n");
}*/
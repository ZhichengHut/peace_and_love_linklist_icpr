#include "Data.h"


Data::Data(Mat I, int L){
	I.copyTo(img);
	label = L;
}

Mat Data::get_Img(){
	return img;
}

int Data::get_Lab(){
	return label;
}
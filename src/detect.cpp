/*
 * detect.cpp
 *
 *  Created on: 2017Äê8ÔÂ22ÈÕ
 *      Author: JYF
 */

#include "detect.h"
#include <windows.h>

void trainMSHog(string label_file, int iter){
	vector<string> file_names;
	vector<int> labels;
	vector<HogParam> params;
	readLabelFile(label_file, file_names, labels);
	stringstream ss;
	ss<<"./params/params_"<<ID<<".txt";
	readParamsFromFile(ss.str(), params);
	printParams(params);

	trainHog(file_names, labels, params, iter);
}

void trainHistEntry(string label_file, int iter){
	vector<string> file_names;
	vector<int> labels;
	readLabelFile(label_file, file_names, labels);
	trainHist(file_names, labels, iter);
}

int detectMSHog(Mat& image, MySVM& svm, vector<HogParam>& params){
	return detectHog(image, svm, params);
}

float detectAndCount(string test_file){
	vector<string> file_names;
	vector<int> labels;
	readLabelFile(test_file, file_names, labels);
	assert(file_names.size()==labels.size());

	int id=ID;
	char file_path[100];
	sprintf(file_path,"./uav_config/svm_person_%d.xml", id);
	const char* svm_file=(const char*)file_path;
	MySVM svm=ml::SVM::load(svm_file);

	stringstream ss;
	ss<<"./uav_config/params_"<<ID<<".txt";
	vector<HogParam> params;
	readParamsFromFile(ss.str(), params);

	int correct=0;
	int total=file_names.size();
	LARGE_INTEGER t_start, t_end, t_freq;
	double t_time;
	QueryPerformanceFrequency(&t_freq);
	QueryPerformanceCounter(&t_start);
	for(unsigned int i=0;i<file_names.size();++i){
		Mat image=imread(file_names[i]);
		int res=detectMSHog(image, svm, params);
		if(res==labels[i]){	++correct;	}
	}
	QueryPerformanceCounter(&t_end);
	t_time=1.0*(t_end.QuadPart-t_start.QuadPart)/t_freq.QuadPart;
	cout<<"test time: "<<t_time<<" s"<<endl;
	return (1.0*correct)/(1.0*total);
}




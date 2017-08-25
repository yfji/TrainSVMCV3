#include "trainsvm.h"
#include <time.h>
#include <windows.h>

string svm_file_hist="./svm_hist.xml";
string svm_file_hog="./svm_hog.xml";

void getHogFeature(Mat& image, HogParam& param, vector<float>& hogFeat, bool cvt){
	Mat gray=image;
	if(cvt){
		int ch=image.channels();
		if(ch==3){
			cvtColor(image, gray, CV_BGR2GRAY);
		}
		resize(gray, gray, param.winSize);
		normalize(gray, gray, 0, 255, NORM_MINMAX);
	}
	HOGDescriptor hog;
	hog.blockSize=Size(param.blockSize, param.blockSize);
	hog.cellSize=Size(param.cellSize, param.cellSize);
	hog.blockStride=Size(param.stride, param.stride);
	hog.winSize=param.winSize;
	hog.compute(image, hogFeat);
}

void getMSHogFeature(Mat& image, vector<HogParam>& params, vector<float>& msHogFeat){
	Mat gray=image;
	int ch=image.channels();
	if(ch==3){
		cvtColor(image, gray, CV_BGR2GRAY);
	}
	resize(gray, gray, params[0].winSize);
	normalize(gray, gray, 0, 255, NORM_MINMAX);
	for(unsigned int i=0;i<params.size();++i){
		HogParam& param=params[i];
		vector<float> ssHogFeat;
		getHogFeature(gray, param, ssHogFeat, false);
		for(unsigned int j=0;j<ssHogFeat.size();++j)
			msHogFeat.push_back(ssHogFeat[j]);
	}
}

void getHistogramFeature(Mat& image, float*& feat){
	resize(image, image, Size(HIST_SIZE, HIST_SIZE));
	int h=image.rows;
	int w=image.cols;
	int stride=w*3;

	int total=256/HIST_SEG;
	total=total*total*total;
	uchar* data=image.data;
	if(feat==NULL){
		cout<<"No storage assigned"<<endl;
		feat=new float[total];
	}
	for(int i=0;i<total;++i){	feat[i]=0;	}
	for(int i=0;i<h*stride;i+=3){
		int b=(int)data[i];
		int g=(int)data[i+1];
		int r=(int)data[i+2];

		b/=HIST_SEG;
		g/=HIST_SEG;
		r/=HIST_SEG;

		feat[b*256+g*16+r]+=1;
	}
	Normalize<float>(feat, total, 0.0f, 1.0f);
}

void trainHog(vector<string>& file_names, vector<int>& labels, vector<HogParam>& params, int iter){
	stringstream ss;
	int id=ID;
	char file_path[100];
	sprintf(file_path,"./params/svm_hog_%d.xml",id);
	const char* svm_file=(const char*)file_path;
	//ss<<"./config/svm_person_"<<id<<".xml";
	//const char* svm_file=ss.str().c_str();
	cout<<svm_file<<endl;
	ifstream in;
	in.open(svm_file, ios::in);
	if(in){
		in.close();
		cout<<"found existing xml file"<<endl;
		return;
	}
	Mat sampleMat, labelMat;
	vector<float> msHogFeat;
	float* samplePtr;
	int* labelPtr;
	cout<<"Preparing data..."<<endl;
	for(unsigned int i=0;i<file_names.size();++i){
		Mat image=imread(file_names[i]);
		getMSHogFeature(image, params, msHogFeat);
		if(sampleMat.empty()){
			sampleMat.create(Size(msHogFeat.size(), file_names.size()), CV_32F);
			cout<<file_names.size()<<","<<msHogFeat.size()<<endl;
		}
		if(labelMat.empty()){
			labelMat.create(Size(1, file_names.size()), CV_32S);
		}
		samplePtr=sampleMat.ptr<float>(i);
		labelPtr=labelMat.ptr<int>(i);
		for(unsigned int j=0;j<msHogFeat.size();++j){
			samplePtr[j]=msHogFeat[j];
		}
		labelPtr[0]=labels[i];
		vector<float>().swap(msHogFeat);
	}
	cout<<"data prepared"<<endl;
	MySVM svm=ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, iter, FLT_EPSILON));

	LARGE_INTEGER t_start, t_end, t_freq;
	double t_time;
	QueryPerformanceFrequency(&t_freq);
	QueryPerformanceCounter(&t_start);

	svm->train(sampleMat, ml::ROW_SAMPLE, labelMat);

	QueryPerformanceCounter(&t_end);
	t_time=1.0*(t_end.QuadPart-t_start.QuadPart)/t_freq.QuadPart;
	svm->save(svm_file);
	cout<<"finished"<<endl;
	cout<<"train time: "<<t_time<<" s"<<endl;
}

int detectHog(Mat& image, MySVM& svm, vector<HogParam>& params){
	vector<float> hogFeat;

	getMSHogFeature(image, params, hogFeat);

	Mat sampleMat(1,hogFeat.size(), CV_32F);
	float* ptr=sampleMat.ptr<float>(0);
	for(size_t k=0;k<hogFeat.size();++k)
		ptr[k]=hogFeat[k];
	return (int)(svm->predict(sampleMat));
}

void trainHist(vector<string>& file_names, vector<int>& labels, int iter){
	const char* svm_file="./svm_hist.xml";
	ifstream in;
	in.open(svm_file, ios::in);
	if(in){
		in.close();
		cout<<"found existing xml file"<<endl;
		return;
	}
	int featLen=256/HIST_SEG;
	featLen=featLen*featLen*featLen;

	float* feat=new float[featLen];
	float* samplePtr;
	int* labelPtr;

	Mat sampleMat(file_names.size(), featLen, CV_32F);
	Mat labelMat(file_names.size(), 1, CV_32S);

	cout<<"Preparing data..."<<endl;
	for(size_t i=0;i<file_names.size();++i){
		Mat image=imread(file_names[i]);
		if(image.rows==0 or image.cols==0){
			cout<<file_names[i]<<endl;
		}
		getHistogramFeature(image, feat);
		samplePtr=sampleMat.ptr<float>(i);
		labelPtr=labelMat.ptr<int>(i);
		for(int j=0;j<featLen;++j){
			samplePtr[j]=1.0*feat[j];
		}
		labelPtr[0]=labels[i];
	}
	cout<<"data prepared"<<endl;
	MySVM svm=ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, iter, FLT_EPSILON));
	svm->train(sampleMat, ml::ROW_SAMPLE, labelMat);
	svm->save(svm_file);
	cout<<"finished"<<endl;
	delete feat;
}

int detectHist(Mat& image, MySVM& svm){
	int featLen=HIST_SIZE/HIST_SEG;
	featLen=featLen*featLen*featLen;
	float* feat=new float[featLen];

	Mat sampleMat(1, featLen, CV_32FC1);

	float* samplePtr=sampleMat.ptr<float>(0);
	getHistogramFeature(image, feat);
	for(int j=0;j<featLen;++j){
		samplePtr[j]=1.0*feat[j];
	}
	return (int)(svm->predict(sampleMat));
}


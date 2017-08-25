#include "detect.h"

int main(int argc, char** argv){
	//string label_file="I:/TestOpenCV/Videos/pot_train/label_hist.txt";
	string label_file="I:/TestOpenCV/Videos/pot_train/label.txt";

	MySVM svm_hist=ml::SVM::create();
	MySVM svm_hog=ml::SVM::create();

	//trainHistEntry(label_file, 1e4);
	trainMSHog(label_file);
	return 0;
}

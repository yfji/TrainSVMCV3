/*
 * detect.h
 *
 *  Created on: 2017Äê8ÔÂ22ÈÕ
 *      Author: JYF
 */

#ifndef SRC_DETECT_H_
#define SRC_DETECT_H_

#include "trainsvm.h"
#include "util.h"

void trainMSHog(string label_file, int iter=1e6);

int detectMSHog(Mat& image, MySVM& svm, vector<HogParam>& params);

void trainHistEntry(string label_file, int iter=1e6);

float detectAndCount(string test_file);



#endif /* SRC_DETECT_H_ */

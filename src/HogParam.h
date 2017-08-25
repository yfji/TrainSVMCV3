/*
 * HogParam.h
 *
 *  Created on: 2017Äê8ÔÂ7ÈÕ
 *      Author: JYF
 */

#ifndef SRC_HOGPARAM_H_
#define SRC_HOGPARAM_H_
#include <opencv2\highgui\highgui.hpp>

struct HogParam{
	int blockSize;
	int cellSize;
	int stride;
	cv::Size winSize;
};


#endif /* SRC_HOGPARAM_H_ */

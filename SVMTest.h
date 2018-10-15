#ifndef __SVMTEST__
#define __SVMTEST__

#include "opencv2/ml.hpp"
#include<fstream>
#include"LBP.h"
using namespace cv;
using namespace std;


#define LOG_DEBUG_SVM_TEST(...)           
#define LOG_ERROR_SVM_TEST(...)             
#define LOG_INFO_SVM_TEST(...)                  
#define LOG_WARN_SVM_TEST(...)                  

#define CELL_SIZE   16

class SVMTest
{
public:
	SVMTest(const string &_trainDataFileList,
		const string &_testDataFileList,
		const string &_svmModelFilePath,
		const string &_predictResultFilePath,
		int svmType, 
		int kernel,
		double c, 
		double coef, 
		double degree,
		double gamma, 
		double nu, 
		double p
	);
	bool Initialize();
	virtual ~SVMTest();

	void Train();
	void Predict();

private:
	string trainDataFileList;
	string testDataFileList;
	string svmModelFilePath;
	string predictResultFilePath;

	SVMParams params;

	// SVM
	CvSVM svm;


	LBP lbp;


};


#endif // SVMTEST

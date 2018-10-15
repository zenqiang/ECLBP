#include"SVMTest.h"
#include <stdio.h> 
#include <string.h> 
#include <vector>

SVMTest::SVMTest(const string &_trainDataFileList,
	const string &_testDataFileList,
	const string &_svmModelFilePath,
	const string &_predictResultFilePath,

	int svmType, // See SVM::Types. Default value is SVM::C_SVC.
	int kernel,
	double c, // For SVM::C_SVC, SVM::EPS_SVR or SVM::NU_SVR. Default value is 0.
	double coef,  // For SVM::POLY or SVM::SIGMOID. Default value is 0.
	double degree, // For SVM::POLY. Default value is 0.
	double gamma, // For SVM::POLY, SVM::RBF, SVM::SIGMOID or SVM::CHI2. Default value is 1.
	double nu,  // For SVM::NU_SVC, SVM::ONE_CLASS or SVM::NU_SVR. Default value is 0.
	double p // For SVM::EPS_SVR. Default value is 0.
) :
	trainDataFileList(_trainDataFileList),
	testDataFileList(_testDataFileList),
	svmModelFilePath(_svmModelFilePath),
	predictResultFilePath(_predictResultFilePath)
{
	// set svm param
	//params.svm_type = CvSVM::C_SVC;
	//params.kernel_type = CvSVM::LINEAR;
	params.svm_type = svmType;
	params.kernel_type = kernel;
	//SVM训练过程的终止条件, max_iter:最大迭代次数  epsilon:结果的精确性  
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, FLT_EPSILON);
	params.C = c;
	params.coef0 = coef;
	params.degree = degree;
	params.gamma = gamma;
	params.nu = nu;
	params.p = p;

}

bool SVMTest::Initialize()
{
	// initialize log
	//InitializeLog("SVMTest");

	return true;

}

SVMTest::~SVMTest()
{
}

int GetLable(string lable)
{
	int lable_int = 0;

	if (lable == "banded")
		lable_int = 0;
	else if (lable == "blotchy")
		lable_int = 1;
	else if (lable == "braided")
		lable_int = 2;
	else if (lable == "bubbly")
		lable_int = 3;
	else if (lable == "bumpy")
		lable_int = 4;
	else if (lable == "chequered")
		lable_int = 5;
	else if (lable == "cobwebbed")
		lable_int = 6;
	else if (lable == "cracked")
		lable_int = 7;
	else if (lable == "crosshatched")
		lable_int = 8;
	else if (lable == "crystalline")
		lable_int = 9;
	else if (lable == "dotted")
		lable_int = 10;
	else if (lable == "fibrous")
		lable_int = 11;
	else if (lable == "flecked")
		lable_int = 12;
	else if (lable == "freckled")
		lable_int = 13;
	else if (lable == "frilly")
		lable_int = 14;
	else if (lable == "gauzy")
		lable_int = 15;
	else if (lable == "grid")
		lable_int = 16;
	else if (lable == "grooved")
		lable_int = 17;
	else if (lable == "honeycombed")
		lable_int = 18;
	else if (lable == "interlaced")
		lable_int = 19;
	else if (lable == "knitted")
		lable_int = 20;
	else if (lable == "lacelike")
		lable_int = 21;
	else if (lable == "lined")
		lable_int = 22;
	else if (lable == "marbled")
		lable_int = 23;
	else if (lable == "matted")
		lable_int = 24;
	else if (lable == "meshed")
		lable_int = 25;
	else if (lable == "paisley")
		lable_int = 26;
	else if (lable == "perforated")
		lable_int = 27;
	else if (lable == "pitted")
		lable_int = 28;
	else if (lable == "pleated")
		lable_int = 29;
	else if (lable == "polka-dotted")
		lable_int = 30;
	else if (lable == "porous")
		lable_int = 31;
	else if (lable == "potholed")
		lable_int = 32;
	else if (lable == "scaly")
		lable_int = 33;
	else if (lable == "smeared")
		lable_int = 34;
	else if (lable == "spiralled")
		lable_int = 35;
	else if (lable == "sprinkled")
		lable_int = 36;
	else if (lable == "stained")
		lable_int = 37;
	else if (lable == "stratified")
		lable_int = 38;
	else if (lable == "striped")
		lable_int = 39;
	else if (lable == "studded")
		lable_int = 40;
	else if (lable == "swirly")
		lable_int = 41;
	else if (lable == "veined")
		lable_int = 42;
	else if (lable == "waffled")
		lable_int = 43;
	else if (lable == "woven")
		lable_int = 44;
	else if (lable == "wrinkled")
		lable_int = 45;
	else if (lable == "zigzagged")
		lable_int = 46;

	return lable_int;

}

void SVMTest::Train()
{
	// 读入训练样本图片路径和类别
	std::vector<string> imagePaths;
	std::vector<int> imageClasses;
	string line;
	std::ifstream trainingData(trainDataFileList, ios::out);
	while (getline(trainingData, line))
	{
		if (line.empty())
			continue;

		stringstream stream(line);
		string imagePath, imageClass;
		stream >> imagePath;
		stream >> imageClass;

		imagePaths.push_back(imagePath);
		int temp_class = atoi(imageClass.c_str());
		imageClasses.push_back(temp_class);
	}
	trainingData.close();

	printf("%d\n", imagePaths.size());

	// extract feature
	Mat featureVectorsOfSample;
	Mat classOfSample;
	printf("get feature...\n");
	LOG_INFO_SVM_TEST("get feature...");
	for (int i = 0; i <= imagePaths.size() - 1; ++i)
	{
		Mat srcImage = imread(imagePaths[i], -1);
		if (srcImage.empty() || srcImage.depth() != CV_8U)
		{
			printf("%s srcImage.empty()||srcImage.depth()!=CV_8U!\n", imagePaths[i].c_str());
			LOG_ERROR_SVM_TEST("%s srcImage.empty()||srcImage.depth()!=CV_8U!", imagePaths[i].c_str());
			continue;
		}
		//cvtColor(srcImage, srcImage, CV_BGR2GRAY);
		// extract feature
		Mat featureVector;
		lbp.ComputeECLBPFeatureVector(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVector);
		if (featureVector.empty())
			continue;

		featureVectorsOfSample.push_back(featureVector);
		classOfSample.push_back(imageClasses[i]);

		printf("get feature... %f% \n", (i + 1)*100.0 / imagePaths.size());
		LOG_INFO_SVM_TEST("get feature... %f", (i + 1)*100.0 / imagePaths.size());
	}

	printf("get feature done!\n");
	LOG_INFO_SVM_TEST("get feature done!");

	// train
	printf("training...\n");
	LOG_INFO_SVM_TEST("training...");
	double time1, time2;
	time1 = getTickCount();
	svm.train(featureVectorsOfSample, classOfSample, Mat(), Mat(), params);
	time2 = getTickCount();
	printf("训练时间:%f\n", (time2 - time1)*1000. / getTickFrequency());
	LOG_INFO_SVM_TEST("训练时间:%f", (time2 - time1)*1000. / getTickFrequency());
	printf("training done!\n");
	LOG_INFO_SVM_TEST("training done!");

	// save model
	//svm.save(svmModelFilePath.c_str());
}

void SVMTest::Predict()
{
	// predict
	std::vector<string> testImagePaths;
	std::vector<int> testImageClasses;
	string line;
	std::ifstream testData(testDataFileList, ios::out);
	while (getline(testData, line))
	{
		if (line.empty())
			continue;

		stringstream stream(line);
		string imagePath, imageClass;
		stream >> imagePath;
		stream >> imageClass;

		testImagePaths.push_back(imagePath);
		testImageClasses.push_back(atoi(imageClass.c_str()));

	}
	testData.close();

	printf("predicting...\n");
	LOG_INFO_SVM_TEST("predicting...");


	int numberOfRight = 0;
	//int numberOfRight_0 = 0;
	//int numberOfError_0 = 0;
	//int numberOfRight_1 = 0;
	//int numberOfError_1 = 0;

	std::ofstream fileOfPredictResult(predictResultFilePath, ios::out); //最后识别的结果
	double sum_Predict = 0, sum_ExtractFeature = 0;
	char line2[256] = { 0 };
	for (int i = 0; i < testImagePaths.size(); ++i)
	{
		Mat srcImage = imread(testImagePaths[i], -1);
		if (srcImage.empty() || srcImage.depth() != CV_8U)
		{
			printf("%s srcImage.empty()||srcImage.depth()!=CV_8U!\n", testImagePaths[i].c_str());
			LOG_ERROR_SVM_TEST("%s srcImage.empty()||srcImage.depth()!=CV_8U!", testImagePaths[i].c_str());
			continue;
		}

		//cvtColor(srcImage, srcImage, CV_BGR2GRAY);

		// extract feature
		double time1_ExtractFeature = getTickCount();
		Mat featureVectorOfTestImage;
		lbp.ComputeECLBPFeatureVector(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVectorOfTestImage);
		if (featureVectorOfTestImage.empty())
			continue;
		double time2_ExtractFeature = getTickCount();
		sum_ExtractFeature += (time2_ExtractFeature - time1_ExtractFeature) * 1000 / getTickFrequency();

		//对测试图片进行分类并写入文件
		double time1_Predict = getTickCount();
		int predictResult = svm.predict(featureVectorOfTestImage);
		double time2_Predict = getTickCount();
		sum_Predict += (time2_Predict - time1_Predict) * 1000 / getTickFrequency();

		sprintf(line2, "%s %d %d\n", testImagePaths[i].c_str(), testImageClasses[i], predictResult);
		fileOfPredictResult << line2;
		LOG_INFO_SVM_TEST("%s %d", testImagePaths[i].c_str(), predictResult);

		if (testImageClasses[i] == predictResult)
		{
			++numberOfRight;
		}

		//// 0
		//if ((testImageClasses[i] == 0) && (predictResult == 0))
		//{
		//	++numberOfRight_0;
		//}
		//if ((testImageClasses[i] == 0) && (predictResult != 0))
		//{
		//	++numberOfError_0;
		//}

		//// 1
		//if ((testImageClasses[i] == 1) && (predictResult == 1))
		//{
		//	++numberOfRight_1;
		//}
		//if ((testImageClasses[i] == 1) && (predictResult != 1))
		//{
		//	++numberOfError_1;
		//}

		printf("predicting...%f%\n", 100.0*(i + 1) / testImagePaths.size());
	}
	printf("predicting done!\n");
	LOG_INFO_SVM_TEST("predicting done!");

	printf("extract feature time：%f\n", sum_ExtractFeature / testImagePaths.size());
	LOG_INFO_SVM_TEST("extract feature time：%f", sum_ExtractFeature / testImagePaths.size());
	sprintf(line2, "extract feature time：%f\n", sum_ExtractFeature / testImagePaths.size());
	fileOfPredictResult << line2;

	printf("predict time：%f\n", sum_Predict / testImagePaths.size());
	LOG_INFO_SVM_TEST("predict time：%f", sum_Predict / testImagePaths.size());
	sprintf(line2, "predict time：%f\n", sum_Predict / testImagePaths.size());
	fileOfPredictResult << line2;


	double accuracy = (100.0*(numberOfRight)) / (testImageClasses.size());
	// 0
	//double accuracy_0 = (100.0*(numberOfRight_0)) / (numberOfError_0 + numberOfRight_0);
	printf("accuracy：%f\n", accuracy);
	LOG_INFO_SVM_TEST("accuracy：%f", accuracy);
	sprintf(line2, "accuracy：%f\n", accuracy);
	fileOfPredictResult << line2;

	int error_count = testImageClasses.size() - numberOfRight;
	printf("error：%d\n", error_count);
	LOG_INFO_SVM_TEST("error：%d", error_count);
	sprintf(line2, "error：%d\n", error_count);
	fileOfPredictResult << line2;

	// 1
	//double accuracy_1 = (100.0*numberOfRight_1) / (numberOfError_1 + numberOfRight_1);
	//printf("1：%f\n", accuracy_1);
	//LOG_INFO_SVM_TEST("1：%f", accuracy_1);
	//sprintf(line2, "1：%f\n", accuracy_1);
	//fileOfPredictResult << line2;

	//// accuracy
	//double accuracy_All = (100.0*(numberOfRight_1 + numberOfRight_0)) / (numberOfError_0 + numberOfRight_0 + numberOfError_1 + numberOfRight_1);
	//printf("accuracy：%f\n", accuracy_All);
	//LOG_INFO_SVM_TEST("accuracy:%f", accuracy_All);
	//sprintf(line2, "accuracy:%f\n", accuracy_All);
	//fileOfPredictResult << line2;

	fileOfPredictResult.close();

	while (1)
	{

	}

}

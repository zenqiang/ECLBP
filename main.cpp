#include"SVMTest.h"
void main()
{
	SVMTest svmTest("train.txt", // train filelist
		"test.txt", // test filelist
		"Classifier.xml",  // classifier
		"PredictResult.txt",  //predict result
		SVM::C_SVC,  // svmType
		SVM::LINEAR, // kernel
		1,   // c
		0,   // coef
		0,   // degree
		1,   // gamma
		0,   // nu
		0);  // p
	if (!svmTest.Initialize())
	{
		printf("initialize failed!\n");
		//LOG_ERROR_SVM_TEST("initialize failed!");
		return;
	}


	Mat src = imread("Outex-TC-00010/images/000000.bmp");
	Mat s = imread("KTH_TIPS/orange_peel/55-scale_1_im_1_col.png");

	cvtColor(src, src, CV_BGR2GRAY);

	Mat hist;

	LBP lbp;

	lbp.ComputeECLBPFeatureVector(src, src.size(), hist);

	int scale = 1;
	int size = hist.cols;
	Mat dstImage(size * scale, size, s.type(), Scalar(255, 255, 255));
	//获取最大值和最小值
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(hist, &minValue, &maxValue, 0, 0);  //  在cv中用的是cvGetMinMaxHistValue
												  //绘制出直方图
	int hpt = saturate_cast<int>(0.9 * size);
	for (int i = 0; i < size; i++)
	{
		float binValue = hist.at<float>(i);           //   注意hist中是float类型    
		int realValue = saturate_cast<int>(binValue * hpt / maxValue);
		//rectangle(dstImage,Point(i*scale, size - 1), Point((i+1)*scale - 1, size - realValue), Scalar(255));
		Scalar color;
		if (i < 2)
			color = Scalar(0, 0, 0);
		else if (i < 2 + 18)
			color = Scalar(255, 0, 0);
		else if (i < 2 + 18 + 18)
			color = Scalar(0, 255, 0);
		else if (i < 2 + 18 + 18 + 18)
			color = Scalar(0, 0, 255);
		else if (i < 2 + 18 + 18 + 18 + 18)
			color = Scalar(255, 255, 0);
		else if (i < 2 + 18 + 18 + 18 + 18 + 18)
			color = Scalar(0, 255, 255);
		else
			color = Scalar(255, 0, 255);

		line(dstImage, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realValue), color);
	}

	imshow("src", src);

	//resize(dstImage, dstImage, Size(500, 500), 0, 0, INTER_LINEAR);

	imshow("hist", dstImage);
	imwrite("hist.jpg", dstImage);

	waitKey(0);


	//svmTest.Train();
	//svmTest.Predict();

}

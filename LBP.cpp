#include"LBP.h"

//获取i中0,1的跳变次数
int LBP::GetHopCount(int i)
{
	// 转换为二进制
	int a[8] = { 0 };
	int k = 7;
	while (i)
	{
		// 除2取余
		a[k] = i % 2;
		i /= 2;
		--k;
	}

	// 计算跳变次数
	int count = 0;
	for (int k = 0; k<8; ++k)
	{
		// 注意，是循环二进制,所以需要判断是否为8
		if (a[k] != a[k + 1 == 8 ? 0 : k + 1])
		{
			++count;
		}
	}
	return count;

}

// 建立等价模式表
// 这里为了便于建立LBP特征图，58种等价模式序号从1开始:1~58,第59类混合模式映射为0
void LBP::BuildUniformPatternTable(int *table)
{
	memset(table, 0, 256 * sizeof(int));
	uchar temp = 1;
	for (int i = 0; i<256; ++i)
	{
		if (GetHopCount(i) <= 2)
		{
			table[i] = temp;
			temp++;
		}
	}

}

void LBP::ComputeLBPFeatureVector_256(const Mat &srcImage, Size cellSize, Mat &featureVector)
{
	// 参数检查，内存分配
	//CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);

	Mat LBPImage;
	ComputeLBPImage_256(srcImage, LBPImage);

	// 计算cell个数
	int widthOfCell = cellSize.width;
	int heightOfCell = cellSize.height;
	int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
	int numberOfCell_Y = srcImage.rows / heightOfCell;

	// 特征向量的个数
	int numberOfDimension = 256 * numberOfCell_X*numberOfCell_Y;
	featureVector.create(1, numberOfDimension, CV_32FC1);
	featureVector.setTo(Scalar(0));

	// 计算LBP特征向量
	int stepOfCell = srcImage.cols;
	int pixelCount = cellSize.width*cellSize.height;
	float *dataOfFeatureVector = (float *)featureVector.data;

	// cell的特征向量在最终特征向量中的起始位置
	int index = -256;
	for (int y = 0; y <= numberOfCell_Y - 1; ++y)
	{
		for (int x = 0; x <= numberOfCell_X - 1; ++x)
		{
			index += 256;

			// 计算每个cell的LBP直方图
			Mat cell = LBPImage(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			uchar *rowOfCell = cell.data;
			for (int y_Cell = 0; y_Cell <= cell.rows - 1; ++y_Cell, rowOfCell += stepOfCell)
			{
				uchar *colOfCell = rowOfCell;
				for (int x_Cell = 0; x_Cell <= cell.cols - 1; ++x_Cell, ++colOfCell)
				{
					++dataOfFeatureVector[index + colOfCell[0]];
				}
			}

			// 一定要归一化！否则分类器计算误差很大
			for (int i = 0; i <= 255; ++i)
				dataOfFeatureVector[index + i] /= pixelCount;

		}
	}

}

//srcImage:灰度图
//LBPImage:LBP图
void LBP::ComputeLBPImage_256(const Mat &srcImage, Mat &LBPImage)
{
	// 参数检查，内存分配
	//CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);
	LBPImage.create(srcImage.size(), srcImage.type());

	// 扩充原图像边界，便于边界处理
	Mat extendedImage;
	copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, BORDER_DEFAULT);

	// 计算LBP特征图
	int heightOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	int widthOfLBP = LBPImage.cols;
	uchar *rowOfExtendedImage = extendedImage.data + widthOfExtendedImage + 1;
	uchar *rowOfLBPImage = LBPImage.data;
	for (int y = 1; y <= heightOfExtendedImage - 2; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBP)
	{
		// 列
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfLBPImage = rowOfLBPImage;
		for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBPImage)
		{
			// 计算LBP值
			int LBPValue = 0;
			if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBPValue += 128;
			if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0])
				LBPValue += 64;
			if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBPValue += 32;
			if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0])
				LBPValue += 16;
			if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBPValue += 8;
			if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0])
				LBPValue += 4;
			if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBPValue += 2;
			if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0])
				LBPValue += 1;

			colOfLBPImage[0] = LBPValue;

		}  // x

	}// y


}

// cellSize:每个cell的大小,如16*16
void LBP::ComputeLBPFeatureVector_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector)
{
	// 参数检查，内存分配

	Mat LBPImage;

	ComputeLBPImage_Uniform(srcImage, LBPImage);

	// 计算cell个数
	int widthOfCell = cellSize.width;
	int heightOfCell = cellSize.height;
	int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
	int numberOfCell_Y = srcImage.rows / heightOfCell;

	// 特征向量的个数
	int numberOfDimension = 58 * numberOfCell_X*numberOfCell_Y;
	featureVector.create(1, numberOfDimension, CV_32FC1);
	featureVector.setTo(Scalar(0));

	// 计算LBP特征向量
	int stepOfCell = srcImage.cols;
	int index = -58;// cell的特征向量在最终特征向量中的起始位置
	float *dataOfFeatureVector = (float *)featureVector.data;
	for (int y = 0; y <= numberOfCell_Y - 1; ++y)
	{
		for (int x = 0; x <= numberOfCell_X - 1; ++x)
		{
			index += 58;

			// 计算每个cell的LBP直方图
			Mat cell = LBPImage(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			uchar *rowOfCell = cell.data;
			int sum = 0; // 每个cell的等价模式总数
			for (int y_Cell = 0; y_Cell <= cell.rows - 1; ++y_Cell, rowOfCell += stepOfCell)
			{
				uchar *colOfCell = rowOfCell;
				for (int x_Cell = 0; x_Cell <= cell.cols - 1; ++x_Cell, ++colOfCell)
				{
					if (colOfCell[0] != 0)
					{
						// 在直方图中转化为0~57，所以是colOfCell[0] - 1
						++dataOfFeatureVector[index + colOfCell[0] - 1];
						++sum;
					}
				}
			}

			// 一定要归一化！否则分类器计算误差很大
			for (int i = 0; i <= 57; ++i)
				dataOfFeatureVector[index + i] /= sum;

		}
	}
}

// 计算等价模式LBP特征图，为了方便表示特征图，58种等价模式表示为1~58,第59种混合模式表示为0
// 注：你可以将第59类混合模式映射为任意数值，因为要突出等价模式特征，所以非等价模式设置为0比较好
void LBP::ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage)
{
	// 参数检查，内存分配
	LBPImage.create(srcImage.size(), srcImage.type());

	// 计算LBP图
	// 扩充原图像边界，便于边界处理
	Mat extendedImage;
	copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, BORDER_DEFAULT);

	// 构建LBP 等价模式查找表
	//int table[256];
	//BuildUniformPatternTable(table);

	// LUT(256种每一种模式对应的等价模式)
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

	// 计算LBP
	int heightOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	int widthOfLBP = LBPImage.cols;
	uchar *rowOfExtendedImage = extendedImage.data + widthOfExtendedImage + 1;
	uchar *rowOfLBPImage = LBPImage.data;
	for (int y = 1; y <= heightOfExtendedImage - 2; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBP)
	{
		// 列
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfLBPImage = rowOfLBPImage;
		for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBPImage)
		{
			// 计算LBP值
			int LBPValue = 0;
			if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBPValue += 128;
			if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0])
				LBPValue += 64;
			if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBPValue += 32;
			if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0])
				LBPValue += 16;
			if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBPValue += 8;
			if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0])
				LBPValue += 4;
			if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBPValue += 2;
			if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0])
				LBPValue += 1;

			colOfLBPImage[0] = table[LBPValue];

		} // x

	}// y
}

// 计算9种等价模式，等价模式编号也是从1开始：1~9
int LBP::ComputeValue9(int value58)
{
	int value9 = 0;
	switch (value58)
	{
	case 1:
		value9 = 1;
		break;
	case 2:
		value9 = 2;
		break;
	case 4:
		value9 = 3;
		break;
	case 7:
		value9 = 4;
		break;
	case 11:
		value9 = 5;
		break;
	case 16:
		value9 = 6;
		break;
	case 22:
		value9 = 7;
		break;
	case 29:
		value9 = 8;
		break;
	case 58:
		value9 = 9;
		break;
	}

	return value9;

}

int LBP::GetMinBinaryByInterger(int binary)
{
	static const int miniBinaryLUT[256] = { 0, 1, 1, 3, 1, 5, 3, 7, 1, 9, 5, 11, 3, 13, 7, 15, 1, 17, 9, 19, 5,
		21, 11, 23, 3, 25, 13, 27, 7, 29, 15, 31, 1, 9, 17, 25, 9, 37, 19, 39, 5, 37, 21, 43, 11, 45,
		23, 47, 3, 19, 25, 51, 13, 53, 27, 55, 7, 39, 29, 59, 15, 61, 31, 63, 1, 5, 9, 13, 17, 21, 25,
		29, 9, 37, 37, 45, 19, 53, 39, 61, 5, 21, 37, 53, 21, 85, 43, 87, 11, 43, 45, 91, 23, 87, 47, 95,
		3, 11, 19, 27, 25, 43, 51, 59, 13, 45, 53, 91, 27, 91, 55, 111, 7, 23, 39, 55, 29, 87, 59, 119, 15,
		47, 61, 111, 31, 95, 63, 127, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 9, 25, 37,
		39, 37, 43, 45, 47, 19, 51, 53, 55, 39, 59, 61, 63, 5, 13, 21, 29, 37, 45, 53, 61, 21, 53, 85,
		87, 43, 91, 87, 95, 11, 27, 43, 59, 45, 91, 91, 111, 23, 55, 87, 119, 47, 111, 95, 127, 3,
		7, 11, 15, 19, 23, 27, 31, 25, 39, 43, 47, 51, 55, 59, 63, 13, 29, 45, 61, 53, 87, 91, 95, 27, 59,
		91, 111, 55, 119, 111, 127, 7, 15, 23, 31, 39, 47, 55, 63, 29, 61, 87, 95, 59, 111, 119, 127, 15, 31, 47, 63,
		61, 95, 111, 127, 31, 63, 95, 127, 63, 127, 127, 255 };

	return miniBinaryLUT[binary];
}

// 获取循环二进制的最小二进制模式
uchar LBP::GetMinBinary(uchar *binary)
{
	// 计算8个二进制
	uchar LBPValue[8] = { 0 };
	for (int i = 0; i <= 7; ++i)
	{
		LBPValue[0] += binary[i] << (7 - i);
		LBPValue[1] += binary[(i + 7) % 8] << (7 - i);
		LBPValue[2] += binary[(i + 6) % 8] << (7 - i);
		LBPValue[3] += binary[(i + 5) % 8] << (7 - i);
		LBPValue[4] += binary[(i + 4) % 8] << (7 - i);
		LBPValue[5] += binary[(i + 3) % 8] << (7 - i);
		LBPValue[6] += binary[(i + 2) % 8] << (7 - i);
		LBPValue[7] += binary[(i + 1) % 8] << (7 - i);
	}

	// 选择最小的
	uchar minValue = LBPValue[0];
	for (int i = 1; i <= 7; ++i)
	{
		if (LBPValue[i] < minValue)
		{
			minValue = LBPValue[i];
		}
	}

	return minValue;

}
// cellSize:每个cell的大小,如16*16
void LBP::ComputeLBPFeatureVector_Rotation_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector)
{
	// 参数检查，内存分配


	Mat LBPImage;
	ComputeLBPImage_Rotation_Uniform(srcImage, LBPImage);

	// 计算cell个数
	int widthOfCell = cellSize.width;
	int heightOfCell = cellSize.height;
	int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
	int numberOfCell_Y = srcImage.rows / heightOfCell;

	// 特征向量的个数
	int numberOfDimension = 9 * numberOfCell_X*numberOfCell_Y;
	featureVector.create(1, numberOfDimension, CV_32FC1);
	featureVector.setTo(Scalar(0));

	// 计算LBP特征向量
	int stepOfCell = srcImage.cols;
	int index = -9;// cell的特征向量在最终特征向量中的起始位置
	float *dataOfFeatureVector = (float *)featureVector.data;
	for (int y = 0; y <= numberOfCell_Y - 1; ++y)
	{
		for (int x = 0; x <= numberOfCell_X - 1; ++x)
		{
			index += 9;

			// 计算每个cell的LBP直方图
			Mat cell = LBPImage(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			uchar *rowOfCell = cell.data;
			int sum = 0; // 每个cell的等价模式总数
			for (int y_Cell = 0; y_Cell <= cell.rows - 1; ++y_Cell, rowOfCell += stepOfCell)
			{
				uchar *colOfCell = rowOfCell;
				for (int x_Cell = 0; x_Cell <= cell.cols - 1; ++x_Cell, ++colOfCell)
				{
					if (colOfCell[0] != 0)
					{
						// 在直方图中转化为0~8，所以是colOfCell[0] - 1
						++dataOfFeatureVector[index + colOfCell[0] - 1];
						++sum;
					}
				}
			}

			// 直方图归一化
			for (int i = 0; i <= 8; ++i)
				dataOfFeatureVector[index + i] /= sum;

		}
	}
}

void LBP::ComputeLBPImage_Rotation_Uniform(const Mat &srcImage, Mat &LBPImage)
{
	// 参数检查，内存分配

	LBPImage.create(srcImage.size(), srcImage.type());

	// 扩充图像，处理边界情况
	Mat extendedImage;
	copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, BORDER_DEFAULT);

	// 构建LBP 等价模式查找表
	//int table[256];
	//BuildUniformPatternTable(table);

	// 查找表
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

	int heigthOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	int widthOfLBPImage = LBPImage.cols;

	uchar *rowOfExtendedImage = extendedImage.data + widthOfExtendedImage + 1;
	uchar *rowOfLBPImage = LBPImage.data;
	for (int y = 1; y <= heigthOfExtendedImage - 2; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBPImage)
	{
		// 列
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfLBPImage = rowOfLBPImage;
		for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBPImage)
		{
			// 计算LBP值
			int LBPValue = 0;
			if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBPValue += 128;
			if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0])
				LBPValue += 64;
			if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBPValue += 32;
			if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0])
				LBPValue += 16;
			if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBPValue += 8;
			if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0])
				LBPValue += 4;
			if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBPValue += 2;
			if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0])
				LBPValue += 1;

			int minValue = GetMinBinaryByInterger(LBPValue);

			// 计算58种等价模式LBP
			int value58 = table[minValue];

			// 计算9种等价模式
			colOfLBPImage[0] = ComputeValue9(value58);
		}

	}

}

void LBP::ComputeLBPImage_Rotation_Uniform_2(const Mat &srcImage, Mat &LBPImage)
{
	// 参数检查，内存分配

	LBPImage.create(srcImage.size(), srcImage.type());

	// 扩充图像，处理边界情况
	Mat extendedImage;
	copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, BORDER_DEFAULT);

	// 构建LBP 等价模式查找表
	//int table[256];
	//BuildUniformPatternTable(table);

	// 通过查找表
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

	uchar binary[8] = { 0 };// 记录每个像素的LBP值
	int heigthOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	int widthOfLBPImage = LBPImage.cols;

	uchar *rowOfExtendedImage = extendedImage.data + widthOfExtendedImage + 1;
	uchar *rowOfLBPImage = LBPImage.data;
	for (int y = 1; y <= heigthOfExtendedImage - 2; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBPImage)
	{
		// 列
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfLBPImage = rowOfLBPImage;
		for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBPImage)
		{
			// 计算旋转不变LBP(最小的二进制模式)
			binary[0] = colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0] ? 1 : 0;
			binary[1] = colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0] ? 1 : 0;
			binary[2] = colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0] ? 1 : 0;
			binary[3] = colOfExtendedImage[0 + 1] >= colOfExtendedImage[0] ? 1 : 0;
			binary[4] = colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0] ? 1 : 0;
			binary[5] = colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0] ? 1 : 0;
			binary[6] = colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0] ? 1 : 0;
			binary[7] = colOfExtendedImage[0 - 1] >= colOfExtendedImage[0] ? 1 : 0;

			int minValue = GetMinBinary(binary);

			// 计算58种等价模式LBP
			int value58 = table[minValue];

			// 计算9种等价模式
			colOfLBPImage[0] = ComputeValue9(value58);
		}

	}
}

//计算质心
float GetCenterOfMass(Mat m)
{
	float m_00 = 0, m_01 = 0, m_10 = 0;

	for (int x = 0; x < m.rows; x++)
	{
		for (int y = 0; y < m.cols; y++)
		{
			m_00 += m.at<uchar>(x, y);
			m_01 += (float)y * m.at<uchar>(x, y);
			m_10 += (float)x * m.at<uchar>(x, y);
		}
	}

	float x_c = m_10 / m_00;
	float y_c = m_01 / m_00;

	return fastAtan2(m_01, m_10);
}

// 验证灰度不变+旋转不变+等价模式种类
void LBP::Test()
{
	uchar LBPValue[8] = { 0 };
	int k = 7, j;
	int temp;
	LBP lbp;
	int number[256] = { 0 };
	int numberOfMinBinary = 0;

	// 旋转不变
	for (int i = 0; i < 256; ++i)
	{
		k = 7;
		temp = i;
		while (k >= 0)
		{
			LBPValue[k] = temp & 1;
			temp = temp >> 1;
			--k;
		}
		int minBinary = lbp.GetMinBinary(LBPValue);

		// 查找有无重复的
		for (j = 0; j <= numberOfMinBinary - 1; ++j)
		{
			if (number[j] == minBinary)
				break;
		}
		if (j == numberOfMinBinary)
		{
			number[numberOfMinBinary++] = minBinary;
		}
	}
	cout << "旋转不变一共有：" << numberOfMinBinary << "种" << endl;

	// LUT
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

	for (int i = 0; i <= numberOfMinBinary - 1; ++i)
	{
		cout << "旋转不变的LBP：" << number[i] << " " << "对应的等价模式：" << table[number[i]] << endl;
	}

}

void LBP::TestGetMinBinaryLUT()
{
	for (int i = 0; i <= 255; ++i)
	{
		uchar a[8] = { 0 };
		int k = 7;
		int j = i;
		while (j)
		{
			// 除2取余
			a[k] = j % 2;
			j /= 2;
			--k;
		}
		uchar minBinary = GetMinBinary(a);
		printf("%d,", minBinary);

	}
}



// cellSize:每个cell的大小,如16*16
void LBP::ComputeCLBPFeatureVector(const Mat &srcImage, Size cellSize, Mat &featureVector)
{
	// 参数检查，内存分配
	Mat LBPImage_C, LBPImage_S, LBPImage_M;
	ComputeCLBPImage(srcImage, LBPImage_C, LBPImage_S, LBPImage_M); //2 + 9 + 9

																	// 计算cell个数
	int widthOfCell = cellSize.width;
	int heightOfCell = cellSize.height;
	int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
	int numberOfCell_Y = srcImage.rows / heightOfCell;

	// 特征向量的个数
	int numberOfDimension = 20 * numberOfCell_X*numberOfCell_Y;
	featureVector.create(1, numberOfDimension, CV_32FC1);
	featureVector.setTo(Scalar(0));

	// 计算LBP特征向量
	int stepOfCell = srcImage.cols;
	int index = -20;// cell的特征向量在最终特征向量中的起始位置
	float *dataOfFeatureVector = (float *)featureVector.data;
	for (int y = 0; y <= numberOfCell_Y - 1; ++y)
	{
		for (int x = 0; x <= numberOfCell_X - 1; ++x)
		{
			index += 20;

			// 计算每个cell的LBP直方图
			Mat cell_c = LBPImage_C(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_s = LBPImage_S(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_m = LBPImage_M(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));

			uchar *rowOfCell_C = cell_c.data;
			uchar *rowOfCell_S = cell_s.data;
			uchar *rowOfCell_M = cell_m.data;

			//int sum = 0; // 每个cell的等价模式总数
			for (int y_Cell = 0; y_Cell <= cell_c.rows - 1; ++y_Cell,
				rowOfCell_C += stepOfCell, rowOfCell_S += stepOfCell, rowOfCell_M += stepOfCell)
			{
				uchar *colOfCell_C = rowOfCell_C;
				uchar *colOfCell_S = rowOfCell_S;
				uchar *colOfCell_M = rowOfCell_M;
				for (int x_Cell = 0; x_Cell <= cell_c.cols - 1; ++x_Cell, ++colOfCell_C, ++colOfCell_S, ++colOfCell_M)
				{
					//串联LBP_C
					++dataOfFeatureVector[index + colOfCell_C[0]];

					//串联LBP_S
					++dataOfFeatureVector[index + 2 + colOfCell_S[0] - 1];

					//串联LBP_M
					++dataOfFeatureVector[index + 2 + 9 + colOfCell_M[0] - 1];

				}
			}

			// 一定要归一化！否则分类器计算误差很大
			//for (int i = 0; i <= 57; ++i)
			//	dataOfFeatureVector[index + i] /= sum;

		}
	}
}

void LBP::ComputeCLBPImage(const Mat& src, Mat &LBP_C, Mat &LBP_S, Mat &LBP_M)
{
	//计算LBP_S,LBP_M
	LBP_S.create(src.size(), src.type());
	LBP_M.create(src.size(), src.type());
	// 计算LBP图
	// 扩充原图像边界，便于边界处理
	Mat extendedImage;
	copyMakeBorder(src, extendedImage, 1, 1, 1, 1, BORDER_DEFAULT);

	// LUT(256种每一种模式对应的等价模式)
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

	// 计算LBP
	int heightOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	int widthOfLBP = LBP_S.cols;
	uchar *rowOfExtendedImage = extendedImage.data + widthOfExtendedImage + 1;
	uchar *rowOfLBP_SImage = LBP_S.data;
	uchar *rowOfLBP_MImage = LBP_M.data;

	int sumpixel = 0;

	//计算图像像素均值
	uchar *rowSrc = src.data;
	for (int y = 1; y <= src.rows; ++y, rowSrc += src.cols)
	{
		uchar *colSrc = rowSrc;
		for (int x = 1; x <= src.cols; ++x, ++colSrc)
		{
			sumpixel += colSrc[0];
		}
	}

	int averagePixel = sumpixel / (src.cols*src.rows);

	for (int y = 1; y <= heightOfExtendedImage - 2; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBP_SImage += widthOfLBP, rowOfLBP_MImage += widthOfLBP)
	{
		// 列
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfLBP_SImage = rowOfLBP_SImage;
		uchar *colOfLBP_MImage = rowOfLBP_MImage;
		for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBP_SImage, ++colOfLBP_MImage)
		{
			int SumOfP = abs(colOfExtendedImage[0 - widthOfExtendedImage - 1] - colOfExtendedImage[0]) +
				abs(colOfExtendedImage[0 - widthOfExtendedImage] - colOfExtendedImage[0]) +
				abs(colOfExtendedImage[0 - widthOfExtendedImage + 1] - colOfExtendedImage[0]) +
				abs(colOfExtendedImage[0 + widthOfExtendedImage - 1] - colOfExtendedImage[0]) +
				abs(colOfExtendedImage[0 + widthOfExtendedImage] - colOfExtendedImage[0]) +
				abs(colOfExtendedImage[0 + widthOfExtendedImage + 1] - colOfExtendedImage[0]) +
				abs(colOfExtendedImage[0 - 1] - colOfExtendedImage[0]) +
				abs(colOfExtendedImage[0 + 1] - colOfExtendedImage[0]);

			int threadvalue = SumOfP / 8;;

			int LBP_MValue = 0;
			if (abs(colOfExtendedImage[0 - widthOfExtendedImage - 1] - colOfExtendedImage[0]) >= threadvalue)
				LBP_MValue += 128;
			if (abs(colOfExtendedImage[0 - widthOfExtendedImage] - colOfExtendedImage[0]) >= threadvalue)
				LBP_MValue += 64;
			if (abs(colOfExtendedImage[0 - widthOfExtendedImage + 1] - colOfExtendedImage[0]) >= threadvalue)
				LBP_MValue += 32;
			if (abs(colOfExtendedImage[0 + 1] - colOfExtendedImage[0]) >= threadvalue)
				LBP_MValue += 16;
			if (abs(colOfExtendedImage[0 + widthOfExtendedImage + 1] - colOfExtendedImage[0]) >= threadvalue)
				LBP_MValue += 8;
			if (abs(colOfExtendedImage[0 + widthOfExtendedImage] - colOfExtendedImage[0]) >= threadvalue)
				LBP_MValue += 4;
			if (abs(colOfExtendedImage[0 + widthOfExtendedImage - 1] - colOfExtendedImage[0]) >= threadvalue)
				LBP_MValue += 2;
			if (abs(colOfExtendedImage[0 - 1] - colOfExtendedImage[0]) >= threadvalue)
				LBP_MValue += 1;

			int minValue_M = GetMinBinaryByInterger(LBP_MValue);
			// 计算58种等价模式LBP
			int value58_M = table[minValue_M];
			// 计算9种等价模式
			colOfLBP_MImage[0] = ComputeValue9(value58_M);
			//colOfLBP_MImage[0] = table[LBP_MValue];



			// 计算LBP值
			int LBP_SValue = 0;
			if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBP_SValue += 128;
			if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0])
				LBP_SValue += 64;
			if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBP_SValue += 32;
			if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0])
				LBP_SValue += 16;
			if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBP_SValue += 8;
			if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0])
				LBP_SValue += 4;
			if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBP_SValue += 2;
			if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0])
				LBP_SValue += 1;

			//colOfLBP_SImage[0] = table[LBP_SValue];

			int minValue_S = GetMinBinaryByInterger(LBP_SValue);
			// 计算58种等价模式LBP
			int value58_S = table[minValue_S];
			// 计算9种等价模式
			colOfLBP_SImage[0] = ComputeValue9(value58_S);

		} // x

	}// y

	 //计算LBP_C
	LBP_C.create(src.size(), src.type());

	uchar *rowOfSrcImage = src.data;
	uchar *rowOfLBP_CImage = LBP_C.data;

	for (int y = 1; y <= src.rows; ++y, rowOfSrcImage += widthOfLBP, rowOfLBP_CImage += widthOfLBP)
	{
		uchar *colOfSrcImage = rowOfSrcImage;
		uchar *colOfLBP_CImage = rowOfLBP_CImage;

		for (int x = 1; x <= src.cols; ++x, ++colOfSrcImage, ++colOfLBP_CImage)
		{
			if (colOfSrcImage[0] >= averagePixel)
				colOfLBP_CImage[0] = 1;
			else
				colOfLBP_CImage[0] = 0;
		}
	}

}



// cellSize:每个cell的大小,如16*16
void LBP::ComputeELBPFeatureVector(const Mat &srcImage, Size cellSize, Mat &featureVector)
{
	// 参数检查，内存分配
	Mat LBPImage_NI, LBPImage_CI, LBPImage_RD;
	int r = 2;
	int R = 3;
	ComputeELBPImage(srcImage, LBPImage_NI, LBPImage_CI, LBPImage_RD, r, R);

	// 计算cell个数
	int widthOfCell = cellSize.width;
	int heightOfCell = cellSize.height;
	int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
	int numberOfCell_Y = srcImage.rows / heightOfCell;

	// 特征向量的个数
	int numberOfDimension = 118 * numberOfCell_X*numberOfCell_Y;//2+58+58
	featureVector.create(1, numberOfDimension, CV_32FC1);
	featureVector.setTo(Scalar(0));

	// 计算LBP特征向量
	int stepOfCell = srcImage.cols;
	int index = -118;// cell的特征向量在最终特征向量中的起始位置
	float *dataOfFeatureVector = (float *)featureVector.data;
	for (int y = 0; y <= numberOfCell_Y - 1; ++y)
	{
		for (int x = 0; x <= numberOfCell_X - 1; ++x)
		{
			index += 118;
			// 计算每个cell的LBP直方图
			Mat cell_ci = LBPImage_CI(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_ni = LBPImage_NI(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_rd = LBPImage_RD(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			uchar *rowOfCell_CI = cell_ci.data;
			uchar *rowOfCell_NI = cell_ni.data;
			uchar *rowOfCell_RD = cell_rd.data;
			for (int y_Cell = 0; y_Cell <= cell_ci.rows - 1; ++y_Cell,
				rowOfCell_CI += stepOfCell, rowOfCell_NI += stepOfCell, rowOfCell_RD += stepOfCell)
			{
				uchar *colOfCell_CI = rowOfCell_CI;
				uchar *colOfCell_NI = rowOfCell_NI;
				uchar *colOfCell_RD = rowOfCell_RD;
				for (int x_Cell = 0; x_Cell <= cell_ci.cols - 1; ++x_Cell, ++colOfCell_CI, ++colOfCell_NI, ++colOfCell_RD)
				{
					//串联LBP_CI
					++dataOfFeatureVector[index + colOfCell_CI[0]];

					//串联LBP_NI
					++dataOfFeatureVector[index + 2 + colOfCell_NI[0] - 1];

					//串联LBP_RD
					++dataOfFeatureVector[index + 2 + 58 + colOfCell_RD[0] - 1];
				}
			}

		}
	}
}

void LBP::ComputeELBPImage(const Mat& src, Mat &NI_LBP, Mat &CI_LBP, Mat &RD_LBP, int r, int R)
{
	//计算CI_LBP
	CI_LBP.create(src.size(), src.type());
	NI_LBP.create(src.size(), src.type());
	RD_LBP.create(src.size(), src.type());
	int sumpixel = 0;
	//计算图像像素均值
	uchar *rowSrc = src.data;
	for (int y = 1; y <= src.rows; ++y, rowSrc += src.cols)
	{
		uchar *colSrc = rowSrc;
		for (int x = 1; x <= src.cols; ++x, ++colSrc)
		{
			sumpixel += colSrc[0];
		}
	}
	int averagePixel = sumpixel / (src.cols*src.rows);

	int widthOfLBP = src.cols;

	uchar *rowOfSrcImage = src.data;
	uchar *rowOfCI_LBPImage = CI_LBP.data;


	for (int y = 1; y <= src.rows; ++y, rowOfSrcImage += widthOfLBP, rowOfCI_LBPImage += widthOfLBP)
	{
		uchar *colOfSrcImage = rowOfSrcImage;
		uchar *colOfCI_LBPImage = rowOfCI_LBPImage;

		for (int x = 1; x <= src.cols; ++x, ++colOfSrcImage, ++colOfCI_LBPImage)
		{
			if (colOfSrcImage[0] >= averagePixel)
				colOfCI_LBPImage[0] = 1;
			else
				colOfCI_LBPImage[0] = 0;
		}
	}


	//计算NI_LBP
	Mat extendedImage;
	copyMakeBorder(src, extendedImage, R, R, R, R, BORDER_DEFAULT);

	// LUT(256种每一种模式对应的等价模式)
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

	// 计算LBP
	int heightOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	uchar *rowOfExtendedImage = extendedImage.data + widthOfExtendedImage*R + R;

	uchar *rowOfNI_LBPImage = NI_LBP.data;
	uchar *rowOfRD_LBPImage = RD_LBP.data;

	for (int y = R; y < heightOfExtendedImage - R; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfNI_LBPImage += widthOfLBP, rowOfRD_LBPImage += widthOfLBP)
	{
		// 列
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfNI_LBPImage = rowOfNI_LBPImage;
		uchar *colOfRD_LBPImage = rowOfRD_LBPImage;
		for (int x = R; x < widthOfExtendedImage - R; ++x, ++colOfExtendedImage, ++colOfNI_LBPImage, ++colOfRD_LBPImage)
		{
			vector<float> pixels_r;
			vector<float> pixels_R;

			int tmp_sum = 0;
			int center = colOfExtendedImage[0];

			for (int n = 0; n<8; n++)
			{
				// 采样点的计算  
				float x2 = static_cast<float>(-(R - 1) / 2 * sin(2.0*CV_PI*n / static_cast<float>(8)));
				float y2 = static_cast<float>((R - 1) / 2 * cos(2.0*CV_PI*n / static_cast<float>(8)));
				// 上取整和下取整的值  
				int fx2 = static_cast<int>(floor(x2));
				int fy2 = static_cast<int>(floor(y2));
				int cx2 = static_cast<int>(ceil(x2));
				int cy2 = static_cast<int>(ceil(y2));
				// 小数部分  
				float ty2 = y2 - fy2;
				float tx2 = x2 - fx2;
				// 设置插值权重  
				float w12 = (1 - tx2) * (1 - ty2);
				float w22 = tx2  * (1 - ty2);
				float w32 = (1 - tx2) *      ty2;
				float w42 = tx2  *      ty2;

				// 采样点的计算  
				float x = static_cast<float>(-r / 2 * sin(2.0*CV_PI*n / static_cast<float>(8)));
				float y = static_cast<float>(r / 2 * cos(2.0*CV_PI*n / static_cast<float>(8)));
				// 上取整和下取整的值  
				int fx = static_cast<int>(floor(x));
				int fy = static_cast<int>(floor(y));
				int cx = static_cast<int>(ceil(x));
				int cy = static_cast<int>(ceil(y));
				// 小数部分  
				float ty = y - fy;
				float tx = x - fx;
				// 设置插值权重  
				float w1 = (1 - tx) * (1 - ty);
				float w2 = tx  * (1 - ty);
				float w3 = (1 - tx) *      ty;
				float w4 = tx  *      ty;

				float tmp_R = static_cast<float>(w12*colOfExtendedImage[widthOfExtendedImage*fy2 + fx2] + w22*colOfExtendedImage[widthOfExtendedImage*fy2 + cx2] + w32*colOfExtendedImage[widthOfExtendedImage*cy2 + fx2] + w42*colOfExtendedImage[widthOfExtendedImage*cy2 + cx2]);
				float tmp_r = static_cast<float>(w1*colOfExtendedImage[widthOfExtendedImage*fy + fx] + w2*colOfExtendedImage[widthOfExtendedImage*fy + cx] + w3*colOfExtendedImage[widthOfExtendedImage*cy + fx] + w4*colOfExtendedImage[widthOfExtendedImage*cy + cx]);

				pixels_R.push_back(tmp_R);
				pixels_r.push_back(tmp_r);

				tmp_sum += tmp_r;

			}

			//计算NI_LBP
			//计算RD_LBP
			float tmp_average = tmp_sum / 8;
			int NI_LBPValue = 0;
			int RD_LBPValue = 0;
			for (int i = 0; i < 8; i++)
			{
				if (pixels_r[i] >= tmp_average)
					NI_LBPValue += pow(2, i);

				if (pixels_R[i] >= pixels_r[i])
					RD_LBPValue += pow(2, i);
			}
			colOfNI_LBPImage[0] = table[NI_LBPValue];
			colOfRD_LBPImage[0] = table[RD_LBPValue];

		} // x

	}// y

}




void LBP::ComputeECLBPFeatureVector(const Mat &srcImage, Size cellSize, Mat &featureVector)
{
	Mat LBPImage_C, LBPImage_M, LBPImage_S;
	Mat gaussImage;
	//medianBlur(srcImage, gaussImage, 1);
	//GaussianBlur(srcImage, gaussImage, Size(3, 3), 0);
	ComputeCLBPImage(srcImage, LBPImage_C, LBPImage_S, LBPImage_M);

	Mat LBPImage_EM, LBPImage_ES;
	//medianBlur(srcImage, gaussImage, 3);
	GaussianBlur(srcImage, gaussImage, Size(3, 3), 0);
	ComputeECLBP_SM_Image(gaussImage, LBPImage_ES, LBPImage_EM, 4);

	Mat LBPImage_EM_5, LBPImage_ES_5;
	//medianBlur(srcImage, gaussImage, 5);
	GaussianBlur(srcImage, gaussImage, Size(5, 5), 0);
	ComputeECLBP_SM_Image(gaussImage, LBPImage_ES_5, LBPImage_EM_5, 6);

	Mat LBPImage_EM_7, LBPImage_ES_7;
	//medianBlur(srcImage, gaussImage, 7);
	GaussianBlur(srcImage, gaussImage, Size(7, 7), 0);
	ComputeECLBP_SM_Image(gaussImage, LBPImage_ES_7, LBPImage_EM_7, 8);

	Mat LBPImage_EM_9, LBPImage_ES_9;
	//medianBlur(srcImage, gaussImage, 9);
	GaussianBlur(srcImage, gaussImage, Size(9, 9), 0);
	ComputeECLBP_SM_Image(gaussImage, LBPImage_ES_9, LBPImage_EM_9, 10);

	Mat LBPImage_EM_11, LBPImage_ES_11;
	//medianBlur(srcImage, gaussImage, 11);
	GaussianBlur(srcImage, gaussImage, Size(11, 11), 0);
	ComputeECLBP_SM_Image(gaussImage, LBPImage_ES_11, LBPImage_EM_11, 12);


	//2+9+9+9+9

	// 计算cell个数
	int widthOfCell = cellSize.width;
	int heightOfCell = cellSize.height;
	int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
	int numberOfCell_Y = srcImage.rows / heightOfCell;

	// 特征向量的个数
	int count = 110;//20,38,56,74,92,110
	int numberOfDimension = count * numberOfCell_X*numberOfCell_Y;
	featureVector.create(1, numberOfDimension, CV_32FC1);
	featureVector.setTo(Scalar(0));

	// 计算LBP特征向量
	int stepOfCell = srcImage.cols;
	int index = -count;// cell的特征向量在最终特征向量中的起始位置
	float *dataOfFeatureVector = (float *)featureVector.data;
	for (int y = 0; y <= numberOfCell_Y - 1; ++y)
	{
		for (int x = 0; x <= numberOfCell_X - 1; ++x)
		{
			index += count;

			// 计算每个cell的LBP直方图
			Mat cell_c = LBPImage_C(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_s = LBPImage_S(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_m = LBPImage_M(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_es = LBPImage_ES(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_em = LBPImage_EM(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_es_5 = LBPImage_ES_5(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_em_5 = LBPImage_EM_5(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_es_7 = LBPImage_ES_7(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_em_7 = LBPImage_EM_7(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_es_9 = LBPImage_ES_9(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_em_9 = LBPImage_EM_9(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_es_11 = LBPImage_ES_11(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			Mat cell_em_11 = LBPImage_EM_11(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));

			uchar *rowOfCell_C = cell_c.data;
			uchar *rowOfCell_S = cell_s.data;
			uchar *rowOfCell_M = cell_m.data;
			uchar *rowOfCell_ES = cell_es.data;
			uchar *rowOfCell_EM = cell_em.data;
			uchar *rowOfCell_ES_5 = cell_es_5.data;
			uchar *rowOfCell_EM_5 = cell_em_5.data;
			uchar *rowOfCell_ES_7 = cell_es_7.data;
			uchar *rowOfCell_EM_7 = cell_em_7.data;
			uchar *rowOfCell_ES_9 = cell_es_9.data;
			uchar *rowOfCell_EM_9 = cell_em_9.data;
			uchar *rowOfCell_ES_11 = cell_es_11.data;
			uchar *rowOfCell_EM_11 = cell_em_11.data;

			for (int y_Cell = 0; y_Cell <= cell_c.rows - 1; ++y_Cell,
				rowOfCell_C += stepOfCell, 
				rowOfCell_S += stepOfCell, rowOfCell_M += stepOfCell)
			{
				uchar *colOfCell_C = rowOfCell_C;
				uchar *colOfCell_S = rowOfCell_S;
				uchar *colOfCell_M = rowOfCell_M;
				uchar *colOfCell_ES = rowOfCell_ES;
				uchar *colOfCell_EM = rowOfCell_EM;
				uchar *colOfCell_ES_5 = rowOfCell_ES_5;
				uchar *colOfCell_EM_5 = rowOfCell_EM_5;
				uchar *colOfCell_ES_7 = rowOfCell_ES_7;
				uchar *colOfCell_EM_7 = rowOfCell_EM_7;
				uchar *colOfCell_ES_9 = rowOfCell_ES_9;
				uchar *colOfCell_EM_9 = rowOfCell_EM_9;
				uchar *colOfCell_ES_11 = rowOfCell_ES_11;
				uchar *colOfCell_EM_11 = rowOfCell_EM_11;

				rowOfCell_ES += stepOfCell;
				rowOfCell_EM += stepOfCell;
				rowOfCell_ES_5 += stepOfCell;
				rowOfCell_EM_5 += stepOfCell;
				rowOfCell_ES_7 += stepOfCell;
				rowOfCell_EM_7 += stepOfCell;
				rowOfCell_ES_9 += stepOfCell;
				rowOfCell_EM_9 += stepOfCell;
				rowOfCell_ES_11 += stepOfCell;
				rowOfCell_EM_11 += stepOfCell;

				for (int x_Cell = 0; x_Cell <= cell_c.cols - 1; ++x_Cell, 
					++colOfCell_C, 
					++colOfCell_S, ++colOfCell_M)
				{
					//串联LBP_C
					++dataOfFeatureVector[index + colOfCell_C[0]];

					//串联LBP_S
					++dataOfFeatureVector[index + 2 + colOfCell_S[0] - 1];

					//串联LBP_M
					++dataOfFeatureVector[index + 2 + 9 + colOfCell_M[0] - 1];

					//串联LBP_ES_3
					++dataOfFeatureVector[index + 2 + 9 + 9 + colOfCell_ES[0] - 1];

					//串联LBP_EM_3
					++dataOfFeatureVector[index + 2 + 9 + 9 + 9 + colOfCell_EM[0] - 1];

					//串联LBP_ES_5
					++dataOfFeatureVector[index + 2 + 9 + 9 + 9 + 9 + colOfCell_ES_5[0] - 1];

					//串联LBP_EM_5
					++dataOfFeatureVector[index + 2 + 9 + 9 + 9 + 9 + 9 + colOfCell_EM_5[0] - 1];

					++dataOfFeatureVector[index + 2 + 9 + 9 + 9 + 9 + 9 + 9+ colOfCell_ES_7[0] - 1];

					++dataOfFeatureVector[index + 2 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + colOfCell_EM_7[0] - 1];

					++dataOfFeatureVector[index + 2 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + colOfCell_ES_9[0] - 1];

					++dataOfFeatureVector[index + 2 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + colOfCell_EM_9[0] - 1];

					++dataOfFeatureVector[index + 2 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + colOfCell_ES_11[0] - 1];

					++dataOfFeatureVector[index + 2 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9 + colOfCell_EM_11[0] - 1];

					++colOfCell_ES;
					++colOfCell_EM;
					++colOfCell_ES_5;
					++colOfCell_EM_5;
					++colOfCell_ES_7;
					++colOfCell_EM_7;
					++colOfCell_ES_9;
					++colOfCell_EM_9;
					++colOfCell_ES_11;
					++colOfCell_EM_11;

				}
			}

		}
	}
}

void LBP::ComputeECLBP_C_Image(const Mat& src, Mat &LBP_C)
{
	//计算LBP_C
	LBP_C.create(src.size(), src.type());

	// LUT(256种每一种模式对应的等价模式)
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

	// 计算LBP
	int widthOfLBP = LBP_C.cols;
	int sumpixel = 0;

	//计算图像像素均值
	uchar *rowSrc = src.data;
	for (int y = 1; y <= src.rows; ++y, rowSrc += src.cols)
	{
		uchar *colSrc = rowSrc;
		for (int x = 1; x <= src.cols; ++x, ++colSrc)
		{
			sumpixel += colSrc[0];
		}
	}

	int averagePixel = sumpixel / (src.cols*src.rows);

	uchar *rowOfSrcImage = src.data;
	uchar *rowOfLBP_CImage = LBP_C.data;

	for (int y = 1; y <= src.rows; ++y, rowOfSrcImage += widthOfLBP, rowOfLBP_CImage += widthOfLBP)
	{
		uchar *colOfSrcImage = rowOfSrcImage;
		uchar *colOfLBP_CImage = rowOfLBP_CImage;

		for (int x = 1; x <= src.cols; ++x, ++colOfSrcImage, ++colOfLBP_CImage)
		{
			if (colOfSrcImage[0] >= averagePixel)
				colOfLBP_CImage[0] = 1;
			else
				colOfLBP_CImage[0] = 0;
		}
	}
}

void LBP::ComputeECLBP_SM_Image(const Mat& src, Mat &LBP_S, Mat &LBP_M, int R)
{
	//计算LBP_S,LBP_M
	LBP_S.create(src.size(), src.type());
	LBP_M.create(src.size(), src.type());

	LBP_S.setTo(Scalar(0));
	LBP_M.setTo(Scalar(0));

	// 计算LBP图
	// 扩充原图像边界，便于边界处理
	Mat extendedImage;
	copyMakeBorder(src, extendedImage, R, R, R, R, BORDER_DEFAULT);

	// LUT(256种每一种模式对应的等价模式)
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

	// 计算LBP
	int heightOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	int widthOfLBP = LBP_S.cols;
	uchar *rowOfExtendedImage = extendedImage.data + (widthOfExtendedImage*R) + R;
	uchar *rowOfLBP_SImage = LBP_S.data;
	uchar *rowOfLBP_MImage = LBP_M.data;

	vector<Point> offset;
	for (int k = 0; k < 8; k++)
	{
		//计算偏移量
		float rx = static_cast<float>(R * cos(2.0 * CV_PI * k / 8));
		float ry = -static_cast<float>(R * sin(2.0 * CV_PI * k / 8));

		int x = round(rx);
		int y = round(ry);

		offset.push_back(Point(x, y));
	}

	for (int y = 0; y <= src.rows - 1; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBP_SImage += widthOfLBP, rowOfLBP_MImage += widthOfLBP)
	{
		// 列
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfLBP_SImage = rowOfLBP_SImage;
		uchar *colOfLBP_MImage = rowOfLBP_MImage;
		for (int x = 0; x <= src.cols - 1; ++x, ++colOfExtendedImage, ++colOfLBP_SImage, ++colOfLBP_MImage)
		{
			int SumOfNeiP = 0;
			int SumOfNeiP_M = 0;

			for (int k = 0; k < 8; k++)
			{
				SumOfNeiP += colOfExtendedImage[0 + (widthOfExtendedImage * offset[k].y) + offset[k].x];
				SumOfNeiP_M += abs(colOfExtendedImage[0 + (widthOfExtendedImage * offset[k].y) + offset[k].x] 
					- colOfExtendedImage[0]);
			}

			int threadvalue_S = SumOfNeiP / 8;
			int threadvalue_M = SumOfNeiP_M / 8;

			int LBP_MValue = 0;
			int LBP_SValue = 0;

			for (int k = 0; k < 8; k++)
			{
				if (abs(colOfExtendedImage[0 + (widthOfExtendedImage * offset[k].y + offset[k].x)] - colOfExtendedImage[0]) >= threadvalue_M)
					LBP_MValue += pow(2, k);

				if (colOfExtendedImage[0 + (widthOfExtendedImage * offset[k].y + offset[k].x)] >= threadvalue_S)
					LBP_SValue += pow(2, k);

			}

			int minValue_M = GetMinBinaryByInterger(LBP_MValue);
			// 计算58种等价模式LBP
			int value58_M = table[minValue_M];
			// 计算9种等价模式
			colOfLBP_MImage[0] = ComputeValue9(value58_M);
			//colOfLBP_MImage[0] = table[LBP_MValue];
		
			int minValue_S = GetMinBinaryByInterger(LBP_SValue);
			// 计算58种等价模式LBP
			int value58_S = table[minValue_S];
			// 计算9种等价模式
			colOfLBP_SImage[0] = ComputeValue9(value58_S);
			//colOfLBP_SImage[0] = table[LBP_SValue];

		} // x

	}// y
}

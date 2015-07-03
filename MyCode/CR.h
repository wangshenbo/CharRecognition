#pragma once

#include "opencv2/opencv.hpp"
#include "string"
#include "vector"
using std::string;
using std::vector;
/*****************************************************
********  一种轻量级的验证码识别程序
********  作者：王慎波
********  时间：20150703
******************************************************/

class CCharRecognition
{
public:
	CCharRecognition();
	~CCharRecognition();
	string RecogniteCharImage(IplImage* Src);
	vector<CvRect> GetImageContours(IplImage* Src);
	void GetSamples(IplImage* Src);
private:
	IplConvKernel* m_se;
	CvANN_MLP m_mlp;
	IplImage* m_GraceImg;
	IplImage* m_GraceImg2;
	vector<CvRect> m_vecRect;//连通域
};
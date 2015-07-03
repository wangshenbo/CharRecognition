#pragma once

#include "opencv2/opencv.hpp"
#include "string"
#include "vector"
using std::string;
using std::vector;
/*****************************************************
********  һ������������֤��ʶ�����
********  ���ߣ�������
********  ʱ�䣺20150703
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
	vector<CvRect> m_vecRect;//��ͨ��
};
// CharRecognition2.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "iostream"
#include "CR.h"

//#define TrainML

int _tmain(int argc, _TCHAR* argv[])
{
	CCharRecognition myCC;

#ifdef TrainML
		for(int i=1;i<214;i++)
#else
		for(int i=1; i<=121; i++)	
#endif
	{
		// 产生文件名
		char fn[1024];
#ifdef TrainML
		sprintf(fn, ".//sample//b (%d).jpg", i);
#else
		sprintf(fn, ".//train//T (%d).jpg", i);
		
#endif
		// 读取图片
		IplImage* gray = cvLoadImage(fn);//, CV_LOAD_IMAGE_GRAYSCALE);
		IplImage* gray2 = cvLoadImage(fn);//, CV_LOAD_IMAGE_GRAYSCALE);
		
		cvShowImage("1", gray2);
		cvWaitKey(10);

#ifdef TrainML
		myCC.GetSamples(gray);
#else
		string ss=myCC.RecogniteCharImage(gray);
		std::cout<<ss<<std::endl;
#endif

		char ch=cvWaitKey(0);
		#ifndef TrainML
			static int right=0;
			static int sum=0;
			if(ch=='a')
				right++;
			sum++;
			std::cout<<'\t'<<right<<'\\'<<sum;
		#endif

		std::cout<<std::endl;

		cvReleaseImage(&gray);
		cvReleaseImage(&gray2);
	}
	cvDestroyAllWindows();
	system("pause");
	return 0;
}


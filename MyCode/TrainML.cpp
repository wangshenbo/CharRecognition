// TrainML.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "iostream"
using namespace std;

void print_mat(CvMat& mat)
{
	int count = 0;
	for(int i=0; i<4/*mat.rows*/; i++)
	{
		for(int j=0; j<mat.cols; j++)
		{
			cout<<mat.data.fl[i*(mat.step/4)+j]/*<<" "*/;
		}
		cout<<endl<<endl;
	}
}

int main( int argc, char *argv[] )
{
	// 读入结果responses 特征data
	FILE* f = fopen( "batch", "rb" );
	fseek(f, 0l, SEEK_END);
	long size = ftell(f);
	fseek(f, 0l, SEEK_SET);
	int count = size/4/(36+256);
	CvMat* batch = cvCreateMat( count, 36+256, CV_32F );
	fread(batch->data.fl, size-1, 1, f);
	CvMat outputs, inputs;
	cvGetCols(batch, &outputs, 0, 36);
	cvGetCols(batch, &inputs, 36, 36+256);
	
	// 新建MPL
	CvANN_MLP mlp;
	int layer_sz[] = { 256, 20, 36 };
	CvMat layer_sizes = cvMat( 1, 3, CV_32S, layer_sz );
	mlp.create( &layer_sizes );
	
	// 训练
	//system( "time" );
	mlp.train( &inputs, &outputs, NULL, NULL,
		CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER,300,0.01), CvANN_MLP_TrainParams::RPROP, 0.01)
		);
	//system( "time" );
	
	// 存储MPL
	mlp.save( "mpl.xml" );
	
	// 测试
	int right = 0;
	CvMat* output = cvCreateMat( 1, 36, CV_32F );
	for(int i=0; i<count; i++)
	{
		CvMat input;
		cvGetRow( &inputs, &input, i );
		
		mlp.predict( &input, output );
		CvPoint max_loc = {0,0};
		cvMinMaxLoc( output, NULL, NULL, NULL, &max_loc, NULL );
		int best = max_loc.x;// 识别结果
		
		int ans = -1;// 实际结果
		for(int j=0; j<36; j++)
		{
			if( outputs.data.fl[i*(outputs.step/4)+j] == 1.0f )
			{
				ans = j;
				break;
			}
		}
		cout<<(char)( best<10 ? '0'+best : 'A'+best-10 );
		cout<<(char)( ans<10 ? '0'+ans : 'A'+ans-10 );
		if( best==ans )
		{
			cout<<"+";
			right++;
		}
		//cin.get();
		cout<<endl;
	}
	cvReleaseMat( &output );
	cout<<endl<<right<<"/"<<count<<endl;
	
	cvReleaseMat( &batch );
	
	system( "pause" );
	
	return 0;
}

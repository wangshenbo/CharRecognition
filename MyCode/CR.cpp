
#include "stdafx.h"
#include "CR.h"

//��ȡ��ֵ  ͨ��OSTU�㷨
int cvThresholdOtsu(IplImage* src)
{
	int height=src->height;
	int width=src->width;	

	//histogram
	float histogram[256]={0};
	for(int i=0;i<height;i++) {
		unsigned char* p=(unsigned char*)src->imageData+src->widthStep*i;
		for(int j=0;j<width;j++) {
			histogram[*p++]++;
		}
	}
	//normalize histogram
	int size=height*width;
	for(int i=0;i<256;i++) {
		histogram[i]=histogram[i]/size;
	}

	//average pixel value
	float avgValue=0;
	for(int i=0;i<256;i++) {
		avgValue+=i*histogram[i];
	}

	int threshold;	
	float maxVariance=0;
	float w=0,u=0;
	for(int i=0;i<256;i++) {
		w+=histogram[i];
		u+=i*histogram[i];

		float t=avgValue*w-u;
		float variance=t*t/(w*(1-w));
		if(variance>maxVariance) {
			maxVariance=variance;
			threshold=i;
		}
	}

	return threshold;
}

//������������ıȽϴ�С�ĺ�����ͨ��x������бȽ�
int CompareRect(const void* R1,const void* R2)
{
	if(R1==NULL||R2==NULL)return 0;
	CvRect* p1=(CvRect*)R1;
	CvRect* p2=(CvRect*)R2;
	if(p1->x<p2->x)return -1;
	else if(p1->x==p2->x)return 0;
	else return 1;
}

//ȥ��ͼ��һЩ�������
void ImageDenoise(IplImage* src)
{
	if(src==NULL)return;
	int lastsum=0;
	int index=0;
	for(int i=0;i<src->width;i++)
	{
		int currentsum=0;
		for(int j=0;j<src->height;j++)
		{
			uchar* Data=(uchar*)(src->imageData+j*src->widthStep);
			if(Data[i]==255)
				currentsum++;
		}
		if(currentsum<3)
		{
			if((i-index)>=3)
			{
				for(int j=0;j<src->height;j++)
				{
					uchar* Data=(uchar*)(src->imageData+j*src->widthStep);
					Data[i]=0;
				}
			}
		}
		else
		{
			lastsum=currentsum;
			index=i;
		}
	}
}

//�Ƚ����������ľ����Ƿ�����Ҫ��
static int CompareContour(const void* a, const void* b, void* )
{
	float           dx,dy;
	float           h,w,ht,wt;
	CvPoint2D32f    pa,pb;
	CvRect          ra,rb;

	CvSeq*          pCA = *(CvSeq**)a;//��һ������
	CvSeq*          pCB = *(CvSeq**)b;//�ڶ�������

	ra = ((CvContour*)pCA)->rect;//ת��Ϊ��������
	rb = ((CvContour*)pCB)->rect;

	pa.x = ra.x + ra.width*0.5f;//���ε��е�����
	pa.y = ra.y + ra.height*0.5f;
	pb.x = rb.x + rb.width*0.5f;
	pb.y = rb.y + rb.height*0.5f;

	w = (ra.width+rb.width)*0.5f;//��͵�һ��
	h = (ra.height+rb.height)*0.5f;//�ߺ͵�һ��

	dx = (float)(fabs(pa.x - pb.x)-w);//2������֮��ľ���
	dy = (float)(fabs(pa.y - pb.y)-h);

	//wt = MAX(ra.width,rb.width)*0.1f;
	wt = 0;
	ht = MAX(ra.height,rb.height)*0.3f;
	//2�����������¹�ϵ��dx<0�� 2��������ǰ���ϵ����dy<0
	return (dx < 3 && dy < 3);   
}

bool IsIORJ(const IplImage* src,const CvRect& rect)
{
	if(src==NULL||rect.width>10||rect.height>10)return false;
	uchar lastpix=0;
	uchar nchange=0;
	uchar count=0;
	for(int i=rect.x;i<rect.x+rect.width;i++)
	{
		uchar sum=0;
		for(int j=rect.y;j<src->height;j++)
		{
			uchar* Data=(uchar*)(src->imageData+j*src->widthStep);
			if(Data[i]>=220&&lastpix<=50)
			{
				nchange++;
				sum=0;
			}
			if(Data[i]>220)sum++;
			lastpix=Data[i];
		}
		if(nchange==2||sum>=10)
			count++;
		nchange=0;
		lastpix=0;
	}
	float rate=(count*1.0)/rect.width;
	//printf("rate:%f\n",rate);
	return rate>0.5?true:false;
}


CCharRecognition::CCharRecognition():m_GraceImg(NULL),m_se(NULL)
{
	m_se = cvCreateStructuringElementEx(2, 2, 1, 1, CV_SHAPE_CROSS);
	m_mlp.load( "mpl.xml" );
}

CCharRecognition::~CCharRecognition()
{
	if(m_se!=NULL)cvReleaseStructuringElement(&m_se);
	if(m_GraceImg!=NULL)
	{
		cvReleaseImage(&m_GraceImg);
	}
	if(m_GraceImg2!=NULL)
	{
		cvReleaseImage(&m_GraceImg2);
	}
}

//�������,����Ϊԭͼ��,�����ͨ�����ʾ�����ַ�ͼƬ�������ַ�ͼƬ������Ӧ�����֣��Զ����м�¼�γ�����batch
void CCharRecognition::GetSamples(IplImage* Src)
{
	// ������ͨ��
	GetImageContours(Src);//��ȡ��ͨ��
	for(int j=0;j<m_vecRect.size();j++)
	{
		CvRect rect = m_vecRect[j];
		if( rect.height>10 )// ������Ҫ��10���ظ߶�
		{
			// ���Ƹ���ͨ����character
			cvZero(m_GraceImg);
			IplImage* character = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
			cvZero(character);
			cvSetImageROI(m_GraceImg2,rect);
			cvCopy(m_GraceImg2,character);
			cvResetImageROI(m_GraceImg2);
			//cvDrawContours(character, p, CV_RGB(255, 255, 255), CV_RGB(0, 0, 0), -1, -1, 8, cvPoint(-rect.x, -rect.y));

			// ��һ��
			IplImage* normal = cvCreateImage(cvSize(16, 16), IPL_DEPTH_8U, 1);
			cvResize(character, normal, CV_INTER_AREA);
			cvThreshold(normal, normal, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);// ����

			// ������������
			float input[256];
			for(int i=0; i<256; i++)
				input[i] = (normal->imageData[i]==-1);

			// �û�������
			cvShowImage("2", normal);
			char c = cvWaitKey(0);
			if(c==27)
				return;

			// ����0-9:0-9 a-z:10-35
			unsigned char cc = 255;
			if(c>='A'&&c<='Z')
				cc=c-'A'+10;
			else if(c>='a'&&c<='z')
				cc=c-'a'+10;
			else if(c>='0'&&c<='9')
				cc=c-'0';

			if(cc!=255)
			{
				// ת�����������
				float output[36];
				for(int i=0; i<36; i++)
					output[i] = 0.0f;
				output[cc] = 1.0f;

				// �洢���������ļ�,�γ�������
				FILE* batch = fopen("batch", "wb");
				fwrite(output, 4*36, 1, batch);
				fwrite(input, 4*256, 1, batch);
				fclose(batch);

				static int count = 0;
				std::cout<<count++<<std::endl;
			}

			cvReleaseImage(&character);
			cvReleaseImage(&normal);
			cvRectangle(m_GraceImg2,cvPoint(rect.x,rect.y),cvPoint(rect.x+rect.width,rect.y+rect.height),CV_RGB(0, 0, 255));
		}
	}
}

//ʶ��ͼ����֤�룺�����ͨ��󣬽�ÿ����ͨ��ͼƬ�γ�һ���������������뵽mlp����ʶ�𣬵ó���Ӧ�����֣�����4�������γ�һ����֤��
string CCharRecognition::RecogniteCharImage(IplImage* Src)
{
	string res="";
	GetImageContours(Src);//��ȡ��ͨ��
	// ������ͨ��
	for(int j=0;j<m_vecRect.size();j++)
	{
		CvRect rect = m_vecRect[j];
		if( rect.height>10 )// ������Ҫ��10���ظ߶�
		{
			// ���Ƹ���ͨ����character
			//cvZero(gray);
			IplImage* character = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
			cvZero(character);
			cvSetImageROI(m_GraceImg2,rect);
			cvCopy(m_GraceImg2,character);
			cvResetImageROI(m_GraceImg2);
			//cvDrawContours(character, p, CV_RGB(255, 255, 255), CV_RGB(0, 0, 0), -1, -1, 8, cvPoint(-rect.x, -rect.y));

			// ��һ��
			IplImage* normal = cvCreateImage(cvSize(16, 16), IPL_DEPTH_8U, 1);
			cvResize(character, normal, CV_INTER_AREA);
			cvThreshold(normal, normal, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);// ����

			// ������������
			float input[256];
			for(int i=0; i<256; i++)
				input[i] = (normal->imageData[i]==-1);

			CvMat* output = cvCreateMat( 1, 36, CV_32F );
			CvMat inputMat = cvMat( 1, 256, CV_32F, input);
			m_mlp.predict( &inputMat, output );
			CvPoint max_loc = {0,0};
			cvMinMaxLoc( output, NULL, NULL, NULL, &max_loc, NULL );
			int best = max_loc.x;// ʶ����
			char c = (char)( best<10 ? '0'+best : 'A'+best-10 );
			if(c=='0')
			{
				if(character->height/(character->width*1.0)<1.2)
					c='o';
			}
			res+=c;
			cvReleaseMat( &output );

			cvReleaseImage(&character);
			cvReleaseImage(&normal);
			cvRectangle(m_GraceImg2,cvPoint(rect.x,rect.y),cvPoint(rect.x+rect.width,rect.y+rect.height),CV_RGB(0, 0, 255));
		}
	}
	return res;
}

//��ȡ��ͨ�򣬽�ͼƬ���зָ�γ�4�������ֵ�ͼƬ
vector<CvRect> CCharRecognition::GetImageContours(IplImage* Src)
{
	m_vecRect.clear();
	if(Src==NULL)return m_vecRect;
	if(m_GraceImg==NULL)
	{
		m_GraceImg=cvCreateImage(cvSize(Src->width,Src->height),8,1);
		m_GraceImg2=cvCreateImage(cvSize(Src->width,Src->height),8,1);
	}
	else if(m_GraceImg->height!=Src->height||m_GraceImg->width!=Src->width)
	{
		cvReleaseImage(&m_GraceImg);
		m_GraceImg=cvCreateImage(cvSize(Src->width,Src->height),8,1);

		cvReleaseImage(&m_GraceImg2);
		m_GraceImg2=cvCreateImage(cvSize(Src->width,Src->height),8,1);
	}

	cvCvtColor(Src,m_GraceImg,CV_BGR2GRAY);
//	cvCopy(Src,m_GraceImg);

	int Thread=cvThresholdOtsu(m_GraceImg);
	cvThreshold(m_GraceImg, m_GraceImg, 50, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	// ȥ�߿�
	cvRectangle(m_GraceImg, cvPoint(0, 0), cvPoint(m_GraceImg->width-1, m_GraceImg->height-1), CV_RGB(255, 255, 255));

	/*
	// �����Ƕ�
	cvShowImage("1", gray);
	IplImage* rote = cvCreateImage( cvGetSize(gray), IPL_DEPTH_8U, 1 );
	double t = tan(10.0 / 180.0 * CV_PI);
	int w = gray->width;
	int h = gray->height;
	for(int i = 0; i<h; i++)
	{
	unsigned char* lineGray = (unsigned char*)gray->imageData + gray->widthStep * i;
	unsigned char* lineRote = (unsigned char*)rote->imageData + rote->widthStep * i;
	for(int j = 0; j<w; j++)
	{
	int j2 = j - ((int)(i*t+0.5));
	if (j2<0)
	j2+=w;
	*(lineRote+j) = *(lineGray+j2);
	}
	}
	cvCopy(rote, gray);
	cvReleaseImage(&rote);
	*/

	// ������ͨ��contour
	cvXorS(m_GraceImg, cvScalarAll(255), m_GraceImg, 0);
	// ȥ��
	cvDilate(m_GraceImg, m_GraceImg, m_se);
 	ImageDenoise(m_GraceImg);

	cvCopy(m_GraceImg,m_GraceImg2);


	CvMemStorage* stor = cvCreateMemStorage(0);//�����洢�������е��ڴ�
	CvSeq* cont = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint) , stor);//������������

	CvSeq* cnt_list = cvCreateSeq( 0,sizeof(CvSeq),sizeof(CvSeq*),stor );//�洢����������
	cvFindContours( m_GraceImg, stor, &cont, sizeof(CvContour),CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );	

	for(;cont;cont = cont->h_next)//��������ѹ��cnt_list
	{
		cvSeqPush( cnt_list, &cont);
	}

	CvSeq* clasters = NULL;
	int claster_cur, claster_num;
	int cnt_list_cur;
	claster_num = cvSeqPartition( cnt_list, stor, &clasters, CompareContour, NULL );//�������,claster_numΪ��ֺ��������

	for(cnt_list_cur=0,claster_cur=0; claster_cur<claster_num; ++claster_cur,++cnt_list_cur)
	{				
		int cnt_cur;   
		CvRect rect_res = cvRect(-1,-1,-1,-1);
		for(cnt_cur=0; cnt_cur<clasters->total; ++cnt_cur) //ÿһ����������Ӧ��һ����ţ�clasters->totalΪ������ŵ�����������������������ȵ�
		{
			CvRect  rect;
			CvSeq*  cnt;

			int k = *(int*)cvGetSeqElem(clasters,cnt_cur); //��ȡ���������Ӧ�����ķ����
			if(k!=claster_cur) continue;//�жϷ������뵱ǰѭ���ķ�����Ƿ����?

			cnt = *(CvSeq**)cvGetSeqElem( cnt_list, cnt_cur );//��ȡ����

			rect = ((CvContour*)cnt)->rect;//������ת��Ϊ����

			if(rect.x<0||rect.y<0||rect.width<=0)continue;	

			if(rect.height<=10)
			{
				if(rect.width>10||rect.width<=2)
					continue;
				else if(rect.width>2)
				{
					if(!IsIORJ(m_GraceImg,rect))continue;
				}
			}

			if(rect_res.height<0)
			{
				rect_res = rect;
			}
			else
			{   
				int x0,x1,y0,y1;

				x0 = MIN(rect_res.x,rect.x);
				y0 = MIN(rect_res.y,rect.y);
				x1 = MAX(rect_res.x+rect_res.width,rect.x+rect.width);
				y1 = MAX(rect_res.y+rect_res.height,rect.y+rect.height);

				rect_res.x = x0;
				rect_res.y = y0;
				rect_res.width = x1-x0;
				rect_res.height = y1-y0;
			}
		}
		m_vecRect.push_back(rect_res);
	}

	qsort(&m_vecRect[0],m_vecRect.size(),sizeof(CvRect),CompareRect);//����

	cvReleaseMemStorage(&stor);

	return m_vecRect;
}
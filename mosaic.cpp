#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <math.h>
#include <fstream>

#define RADS 57.2958

using namespace cv;

//eww global
CvSize imgSize = {640,480};

int ptMode = 0;
int outlierMode = 0;

//vector<Mat_<double>(3,1)> augs;
vector<Mat> augs;

void mouseFunc(int evnt,int x,int y,int flags,void* data){
	if(evnt == EVENT_LBUTTONDOWN){
		Mat tmp = (Mat_<double>(3,1) << x,y,-1); 
		augs.push_back(tmp);
	}
}

int main(int argc,char** argv){
	Mat feed;
	Mat prevFeed;
	vector<Point2f> prevFeatures;
	VideoCapture cap;
	FastFeatureDetector feat;


	//Open video
	//if(argc!=2){
	//	std::cout << "Please number of camera to read from\n";
	//	exit(0);
	//} else {
		//cap = VideoCapture(atoi(argv[1]));
		if(argc != 2){
			std::cout << "Video file not provided, using camera 0\n";
			cap = VideoCapture(0);
		} else {
			cap = VideoCapture(argv[1]);
		}

	//}

	if(!cap.isOpened()){
		std::cout << "Video capture failed to open\n";
		exit(1);	
	}

	namedWindow("MainWindow",1);

	setMouseCallback("MainWindow",mouseFunc,NULL);


	feat = FastFeatureDetector();

	Mat transform;
	
	while(true){
		cap >> feed;
		resize(feed,feed,imgSize);
		Mat frame = feed.clone();

		vector<KeyPoint> features;
		vector<Point2f> feats;
		feat.detect(frame,features); 
		for(int i = 0;i < features.size();i++){
			//Vec3b pixel = feed.at<Vec3b>(features[i].pt.y,features[i].pt.x);
			Vec3b pixel = Vec3b(0,255,255);
			if(ptMode == 1){
				frame.at<Vec3b>(features[i].pt.y,features[i].pt.x) = pixel;
				frame.at<Vec3b>(features[i].pt.y+1,features[i].pt.x) = pixel;
				frame.at<Vec3b>(features[i].pt.y-1,features[i].pt.x) = pixel;
				frame.at<Vec3b>(features[i].pt.y,features[i].pt.x+1) = pixel;
				frame.at<Vec3b>(features[i].pt.y,features[i].pt.x-1) = pixel;
			}
			feats.push_back(Point2f(features[i].pt.x,features[i].pt.y));
		}
		if(!prevFeed.empty() && prevFeatures.size() > 0){
			vector<Point2f> validFeatures;
			vector<unsigned char> status;
			vector<float> err;
			//calcOpticalFlowPyrLK(prevFeed,feed,prevFeatures,feats,status,err);
			calcOpticalFlowPyrLK(prevFeed,feed,prevFeatures,validFeatures,status,err);
		
			vector<Point2f> srcPts;
			vector<Point2f> dstPts;
			Mat mask;


			if(status.size() > 0){
				for(int i = 0;i < status.size();i++){
					if(status[i] == 1){
						srcPts.push_back(Point2f(prevFeatures[i].x,prevFeatures[i].y));
						dstPts.push_back(Point2f(validFeatures[i].x,validFeatures[i].y));
						//if(ptMode == 2){
						//	line(frame,prevFeatures[i],validFeatures[i],Scalar(0,255,255));
						//}
					}
				}

				Mat homograph = findHomography(srcPts,dstPts,CV_RANSAC,1,mask);
			
				if(transform.empty()){
					transform = homograph.clone();
				} else {
					transform = transform * homograph;
				}

				for(int i = 0;i < srcPts.size();i++){
					if(ptMode == 2){
						int red = 0;
						int green = 0;
						int blue = 0;
						if(outlierMode == 0){
							green = 255;
							red = 255;
						} else {
							if(mask.at<uchar>(i,1) == 1){
								green = 255;
							} else {
								red = 255;
							}
						}
						line(frame,srcPts[i],dstPts[i],Scalar(blue,green,red));
					}
				}
				
				for(int i = 0;i < augs.size();i++){
					if(augs[i].at<double>(2,0) == -1){
						Mat invertMat;
						invert(transform,invertMat);
						Mat reverse = augs[i].clone();
						reverse.at<double>(2,0) = 1;
						reverse = invertMat * reverse;
						augs[i] = reverse.clone();
					}
					//Mat where = transform * augs[i];
					
					vector<Point2f> pts;
					pts.push_back(Point2f(320,240));
					
					vector<Point2f> out_pts(pts.size());
					Mat out(out_pts);
					perspectiveTransform(Mat(pts),out,transform);
					
					/*for(int i = -4;i <= 4;i++){
						for(int j = -4;j <= 4;j++){
							frame.at<Vec3b>((int)where.at<double>(1,0)+j,(int)where.at<double>(0,0)+i) = Vec3b(0,0,0);
						}
					}*/
					for(int i = -4;i <= 4;i++){
						for(int j = -4;j <= 4;j++){
							frame.at<Vec3b>(out_pts[0].y + j,out_pts[0].x + i) = Vec3b(0,0,0);
						}
					}

				}
			}
		
		}	

		prevFeatures = feats;
		prevFeed = feed.clone(); 
		
		imshow("MainWindow",frame);

		int key = waitKey(10);
		if(key != -1){
			if(key == 102) ptMode = (ptMode + 1) % 3;
			if(key == 's') outlierMode = (outlierMode + 1) % 2;
		}
	}

	return 0;
}

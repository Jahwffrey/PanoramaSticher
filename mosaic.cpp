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

int main(int argc,char** argv){
	Mat feed;
	Mat prevFeed;
	vector<Point2f> prevFeatures;
	VideoCapture cap;
	FastFeatureDetector feat;

	int hasVid = 0;

	//Open video
	if(argc!=2){
		std::cout << "Please number of camera to read from\n";
		exit(0);
	} else {
		cap = VideoCapture(atoi(argv[1]));
	}

	if(!cap.isOpened()){
		std::cout << "Video capture failed to open\n";
		exit(1);	
	}

	for(int i = 0;i < 30;i++){
		std::cout << "MAKE IT ACCEPT A VIDEO FILE FEED!!!!!!!!!\n";
	}


	feat = FastFeatureDetector();

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
		
			for(int i = 0;i < status.size();i++){
				if(status[i] == 1){
					if(ptMode == 2){
						line(frame,prevFeatures[i],validFeatures[i],Scalar(0,255,255));
					}
				}
			}
		}	

		prevFeatures = feats;
		prevFeed = feed.clone(); 
		
		imshow("Whaddap",frame);

		int key = waitKey(10);
		if(key != -1){
			if(key == 102) ptMode = (ptMode + 1) % 3;
		}
	}

	return 0;
}

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

int main(int argc,char** argv){
	Mat feed;
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
		vector<KeyPoint> features;
		feat.detect(feed,features); 

		for(int i = 0;i < features.size();i++){
			//Vec3b pixel = feed.at<Vec3b>(features[i].pt.y,features[i].pt.x);
			Vec3b pixel = Vec3b(0,255,255);
			feed.at<Vec3b>(features[i].pt.y,features[i].pt.x) = pixel;
			feed.at<Vec3b>(features[i].pt.y+1,features[i].pt.x) = pixel;
			feed.at<Vec3b>(features[i].pt.y-1,features[i].pt.x) = pixel;
			feed.at<Vec3b>(features[i].pt.y,features[i].pt.x+1) = pixel;
			feed.at<Vec3b>(features[i].pt.y,features[i].pt.x-1) = pixel;
		}	

		imshow("Whaddap",feed);

	}

	return 0;
}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <math.h>
#include <fstream>

#define RADS 57.2958
#define imW 320//640
#define imH 240//480

using namespace cv;

//eww global
//CvSize imgSize = {640,480};
CvSize imgSize = {imW,imH};
//CvSize bigSize = {3200,1440};
CvSize bigSize = {imW*5,imH*3};

int ptMode = 0;
int outlierMode = 0;
int doCap = 1;

vector<Point2f> prePts;
vector<Point2f> usrPts;
vector<Point2f> panoPts;

void mouseFunc(int evnt,int x,int y,int flags,void* data){
	if(evnt == EVENT_LBUTTONDOWN){
		prePts.push_back(Point2f(x,y));
	}
}

int main(int argc,char** argv){
	Mat feed;
	Mat prevFeed;

	Mat fullImg;
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
			doCap = 0;
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

	bool cont = true;

	while(cap.read(feed) && cont){
		if(doCap == 1){
			resize(feed,feed,imgSize);
			Mat frame = feed.clone();
			Mat dispFrame = feed.clone();
	
			vector<KeyPoint> features;
			vector<Point2f> feats;
			feat.detect(frame,features); 
			for(int i = 0;i < features.size();i++){
				//Vec3b pixel = feed.at<Vec3b>(features[i].pt.y,features[i].pt.x);
				Vec3b pixel = Vec3b(0,255,255);
				if(ptMode == 1){
					dispFrame.at<Vec3b>(features[i].pt.y,features[i].pt.x) = pixel;
					dispFrame.at<Vec3b>(features[i].pt.y+1,features[i].pt.x) = pixel;
					dispFrame.at<Vec3b>(features[i].pt.y-1,features[i].pt.x) = pixel;
					dispFrame.at<Vec3b>(features[i].pt.y,features[i].pt.x+1) = pixel;
					dispFrame.at<Vec3b>(features[i].pt.y,features[i].pt.x-1) = pixel;
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
						transform = homograph * transform;//transform * homograph.clone();
					}
	
					Mat invertMat;
					invert(transform,invertMat);

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
							line(dispFrame,srcPts[i],dstPts[i],Scalar(blue,green,red));
						}
					}
				
					if(prePts.size() > 0){
	
						vector<Point2f> newPts(prePts.size());
						Mat npts(newPts);
						perspectiveTransform(Mat(prePts),npts,invertMat);	
						for(int i = 0;i < newPts.size();i++){
							usrPts.push_back(Point2f(newPts[i].x,newPts[i].y));
							panoPts.push_back(Point2f(newPts[i].x,newPts[i].y));
							prePts.pop_back();
						}
					}

					vector<Point2f> out_pts(usrPts.size());
					Mat out(out_pts);
					perspectiveTransform(Mat(usrPts),out,transform);
					for(int k = 0;k < out_pts.size();k++){
						Point2f pos = out_pts[k];
						if(!(pos.x < 0 || pos.x > imgSize.width)){
							if(!(pos.y < 0 || pos.y > imgSize.height)){
								for(int i = -4;i <= 4;i++){
									for(int j = -4;j <= 4;j++){
										dispFrame.at<Vec3b>(out_pts[k].y + j,out_pts[k].x + i) = Vec3b(255,0,255);
									}
								}
							}
						}
					}
	
					Mat warpFrame;
					//Mat grayImg;
					Mat offset = (Mat_<double>(3,3) << 1,0,bigSize.width/2 - imgSize.width/2,0,1,bigSize.height/2 - imgSize.height/2,0,0,1);
					warpPerspective(frame,warpFrame,offset * invertMat,bigSize);
					//cvtColor(warpFrame,grayImg,CV_BGR2GRAY);
					//threshold(grayImg,grayImg,20,255,THRESH_BINARY);
					if(!fullImg.empty()){
						//fullImg.copyTo(warpFrame,fullImg);
						//addWeighted(fullImg,0.5,warpFrame,0.5,0,fullImg);
						Mat grayImg;
						Mat warpGray;
						cvtColor(warpFrame,warpGray,CV_BGR2GRAY);
						cvtColor(fullImg,grayImg,CV_BGR2GRAY);
						/*for(int i = 0;i < grayImg.rows;i++){
							for(int j = 0;j < grayImg.cols;j++){
								if(warpGray.at<unsigned char>(i,j) > grayImg.at<unsigned char>(i,j)){
									fullImg.at<Vec3b>(i,j) = warpFrame.at<Vec3b>(i,j);
								}
							}
						}*/
	
						compare(warpGray,grayImg,grayImg,CMP_GT);
						warpFrame.copyTo(fullImg,grayImg);

						for(int k = 0;k < panoPts.size();k++){
							Point2f pos = panoPts[k];
							pos.x = pos.x + bigSize.width/2 - imgSize.width/2;
							pos.y = pos.y + bigSize.height/2 - imgSize.height/2;
							if(!(pos.x < 0 || pos.x > bigSize.width)){
								if(!(pos.y < 0 || pos.y > bigSize.height)){
									circle(fullImg,pos,5,Scalar(255,0,255),-1);
									/*for(int i = -4;i <= 4;i++){
										for(int j = -4;j <= 4;j++){
											fullImg.at<Vec3b>(pos.y + j,pos.x + i) = Vec3b(255,0,255);
										}
									}*/
								}
							}
						}
					} else {
						fullImg = warpFrame.clone();
					}
					imshow("Pano",fullImg);
				}
			}	
	
			prevFeatures = feats;
			prevFeed = feed.clone(); 
	
			imshow("MainWindow",dispFrame);
		} else {
			imshow("Feed!",feed);
		}

		int key = waitKey(10);
		if(key != -1){
			if(key == 102) ptMode = (ptMode + 1) % 3;
			if(key == 's') outlierMode = (outlierMode + 1) % 2;
			if(key == 'q'){
				cont = false;
			}
			if(key == 'a'){
				doCap = (doCap + 1) % 2;
			}
		}
	}


	imwrite("mosaic.jpg",fullImg);

	return 0;
}

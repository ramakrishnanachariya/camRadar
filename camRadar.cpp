#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp" 
#include <iostream>
#include <stdio.h>
 
using namespace std;
using namespace cv;
 
double min_face_size=20;
double max_face_size=200;
Mat mask,mask1;
 

//initialize the known distance from the camera to the object
double KNOWN_DISTANCE;

//initialize the known object width
double KNOWN_WIDTH;

//Room grid size(inch)
double GridX,GridY;

// focal length;
double focalLength; 

int flag=0;
int notFoundCount = 0;
bool found = false;
// compute and return the distance from the maker to the camera
Mat mapwin;


 //  Kalman Filter
   int stateSize = 6;
   int measSize = 4;
   int contrSize = 0;
 //Kalman Settings 
   unsigned int type = CV_32F;
   cv::KalmanFilter kf(stateSize, measSize, contrSize, type);
   cv::Mat state(stateSize, 1, type); 
   cv::Mat meas(measSize, 1, type);   
  
double precTick,dT;double ticks = 0;

// Distance calculation 
double distance_to_camera(double knownWidth,double focalLength,double perWidth)
{
	return (knownWidth * focalLength) / perWidth;
}

int Face(Mat image)
{
    // Load Face cascade (.xml file)
    CascadeClassifier face_cascade( "haarcascade_frontalface_alt.xml" );
 
    // Detect faces
    std::vector<Rect> faces;
 
    face_cascade.detectMultiScale( image, faces, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size) );
     
    // Draw circles on the detected faces
   for( int i = 0; i < faces.size(); i++ )
    {   // Lets only track the first face, i.e. face[0]
	rectangle( image, faces[i], Scalar(255,0,0), 2, 8, 0 );
	flag=1;return faces[0].width;
    	 imshow("window1",image);
		waitKey(0);
	}
}

 
Mat detectFace(Mat image,double &distance)
{
    cv::Rect predRect;
 if (found)
      {
         kf.transitionMatrix.at<float>(2) = dT;
         kf.transitionMatrix.at<float>(9) = dT;
        	       
         state = kf.predict();
         predRect.width = state.at<float>(4);
         predRect.height = state.at<float>(5);
         predRect.x = state.at<float>(0) - predRect.width / 2;
         predRect.y = state.at<float>(1) - predRect.height / 2;
 	 rectangle(image, predRect, CV_RGB(255,0,0), 2);
      }



 // Load Face cascade (.xml file)
    CascadeClassifier face_cascade( "haarcascade_frontalface_alt.xml" );
 
    // Detect faces
    std::vector<Rect> faces;
 
    face_cascade.detectMultiScale( image, faces, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size) );
     
  	
//  Kalman Update
      if (faces.size() == 0)
      {
         notFoundCount++;
         cout << "notFoundCount:" << notFoundCount << endl;
         if( notFoundCount >= 100 )
         {
            found = false;
         }
         else
            kf.statePost = state;
      }
      else
      {
         notFoundCount = 0;
 
         meas.at<float>(0) = faces[0].x + faces[0].width / 2;
         meas.at<float>(1) = faces[0].y + faces[0].height / 2;
         meas.at<float>(2) = (float)faces[0].width;
         meas.at<float>(3) = (float)faces[0].height;
 
         if (!found) // First detection!
         {
            // >>>> Initialization
            kf.errorCovPre.at<float>(0) = 1; // px
            kf.errorCovPre.at<float>(7) = 1; // px
            kf.errorCovPre.at<float>(14) = 1;
            kf.errorCovPre.at<float>(21) = 1;
            kf.errorCovPre.at<float>(28) = 1; // px
            kf.errorCovPre.at<float>(35) = 1; // px
 
            state.at<float>(0) = meas.at<float>(0);
            state.at<float>(1) = meas.at<float>(1);
            state.at<float>(2) = 0;
            state.at<float>(3) = 0;
            state.at<float>(4) = meas.at<float>(2);
            state.at<float>(5) = meas.at<float>(3);
            // <<<< Initialization
 
            found = true;
         }
         else
            kf.correct(meas); // Kalman Correction

	}
	Size s=image.size();
	int x=s.width/GridX;
	int y=s.height/GridY;
	 if (notFoundCount>= 2){
	distance=distance_to_camera(KNOWN_WIDTH, focalLength,predRect.width);
	rectangle( mapwin, Rect(predRect.x+(predRect.width/2),y*distance,x,y), Scalar(255,0,255), 8, 8, 0 );
	    }
	else{    

	
	for( int i = 0; i < faces.size(); i++ )
	    {   // Lets only track the first face, i.e. face[0]
        distance=distance_to_camera(KNOWN_WIDTH, focalLength,faces[i].width);
	rectangle( mapwin, Rect(faces[i].x+(faces[i].width/2),y*distance,x,y), Scalar(255,0,255), 8, 8, 0 );
	rectangle( image, faces[i], Scalar(0,255,0), 1, 1, 0 );
	}	

	}
	return image;
	}

	//Mapping 
	Mat drawGrid(Mat &frame,int distance)
	{
	 Size s=frame.size();
	int x=s.width/GridX;
	int y=s.height/GridY;
	for( int i = 1; i < s.width;i=i+x)
	for( int j = 1; j < s.height;j=j+y)
	rectangle( frame,Rect(i,j,x,y), Scalar(25,25,25), 1, 8, 0 );

	return frame;
	}

int main( )
{
    cv::setIdentity(kf.transitionMatrix);
   kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
   kf.measurementMatrix.at<float>(0) = 1.0f;
   kf.measurementMatrix.at<float>(7) = 1.0f;
   kf.measurementMatrix.at<float>(16) = 1.0f;
   kf.measurementMatrix.at<float>(23) = 1.0f;
 
   kf.processNoiseCov.at<float>(0) = 1e-2;
   kf.processNoiseCov.at<float>(7) = 1e-2;
   kf.processNoiseCov.at<float>(14) = 2.0f;
   kf.processNoiseCov.at<float>(21) = 1.0f;
   kf.processNoiseCov.at<float>(28) = 1e-2;
   kf.processNoiseCov.at<float>(35) = 1e-2;
 
   // Measures Noise Covariance Matrix R
   cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
   
	// Camera Index
   int idx = 0;
 
   // Camera Capture
   cv::VideoCapture cap;
 
   // >>>>> Camera Settings
   if (!cap.open(idx))
   {
      cout << "Webcam not connected.\n" << "Please verify\n";
      return EXIT_FAILURE;
   }
   cap.set(CV_CAP_PROP_FRAME_WIDTH, 1024);
   cap.set(CV_CAP_PROP_FRAME_HEIGHT, 768);
   // <<<<< Camera Settings
	
    namedWindow( "window1", 1 );
    namedWindow( "Room Grid Map", 1); 	  
    Mat frame;
	int faceWidth;
	cout<<"\n\t\tcamRadar\n";	
	cout<<"1st Time Intialiize";
	cout<<"\n Please Keep away from camera (exact 1 feet)"<<endl;	
	cout<<"\nGrid(Feet):";
	cin>>GridX;
	GridY=GridX;
	
	
	 while(1)
   	 {
        cap >> frame;        
        faceWidth=Face(frame);
        imshow( "window1", frame );
   	    if (flag==1) break;
        }
	
	
	// create a facing sampling
	//GridX=4;GridY=4;	//48*48 inch =4 feet 
    	KNOWN_DISTANCE=1;	// 12inch=1 feet 


	// Normal Face Width 
	KNOWN_WIDTH=.5;		

	/*cout<<"\nKNOWN_DISTANCE:";
	cin>>KNOWN_DISTANCE;
    	cout<<"\nKNOWN_WIDTH:";
	cin>>KNOWN_WIDTH;*/
	
	//Caluccate Focal lenght with Dected Object width
	focalLength = (faceWidth * KNOWN_DISTANCE) / KNOWN_WIDTH;
	cout<<"Focal Length:"<<focalLength<<endl;
	
	double distance;
    while(1)
    {
	double precTick = ticks;
      	ticks = (double) cv::getTickCount();
 
 	dT = (ticks - precTick) / cv::getTickFrequency();	
	mapwin = cv::Mat::zeros(frame.size(), frame.type());
        Mat frame;
        cap >> frame;        
        frame=detectFace(frame,distance);
	cout<<"Distance :"<<distance<<endl;
	drawGrid(mapwin,distance);
       imshow( "window1", frame );
       imshow( "Room Grid Map", mapwin );

    	// Press 'c' to escape
        if(waitKey(1) == 'c') break;       
    }
 
    waitKey(0);                 
    return 0;
}




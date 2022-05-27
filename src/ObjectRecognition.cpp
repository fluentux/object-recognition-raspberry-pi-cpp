/// <summary> Object recognition application with OpenCV on Raspberry Pi. Detects objects from 
///           video image from Raspberry Pi Camera. Training of the objects is done 
///           by user.</summary>

#include <cmath>
#include <ctime>
#include <iostream>
#include <raspicam/raspicam_cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Point clickPosition;

namespace
{
  /// <summary> An object detected in the image. Detected objects are used in training and 
  ///           classifying the objects recognized in image. </summary>
  struct DetectedObject
  {
    /// <summary> Unique identifier for the object. </summary>
    long id;

    /// <summary> Name for the object. </summary>
    std::string name;

    /// <summary> Contour points. </summary>
    std::vector<cv::Point> contour;

    /// <summary> Object's color (blue, green, red). </summary>
    cv::Scalar color;
  };

  /// <summary> Object recognition application class containing methods for detecting and 
  ///           classifying objects in image. </summary>
  class ObjectRecognition
  {
  public:

    /// <summary> Calculates average BGR color value of the specified contour in image. Average 
    ///           color is calculated by using pixels inside contour area in the image.</summary>
    /// <param name="image">   Image where the pixels inside contour are read. </param>
    /// <param name="contour"> Contour points that define the area where to calculate the average 
    ///                        color. </param>
    /// <returns> Average BGR color inside contour in the image. </returns>
    const cv::Scalar GetAverageColor(const cv::Mat& image, 
      const std::vector<cv::Point>& contour) const
    {
      std::vector<std::vector<cv::Point> > contours;
      contours.push_back(contour);

      // Create mask for the detected object
      cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
      cv::drawContours(mask, contours, CV_FILLED, cv::Scalar(255), CV_FILLED);

      // Erode the mask slightly to reduce pixels from background
      cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);

      // Calculate average color (BGR) of the object
      return cv::mean(image, mask);
    }

    /// <summary> Classifies object by comparing average color to trained object average 
    ///           colors. Classification is done by color difference between object's average color 
    ///           and trained object's average color. Object is classified to be the trained object 
    ///           which has the smallest euclidean difference in average color.</summary>
    /// <param name="color">          Average color of the object to be classified. </param>
    /// <param name="trainedObjects"> List of trained objects which name is known. </param>
    /// <returns> Name of the trained object that the object is classified to be. </returns>
    const std::string ClassifyObject(const cv::Scalar& color, 
      const std::vector<DetectedObject>& trainedObjects) const
    {
      std::string name = "unknown";
      double minDistance = DBL_MAX;

      std::vector<DetectedObject>::const_iterator iter;
      for (iter = trainedObjects.begin(); iter != trainedObjects.end();
        iter++)
      {
        DetectedObject trainedObject = *iter;
        cv::Scalar trainedColor = trainedObject.color;
        double distance = sqrt(
          (trainedColor[0] - color[0]) * (trainedColor[0] - color[0]) +
          (trainedColor[1] - color[1]) * (trainedColor[1] - color[1]) +
          (trainedColor[2] - color[2]) * (trainedColor[2] - color[2]));

        if (distance < minDistance)
        {
          name = trainedObject.name;
          minDistance = distance;
        }
      }

      return name;
    }

    /// <summary> Detects objects in image and tries to classify them. Objects are extracted from 
    ///           background with Otsu's thresholding method. Then their contours are found. Each 
    ///           detected object is classified by average color of the area inside object 
    ///           contour. Classification is done by comparing the average color agains trained 
    ///           object average colors. </summary>
    /// <param name="image">          Image where the objects are detected. </param>
    /// <param name="trainedObjects"> Known objects to compare detected objects against. </param>
    /// <returns> Detected objects in the image. </returns>
    std::vector<DetectedObject> DetectObjects(const cv::Mat& image,
      const std::vector<DetectedObject>& trainedObjects) const
    {
      std::vector<DetectedObject> detectedObjects;

      // Define minimum and maximum size of the objects
      cv::Size image_size = image.size();
      int objectMinimumWidth = 0.05 * image_size.width;
      int objectMinimumHeight = 0.05 * image_size.height;
      int objectMaximumWidth = 0.95 * image_size.width;
      int objectMaximumHeight = 0.95 * image_size.height;

      cv::Mat gray;
      cv::cvtColor(image, gray, CV_BGR2GRAY);

      cv::Mat thresh;
      cv::threshold(gray, thresh, 0, 255,
        cv::THRESH_BINARY + cv::THRESH_OTSU);

      std::vector<std::vector<cv::Point> > contours;
      std::vector<cv::Vec4i> hierarchy;

      cv::findContours(thresh, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

      long id = 1;
      std::vector<std::vector<cv::Point> >::const_iterator iter;
      for (iter = contours.begin(); iter != contours.end(); iter++)
      {
        cv::Rect bounds = cv::boundingRect(*iter);

        if (bounds.width >= objectMinimumWidth &&
          bounds.height >= objectMinimumHeight &&
          bounds.width <= objectMaximumWidth &&
          bounds.height <= objectMaximumHeight)
        {
          cv::Scalar averageColor = this->GetAverageColor(image, *iter);
          DetectedObject detectedObject;
          detectedObject.id = id++;
          detectedObject.name = this->ClassifyObject(averageColor, trainedObjects);
          detectedObject.color = averageColor;
          detectedObject.contour = (*iter);
          detectedObjects.push_back(detectedObject);
        }
      }


      return detectedObjects;
    }

    /// <summary> Draws detected objects on image. Bounding rectangle and name are drawn for each 
    ///           detected object. </summary>
    /// <param name="image">           Image where the detected objects are drawn. </param>
    /// <param name="detectedObjects"> List of detected objects to be drawn. </param>
    /// <param name="clickedObject">   Detected object that is clicked and is drawn with 
    ///                                highlighting. </param>
    void DrawDetectedObjects(cv::Mat& image, const std::vector<DetectedObject>& detectedObjects,
      const DetectedObject& clickedObject) const
    {
      cv::Scalar greenColor = cv::Scalar(0, 255, 0);
      cv::Scalar redColor = cv::Scalar(0, 0, 255);
      cv::Scalar blueColor = cv::Scalar(255, 255, 0);
      cv::Scalar objectColor = greenColor;
      cv::Scalar textColor = blueColor;
      int textOffsetY = 10;
      int lineThickness = 2;

      std::vector<DetectedObject>::const_iterator iter;
      for (iter = detectedObjects.begin(); iter != detectedObjects.end();
        iter++)
      {
        DetectedObject detectedObject = *iter;
        cv::Rect bounds = cv::boundingRect(detectedObject.contour);

        if (detectedObject.id == clickedObject.id)
        {
          objectColor = redColor;
        }
        else
        {
          objectColor = greenColor;
        }

        // Draw bounding rectangle for the detected object
        cv::rectangle(image, bounds, objectColor, lineThickness);

        // Draw name of the detected object
        cv::putText(image, detectedObject.name, cv::Point(bounds.x, bounds.y - textOffsetY),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, cv::LINE_AA);
      }
    }

    /// <summary> Determines which object is clicked if any. Tests if any object's bounding 
    ///           rectangle was hit and returns the object that was closest to click 
    ///           position. </summary>
    /// <param name="detectedObjects"> List of detected objects that are tested for 
    ///                                clicking. </param>
    /// <param name="clickPosition">   Mouse-clicked position. </param>
    /// <returns> The object that was clicked if click location hit any object, otherwise empty 
    ///           object with zero as id. </returns>
    DetectedObject GetClickedObject(const std::vector<DetectedObject>& detectedObjects,
      const cv::Point& clickPosition) const
    {
      DetectedObject emptyObject = DetectedObject();
      emptyObject.id = 0;
      DetectedObject clickedObject = emptyObject;

      if (clickPosition.x != -1 && clickPosition.y != -1)
      {
        int clickedX = clickPosition.x;
        int clickedY = clickPosition.y;
        double minDistance = DBL_MAX;

        // Determine the closest object to the mouse click position
        std::vector<DetectedObject>::const_iterator iter;
        for (iter = detectedObjects.begin(); iter != detectedObjects.end();
          iter++)
        {
          DetectedObject detectedObject = *iter;
          cv::Rect bounds = cv::boundingRect(detectedObject.contour);
          double distance = sqrt(
            (bounds.x - clickedX) * (bounds.x - clickedX) +
            (bounds.y - clickedY) * (bounds.y - clickedY));

          if (distance < minDistance &&
            clickedX >= bounds.x &&
            clickedX <= bounds.x + bounds.width &&
            clickedY >= bounds.y &&
            clickedY <= bounds.y + bounds.height)
          {
            clickedObject = detectedObject;
            minDistance = distance;
          }
        }
      }

      return clickedObject;
    }
  };
}

/// <summary> Handles mouse event click used for training objects. Sets the click position to 
///           mouse-clicked position when user clicks with left mouse button. </summary>
/// <param name="event">      Mouse event. </param>
/// <param name="x">          The x-coordinate of the mouse event. </param>
/// <param name="y">          The y-coordinate of the mouse event.. </param>
/// <param name="flags"> Mouse event flags.. </param>
/// <param name="param"> Optional parameters. </param>
static void ObjectMouseClick(int event, int x, int y, int /*flags*/, void* /*param*/)
{
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    clickPosition = cv::Point(x, y);
  }
}

/// <summary> Sets up the camera and performs object recognition from live video. Each image from 
///           camera is analyzed and objects are recognized. User can train an object by clicking 
///           it on image. </summary>
int main()
{
  // Initialize camera settings
  raspicam::RaspiCam_Cv camera;
  camera.set(CV_CAP_PROP_FORMAT, CV_8UC3);
  camera.set(CV_CAP_PROP_FRAME_WIDTH, 1024);
  camera.set(CV_CAP_PROP_FRAME_HEIGHT, 768);

  clickPosition = cv::Point(-1, -1);
  std::vector<DetectedObject> trainedObjects;
  std::string mainWindowName = "Object recognition";
  cv::Mat image;
  int key = 0;

  cv::namedWindow(mainWindowName, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(mainWindowName, 600, 200);
  cv::setMouseCallback(mainWindowName, ObjectMouseClick, 0);

  ObjectRecognition objectRecognition;
  camera.open();
  
  // Video loop
  while (true)
  {
    camera.grab();
    camera.retrieve(image);

    std::vector<DetectedObject> detectedObjects =
      objectRecognition.DetectObjects(image, trainedObjects);

    DetectedObject clickedObject =
      objectRecognition.GetClickedObject(detectedObjects, clickPosition);

    objectRecognition.DrawDetectedObjects(image, detectedObjects,
      clickedObject);

    cv::imshow(mainWindowName, image);

    key = cv::waitKey(1);

    // Request name for an object if user clicked detected object
    if (clickedObject.id != 0)
    {
      std::string input;
      std::cout << "Type name for the object: ";
      std::getline(std::cin, input);
      if (!input.empty())
      {
        clickedObject.name = input;
        trainedObjects.push_back(clickedObject);
      }
      clickPosition = cv::Point(-1, -1);
    }

    // Check if window ESC is pressed or window is closed
    if (key == 27 || cv::getWindowProperty(mainWindowName, 0) == -1)
    {
      break;
    }
  }

  camera.release();

  cv::destroyAllWindows();
}

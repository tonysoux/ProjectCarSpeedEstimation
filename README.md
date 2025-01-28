# ProjectCarSpeedEstimation
Master Project in electronics course applied to automotive

# 🚗 Highway Video Monitoring with YOLO and Vehicle Tracking  

This program uses the YOLOv8 object detection model and tracking methods to monitor vehicles on a highway from a video feed. It provides functionalities to:  

- Select a Region of Interest (ROI)  
- Track detected vehicles  
- Calculate their speed based on their positions  
- Visualize the results in real-time and on a 2D highway plan  

---

## 📋 Features  

- **ROI Selection:**  
  Draw a polygon on the first frame of the video to select a specific region on the highway for detection.  
- **Object Detection and Tracking:**  
  Track vehicles within the ROI and assign them unique IDs.  
- **Speed Calculation:**  
  Estimate vehicle speeds in km/h based on their positions in the highway plan.  
- **2D Plan Visualization:**  
  Display vehicle positions on a top-down highway plan with marked lanes.  
- **ROI Save/Load:**  
  Save and load the ROI selection for future use.  

---

## 📦 Dependencies  

To run the program, install the following libraries:  

```bash
pip install opencv-python numpy ultralytics matplotlib

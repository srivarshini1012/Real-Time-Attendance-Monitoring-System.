# Real-Time-Attendance-Monitoring-System.

The Face Recognition Attendance System is a web-based application that uses Artificial Intelligence and Computer Vision to automate attendance marking.
It eliminates manual attendance processes by identifying individuals in real-time through a webcam or uploaded images, and storing their attendance details in a CSV file for record-keeping.


This system is built with Flask for the backend, HTML/CSS/JavaScript for the frontend, and OpenCV with KNN classifier for face detection and recognition.
It includes a simple Admin Panel to view, search, and filter attendance records by date, and also provides a Download CSV feature.



--How It Works
1.Upload Faces
-Users can capture face images directly from their webcam or upload an image file.
-Each person’s face is stored in a separate folder inside the dataset.

2.Train Model
-The system processes all stored faces and trains a K-Nearest Neighbors (KNN) model to recognize them.

3.Mark Attendance
-Using the webcam, the system detects and recognizes faces in real-time.
-Attendance is automatically recorded in attendance.csv with Name, Date, and Time.

4.View Attendance
-Admin can view all attendance records in a table.
-Includes search by name and date filter.
-Records can be downloaded in CSV format.




---FILES STRUCTURE

face-recognition-attendance/
  app.py
  haarcascade_frontalface_default.xml
  static/
       style.css
  templates/
        index.html
        upload.html
        attendance.html
        admin.html
        base.html

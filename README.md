# AI Based Online Proctoring System

This project is an attempt to create a simple online examination system with AI-based proctoring features. The goal was to monitor students during an online exam and detect suspicious activities such as looking away from the screen, multiple people appearing in front of the camera, mobile phone usage, and absence from the webcam.

The system automatically records such events and can take actions like issuing warnings or submitting the exam if violations exceed a certain limit.

## Why I Built This

With the increasing use of online examinations, maintaining fairness and preventing cheating has become a challenge. I wanted to explore how computer vision and AI could be used to assist in monitoring candidates during an exam.

This project also helped me learn how to integrate AI models with a full-stack web application.

## Features

### Student Side

* Student login system
* Attempt online exams
* Real-time webcam monitoring during exams
* Automatic result generation
* Auto submission after repeated violations

### Admin Side

* Admin login
* Create and manage exams
* Manage students
* View exam results
* View misconduct logs

### Proctoring Features

* Face detection
* Multiple face detection
* No face detection
* Mobile phone detection using YOLOv8
* Basic head movement monitoring
* Warning generation and logging

## Technologies Used

* Python
* Flask
* SQLite
* SQLAlchemy
* OpenCV
* MediaPipe
* YOLOv8
* HTML
* CSS
* JavaScript

## How It Works

During an exam, the student's webcam feed is processed in real time.

The system checks for:

* Whether a face is visible
* Whether multiple faces are present
* Whether a mobile phone is detected
* Whether the student is frequently looking away

If suspicious activity is detected, a warning is generated and stored in the database. Repeated violations can result in automatic submission of the exam.

## Running the Project

Clone the repository:

```bash
git clone <your-repo-link>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open your browser and visit:

```text
http://localhost:5000
```

## Future Improvements

Some features I would like to add in future versions:

* Eye gaze tracking
* Tab switching detection
* Full screen monitoring
* Better head pose estimation
* PostgreSQL support
* Dashboard analytics
* Cloud deployment

## What I Learned

This project helped me understand:

* Flask application development
* Database integration using SQLAlchemy
* Computer vision with OpenCV
* Using MediaPipe for facial landmark detection
* Integrating YOLO models into web applications
* Handling real-time video streams
* Building complete end-to-end projects

## Note

This project was built mainly for learning and experimentation purposes and is not intended to be a production-ready proctoring solution.

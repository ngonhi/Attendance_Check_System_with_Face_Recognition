# Attendance Check System using Face Recognition
The project can be divided into 2 big modules: face recognition module and attendance check system.
## Face Recognition
* We follow [tuna-date/Face-Recognition-with-InsightFace](https://github.com/tuna-date/Face-Recognition-with-InsightFace)
* For face detection, we use MTCNN, which outputs both bounding boxes for faces and facial landmarks. The facial landmarks are then used for facial alignment. To extract feature embedding, we use pretrained MobileNet trained with ArcFace loss from InsightFace
* For face recognition, we use KNN, SVM and train a SoftMax MLP model to tell the face identity. To train the classification models, please refer to `src/train_knn.py`, `src/train_svm.py`, `src/train_softmax.py`.
* To record faces from stream or videos, please refer to `src/get_faces_from_camera.py` and `src/get_faces_from_video.py`
* To test the recognition model, please refer to `src/recognizer_stream.py`

## Attendance Check System
- We create a simple web app to manage the attendance system. Features include adding new employee, taking attendance and displaying check-in time for each employee.
- After adding a new employee, you have to train the model before it can recognize new faces. 
- Please look at `AttendanceCheckSystem` folder for more instructions on how to run the system.
- The system is written using Flask and HTML, with SQLite database.

Followings are some screenshots of the web app.
1. Log in as admin 
![alt text][login]
2. Admin homepage
![alt text][homepage]
3. Show all employees (press `Show Employee` button on Homepage)
![alt text][show]
4. Insert new employee (press `Insert Employee` button on Homepage)
After pressing `Register` button, the system will start taking employee's faces. The collecting process will stop at 25 photos. Please press `Finish` after it has finished collecting 25 photos.
After inserting or removing employees, admin has to press the `Train model` button on Homepage
![alt text][insert]
5. Admin will be notified when model has finished training
6. Press `Start Checking Attendance` to take attendance
![alt text][attendance]
7. Attendance sheet is displayed on admin homepage

[login]: https://github.com/ngonhi/Attendance_Check_System_with_Face_Recognition/blob/main/images/login.png "Log In Page"
[homepage]: https://github.com/ngonhi/Attendance_Check_System_with_Face_Recognition/blob/main/images/finish_training.png "Admin Homepage"
[show]: https://github.com/ngonhi/Attendance_Check_System_with_Face_Recognition/blob/main/images/list_emp.png "Show All Employees"
[insert]: https://github.com/ngonhi/Attendance_Check_System_with_Face_Recognition/blob/main/images/insert_emp.png "Insert New Employee"
[attendance]: https://github.com/ngonhi/Attendance_Check_System_with_Face_Recognition/blob/main/images/taking_attendance.png "Taking Attendance"


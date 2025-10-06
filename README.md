# 🎓 Facial Attendance System  

A **Facial Recognition-Based Attendance System** built in **Python** that uses a camera to detect and recognize students' faces for attendance marking.  
The system is designed for educational institutions to simplify attendance tracking while improving accuracy and efficiency.  

---

## 🧠 Overview  

This project automates attendance using **facial recognition** powered by the **RetinaFace** library.  
It allows you to:  
- Add and manage students  
- Mark attendance for a single student or entire class  
- Detect unauthorized faces  
- View, update, and manage attendance records  

All features are accessible through a **command-line interface (CLI)** menu system.

---

## 🧩 Features  

✅ **Add Subjects** — Create and organize subjects by semester  
✅ **Add Single/Multiple Students** — Capture student faces using the camera  
✅ **Delete Entire Class or Single Student** — Manage stored student data easily  
✅ **Mark Attendance** — For one student or an entire class  
✅ **View Attendance** — See detailed records  
✅ **Detect Unauthorized Faces** — Identify unknown or unregistered individuals  
✅ **Re-register Faces** — Update student face data if needed  
✅ **Update Subjects** — Modify existing subject information  

---

## 🧱 Project Structure 

FacialAttendance/
│
├── FacialAttendance.py # Handles facial recognition and core attendance logic

├── AttendanceSystem.py # CLI-based main menu for running the system

├── data/ # Stores face encodings, attendance records, and images

├── attendance_records/ # JSON or CSV attendance data

└── README.md # Project documentation

---

## ⚙️ How It Works  

1. **Face Registration:**  
   - The system captures a student's face through the camera.  
   - The facial data is stored and linked with their name, ID, and subject.  

2. **Attendance Marking:**  
   - The system compares live camera input with stored faces using **RetinaFace**.  
   - Matches are recorded as present, while unknown faces are flagged.  

3. **Viewing & Management:**  
   - Attendance records can be viewed, deleted, or updated anytime.  

---

## 🖥️ Main Menu  

When you run the program, you’ll see a menu like this:

1. Add Subjects

2. Add Single Student

3. Add Multiple Students

4. Delete Entire Class

5. Delete Single Student

6. Mark Single Student Attendance

7. Mark Entire Class Attendance

8. View Attendance

9. Detect Unauthorized Faces

10. Re-Register Face

11. Update Subject

0. Exit

---


Each option triggers its corresponding function inside the `FacialAttendance` class.

---

## 🚀 Getting Started  

### 🔧 Prerequisites  
Make sure you have these installed:  
- Python 3.8+  
- OpenCV  
- RetinaFace  
- NumPy  
- Pandas  

## 👨‍💻 Developer

**Abdullah Akram**  
📍 Pakistan  
💻 Android & Web Developer  
📧 [Email](mailto:m.abdullahakram01@gmail.com)  
🔗 [GitHub](#) 

---

⭐ **If you like this app, please give it a star on GitHub!**

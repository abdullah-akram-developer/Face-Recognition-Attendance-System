import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
from datetime import datetime
import json
from retinaface import RetinaFace
import csv



class FacialAttendance:
    def __init__(self, database_path="student_faces.json", attendance_file="attendance.json"):
        self.database_path = database_path
        self.attendance_file = attendance_file
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))  # RetinaFace + ArcFace
        self.student_db = self.load_student_database()
        self.attendance_db = self.load_attendance()
        self.subjects_db = self.load_subjects_db()

        if not os.path.exists("subjects_db.json"):
            self.save_subjects_db()

    def load_subjects_db(self):
        """Load subjects from JSON or return empty dict if file doesn't exist or is empty."""
        subjects_file = "subjects_db.json"
        try:
            if os.path.exists(subjects_file) and os.path.getsize(subjects_file) > 0:
                with open(subjects_file, "r") as f:
                    return json.load(f)
            # Return empty dict if file doesn't exist or is empty
            return {}
        except json.JSONDecodeError:
            print("⚠️ Warning: subjects_db.json is corrupted. Creating new empty database.")
            return {}
    
    def save_subjects_db(self):
        """Save subjects to JSON."""
        with open("subjects_db.json", "w") as f:
            json.dump(self.subjects_db, f, indent=4)

    def load_student_database(self):
        """Load stored student face embeddings."""
        if os.path.exists(self.database_path) and os.path.getsize(self.database_path) > 0:
            with open(self.database_path, "r") as file:
                return json.load(file)
        return {}  # Return an empty dictionary if file is empty or missing
# ok
    def load_attendance(self):
        """Load attendance records from a file or create a new one if missing."""
        if not os.path.exists(self.attendance_file):
            print("📂 Attendance file not found. Creating a new one...")
            with open(self.attendance_file, "w") as file:
                json.dump({}, file)  # Create an empty JSON file

        with open(self.attendance_file, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                print("❌ Error: Attendance file is corrupted. Resetting data.")
                return {}
# ok
    def save_attendance(self):
        """Save updated attendance records."""
        with open(self.attendance_file, "w") as file:
            json.dump(self.attendance_db, file, indent=4)
# ok
    def save_student_database(self):
        """Save the updated student database to a JSON file."""
        with open(self.database_path, "w") as file:
            json.dump(self.student_db, file, indent=4)

    def extract_embeddings(self, image):
        """Extract facial embeddings for all detected faces in a frame."""
        faces = self.app.get(image)  
        embeddings, bboxes = [], []

        for face in faces:
            bbox = face.bbox.astype(int)  # Convert bounding box to integers
            embeddings.append(face.embedding)  # Store face embeddings
            bboxes.append(bbox)  # Store bounding box

        return embeddings, bboxes  # Return all detected faces

    def match_student(self, new_embedding, threshold=0.5):
        """Compare new face embedding with stored ones."""
        new_embedding = np.array(new_embedding)  # Ensure input is a NumPy array
        
        for student_id, data in self.student_db.items():
            stored_embedding = np.array(data["embedding"])  # Convert stored embedding to NumPy array

            if stored_embedding.shape != new_embedding.shape:
                continue  # Skip if shapes don't match

            similarity = np.dot(new_embedding, stored_embedding) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(stored_embedding)
            )

            if similarity > threshold:
                return student_id, data["name"]  

        return None, None  # No match found

    def is_duplicate_face(self, new_embedding, threshold=0.6):
        """Check if the face already exists in the database. Return student ID if found."""
        for student_id, data in self.student_db.items():
            existing_embedding = np.array(data["embedding"])
            similarity = np.dot(new_embedding, existing_embedding) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
            )
            if similarity > threshold:
                return student_id  # Return the student ID if face exists
        return None  # No matching face found

    def capture_image(self):
        """Capture an image from the webcam."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the camera!")
            return None

        print("Press SPACE to capture image or ESC to cancel...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image!")
                break

            cv2.imshow("Capture Face", frame)

            key = cv2.waitKey(1)
            if key == 32:  # Space key to capture
                cv2.imwrite("captured_face.jpg", frame)
                print("Image captured successfully!")
                break
            elif key == 27:  # ESC key to cancel
                print("Capture cancelled.")
                frame = None
                break

        cap.release()
        cv2.destroyAllWindows()
        return frame

    def get_student_details(self):
        """Prompt user to select semester, department, shift, and multiple subjects (max 8)."""
        semesters = [f"Semester {i}" for i in range(1, 9)]
        departments = ["Computer Science", "Information Technology", "Software Engineering", "Electrical Engineering"]
        shifts = ["Morning", "Evening"]

        print("\nSelect Semester:")
        for i, sem in enumerate(semesters, 1):
            print(f"{i}. {sem}")
        semester = semesters[int(input("Enter choice: ")) - 1]

        print("\nSelect Department:")
        for i, dept in enumerate(departments, 1):
            print(f"{i}. {dept}")
        department = departments[int(input("Enter choice: ")) - 1]

        print("\nSelect Shift:")
        for i, shift in enumerate(shifts, 1):
            print(f"{i}. {shift}")
        shift = shifts[int(input("Enter choice: ")) - 1]

        # Allow multiple subjects (maximum 8)
        subjects = []
        print("\nEnter subjects one by one. Type 'done' to finish (maximum 8 subjects allowed):")
        while len(subjects) < 8:  # Restrict to 8 subjects
            subject = input("Enter Subject Name: ").strip()
            if subject.lower() == 'done':
                break
            if subject:
                subjects.append(subject)
            else:
                print("⚠️ Subject name cannot be empty. Please try again.")

        if len(subjects) >= 8:
            print("⚠️ Maximum of 8 subjects reached. You cannot add more subjects.")

        return semester, department, shift, subjects
 
# finalss
    def delete_student_from_face(self):
        """Delete a student by face and remove their attendance records."""
        # Step 1: Capture face
        print("📸 Capturing face for student verification...")
        image = self.capture_image()
        if image is None:
            print("❌ Error: Failed to capture image.")
            return False

        # Step 2: Find matching student
        embeddings, _ = self.extract_embeddings(image)
        if not embeddings:
            print("❌ Error: No face detected!")
            return False

        new_embedding = np.array(embeddings[0])
        student_id, student_name = self.match_student(new_embedding)

        if not student_id:
            print("❌ Error: No matching student found.")
            return False

        # Step 3: Confirm deletion with ID
        print(f"\nFound student: {student_name} (ID: {student_id})")
        confirm_id = input("Enter student ID to confirm deletion: ").strip()
        if confirm_id != student_id:
            print("❌ Error: ID doesn't match. Deletion canceled.")
            return False

        # Step 4: Delete student and attendance
        deleted_student = self.student_db.pop(student_id, None)
        attendance_deleted = False
        
        if student_id in self.attendance_db:
            del self.attendance_db[student_id]
            attendance_deleted = True

        # Step 5: Save changes
        if deleted_student:
            self.save_student_database()
            self.save_attendance()
            print(f"\n✅ Successfully deleted:")
            print(f"👤 Student: {student_name} (ID: {student_id})")
            print(f"📅 Attendance records: {'Deleted' if attendance_deleted else 'None found'}")
            return True
        
        print("❌ Error: Student not found in database.")
        return False

    def delete_entire_semester(self):
        """Delete all students in a semester-department-shift and their attendance."""
        # Step 1: Get semester, department, shift
        print("\nSelect Academic Program to Delete:")
        semester, department, shift = self.get_semester_dept_shift()
        program_key = f"{semester}-{department}-{shift}"

        # Step 2: Find matching students
        students_to_delete = []
        for student_id, student_data in list(self.student_db.items()):
            if (student_data.get("semester") == semester and
                student_data.get("department") == department and
                student_data.get("shift") == shift):
                students_to_delete.append((student_id, student_data))

        if not students_to_delete:
            print(f"\n⚠️ No students found in {program_key}")
            return False

        # Step 3: Confirm deletion
        print(f"\nFound {len(students_to_delete)} students in {program_key}")
        confirm = input("⚠️ Delete ALL these students AND their attendance? (yes/no): ").lower()
        if confirm != 'yes':
            print("❌ Deletion canceled.")
            return False

        # Step 4: Perform deletion
        attendance_removed = 0
        for student_id, student_data in students_to_delete:
            del self.student_db[student_id]
            if student_id in self.attendance_db:
                del self.attendance_db[student_id]
                attendance_removed += 1

        # Step 5: Save changes
        self.save_student_database()
        self.save_attendance()
        
        # Step 6: Show results
        print(f"\n✅ Deletion Complete:")
        print(f"🗑️ Deleted {len(students_to_delete)} students")
        print(f"📅 Removed {attendance_removed} attendance records")
        print("\nDeleted Students:")
        for student_id, student_data in students_to_delete:
            print(f"- {student_data['name']} (ID: {student_id})")
        
        return True 
    
    def is_attendance_marked(self, student_id):
        """Check if attendance is already marked for a student."""
        today = datetime.now().strftime("%Y-%m-%d")
        if "attendance" not in self.student_db[student_id]:
            return False
        return today in self.student_db[student_id]["attendance"]

    def mark_single_attendance_from_face(self):
        """Mark attendance for a single student from face capture, avoiding duplicates and allowing multiple subjects."""
        
        # Initialize InsightFace
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return False

        print("📸 Capturing face for attendance... (Press 'q' to quit)")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame.")
                break

            # Detect faces using RetinaFace
            faces = RetinaFace.detect_faces(frame)
            if not faces:
                cv2.imshow("Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Extract embeddings using InsightFace
            face_results = face_app.get(frame)
            if not face_results:
                cv2.imshow("Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Process each detected face
            for face in face_results:
                bbox = face.bbox.astype(int)  # Get bounding box coordinates
                embedding = face.embedding  # Get face embedding

                # Match the face with the database
                student_id, student_name = self.match_student(embedding)

                if student_id:
                    # Fetch student details
                    student_data = self.student_db[student_id]
                    semester = student_data.get("semester", "N/A")
                    department = student_data.get("department", "N/A")
                    shift = student_data.get("shift", "N/A")
                    enrolled_subjects = student_data.get("subjects", [])

                    if not enrolled_subjects:
                        print(f"⚠️ No subjects enrolled for {student_name}. Please add subjects first.")
                        continue

                    # Get current date and time
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    current_time = datetime.now().strftime("%H:%M:%S")

                    # Check if attendance is already marked for all enrolled subjects
                    all_marked = True
                    for subject in enrolled_subjects:
                        if not (student_id in self.attendance_db and 
                                current_date in self.attendance_db[student_id] and 
                                subject in self.attendance_db[student_id][current_date]):
                            all_marked = False
                            break

                    if all_marked:
                        print(f"⏳ Attendance for all subjects is already marked today for {student_name}.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True  # Exit if all subjects are already marked

                    # Display enrolled subjects and allow selection
                    print(f"\n📖 Enrolled Subjects for {student_name} (ID: {student_id}):")
                    for i, subject in enumerate(enrolled_subjects, 1):
                        print(f"{i}. {subject}")

                    selected_indices = input("\nEnter the numbers of subjects to mark attendance (e.g., 1 2 3) or 'q' to quit: ").strip().split()
                    
                    # Allow the user to quit
                    if 'q' in selected_indices:
                        print("🚪 Exiting attendance marking process.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True

                    selected_subjects = [enrolled_subjects[int(i) - 1] for i in selected_indices if i.isdigit() and 0 < int(i) <= len(enrolled_subjects)]

                    if not selected_subjects:
                        print("❌ Error: No valid subjects selected. Please try again.")
                        continue

                    # Mark attendance for selected subjects
                    attendance_marked = False
                    for subject in selected_subjects:
                        # Check if attendance is already marked for the day
                        if (student_id in self.attendance_db and 
                            current_date in self.attendance_db[student_id] and 
                            subject in self.attendance_db[student_id][current_date]):
                            print(f"⏳ Attendance for {subject} is already marked today for {student_name}.")
                            continue

                        # Mark attendance
                        self.attendance_db.setdefault(student_id, {}).setdefault(current_date, {})[subject] = {
                            "semester": semester,
                            "department": department,
                            "shift": shift,
                            "time": current_time
                        }
                        attendance_marked = True
                        print(f"✅ Attendance marked for {student_name} in {subject} at {current_time}.")

                    # Save attendance if marked
                    if attendance_marked:
                        self.save_attendance()
                        print("\n📄 Attendance Details:")
                        print(f"Student Name: {student_name}")
                        print(f"Student ID: {student_id}")
                        print(f"Semester: {semester}")
                        print(f"Department: {department}")
                        print(f"Shift: {shift}")
                        print(f"Subjects: {', '.join(selected_subjects)}")
                        print(f"Date: {current_date}")
                        print(f"Time: {current_time}")

                        # Draw rectangle and label on the frame
                        color = (0, 255, 0)  # Green
                        label = f"{student_name} (Attendance Marked)"
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        # Display the frame with the marked attendance
                        cv2.imshow("Attendance", frame)
                        cv2.waitKey(2000)  # Wait for 2 seconds to show the result
                        cap.release()
                        cv2.destroyAllWindows()
                        return True  # Exit after marking attendance

                else:
                    # Draw rectangle and label for unknown face
                    color = (0, 0, 255)  # Red
                    label = "Unknown"
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Display the frame
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()
        return True

    def mark_entire_semester_attendance_with_faces(self):
        """Mark attendance for all students in a specific semester and department using real-time face capture."""
        
        # Step 1: Get semester, department, and shift
        print("\nSelect Semester:")
        semesters = [f"Semester {i}" for i in range(1, 9)]
        for i, sem in enumerate(semesters, 1):
            print(f"{i}. {sem}")
        semester_choice = int(input("Enter choice: ")) - 1
        semester = semesters[semester_choice]

        print("\nSelect Department:")
        departments = ["Computer Science", "Information Technology", "Software Engineering", "Electrical Engineering"]
        for i, dept in enumerate(departments, 1):
            print(f"{i}. {dept}")
        department_choice = int(input("Enter choice: ")) - 1
        department = departments[department_choice]

        print("\nSelect Shift:")
        shifts = ["Morning", "Evening"]
        for i, shift in enumerate(shifts, 1):
            print(f"{i}. {shift}")
        shift_choice = int(input("Enter choice: ")) - 1
        shift = shifts[shift_choice]

        # Step 2: Get current date and time
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        # Step 3: Get subject selection for the entire semester
        print("\nSelect Subjects for the Semester:")
        # Collect all unique subjects for the selected semester, department, and shift
        all_subjects = set()
        for student_id, student_data in self.student_db.items():
            if (student_data.get("semester") == semester and
                student_data.get("department") == department and
                student_data.get("shift") == shift):
                all_subjects.update(student_data.get("subjects", []))

        if not all_subjects:
            print("⚠️ No subjects found for the selected semester, department, and shift.")
            return

        # Display all unique subjects
        print("Available Subjects:")
        for i, subject in enumerate(all_subjects, 1):
            print(f"{i}. {subject}")

        # Allow the user to select subjects
        selected_indices = input("\nEnter the numbers of subjects to mark attendance (e.g., 1 2 3): ").strip().split()
        selected_subjects = [list(all_subjects)[int(i) - 1] for i in selected_indices if i.isdigit() and 0 < int(i) <= len(all_subjects)]

        if not selected_subjects:
            print("❌ Error: No valid subjects selected. Please try again.")
            return

        # Step 4: Initialize InsightFace for face detection and recognition
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Step 5: Open camera for real-time face capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return

        print("\n🎥 Starting real-time attendance marking. Press 'q' to quit...")

        # Track marked students
        marked_students = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame.")
                break

            # Detect faces using RetinaFace
            faces = RetinaFace.detect_faces(frame)
            if not faces:
                cv2.imshow("Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Extract embeddings using InsightFace
            face_results = face_app.get(frame)
            if not face_results:
                cv2.imshow("Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Process each detected face
            for face in face_results:
                bbox = face.bbox.astype(int)  # Get bounding box coordinates
                embedding = face.embedding  # Get face embedding

                # Match the face with the database
                student_id, student_name = self.match_student(embedding)

                if student_id:
                    # Fetch student details
                    student_data = self.student_db[student_id]
                    if (student_data.get("semester") == semester and
                        student_data.get("department") == department and
                        student_data.get("shift") == shift):

                        enrolled_subjects = student_data.get("subjects", [])

                        # Check if attendance is already marked for all selected subjects
                        all_marked = True
                        for subject in selected_subjects:
                            if subject not in enrolled_subjects:
                                continue  # Skip if the student is not enrolled in the subject
                            if not (student_id in self.attendance_db and 
                                    current_date in self.attendance_db[student_id] and 
                                    subject in self.attendance_db[student_id][current_date]):
                                all_marked = False
                                break

                        if all_marked:
                            print(f"⏳ Attendance for all selected subjects is already marked today for {student_name}.")
                            marked_students.add(student_id)
                            continue  # Skip to the next student

                        # Mark attendance for the remaining subjects
                        for subject in selected_subjects:
                            if subject not in enrolled_subjects:
                                continue  # Skip if the student is not enrolled in the subject

                            # Check if attendance is already marked for the day
                            if (student_id in self.attendance_db and 
                                current_date in self.attendance_db[student_id] and 
                                subject in self.attendance_db[student_id][current_date]):
                                print(f"⏳ Attendance for {subject} is already marked today for {student_name}.")
                                continue

                            # Mark attendance
                            self.attendance_db.setdefault(student_id, {}).setdefault(current_date, {})[subject] = {
                                "semester": semester,
                                "department": department,
                                "shift": shift,
                                "time": current_time
                            }
                            print(f"✅ Attendance marked for {student_name} in {subject} at {current_time}.")
                            marked_students.add(student_id)

                        # Draw rectangle and label on the frame
                        color = (0, 255, 0)  # Green
                        label = f"{student_name} (Attendance Marked)"
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                else:
                    # Draw rectangle and label for unknown face
                    color = (0, 0, 255)  # Red
                    label = "Unknown"
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Check if all students in the semester have been marked
            total_students = 0
            for student_id, student_data in self.student_db.items():
                if (student_data.get("semester") == semester and
                    student_data.get("department") == department and
                    student_data.get("shift") == shift):
                    total_students += 1

            if len(marked_students) >= total_students:
                print("\n✅ Attendance for all students in the selected semester has been marked.")
                break

            # Display the frame
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Step 6: Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()

        # Step 7: Save the updated attendance records
        self.save_attendance()
        print("\n✅ Attendance marking for the entire semester is complete.")

    def view_attendance(self):
        """View attendance records with filtering options."""

        if not self.attendance_db:
            print("📂 No attendance records found!")
            return

        print("\n🔍 Attendance Viewing Options:")
        print("1️⃣ View all attendance")
        print("2️⃣ View by Student ID")
        print("3️⃣ View by Date")
        print("4️⃣ View by Semester, Department, Shift, and Subject")
        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            # Show full attendance
            print("\n📋 **All Attendance Records:**")
            for student_id, dates in self.attendance_db.items():
                student_name = self.student_db.get(student_id, {}).get("name", "Unknown")
                print(f"\n👤 {student_name} (ID: {student_id})")
                for date, subjects in dates.items():
                    print(f"📅 {date}:")
                    for subject, details in subjects.items():
                        print(f"   📖 {subject} | ⏰ {details['time']} | 🎓 {details['semester']} | 🏢 {details['department']} | 🌅 {details['shift']}")

        elif choice == "2":
            student_id = input("Enter Student ID: ")
            if student_id not in self.attendance_db:
                print("❌ No attendance found for this student.")
                return
            student_name = self.student_db.get(student_id, {}).get("name", "Unknown")
            print(f"\n📋 Attendance for {student_name} (ID: {student_id}):")
            for date, subjects in self.attendance_db[student_id].items():
                print(f"📅 {date}:")
                for subject, details in subjects.items():
                    print(f"   📖 {subject} | ⏰ {details['time']} | 🎓 {details['semester']} | 🏢 {details['department']} | 🌅 {details['shift']}")

        elif choice == "3":
            date = input("Enter Date (YYYY-MM-DD): ")
            print(f"\n📋 Attendance on {date}:")
            found = False
            for student_id, dates in self.attendance_db.items():
                if date in dates:
                    found = True
                    student_name = self.student_db.get(student_id, {}).get("name", "Unknown")
                    print(f"\n👤 {student_name} (ID: {student_id})")
                    for subject, details in dates[date].items():
                        print(f"   📖 {subject} | ⏰ {details['time']} | 🎓 {details['semester']} | 🏢 {details['department']} | 🌅 {details['shift']}")
            if not found:
                print("❌ No attendance records for this date.")

        elif choice == "4":
            # Step 1: Get semester, department, and shift
            print("\nSelect Semester:")
            semesters = [f"Semester {i}" for i in range(1, 9)]
            for i, sem in enumerate(semesters, 1):
                print(f"{i}. {sem}")
            
            while True:
                semester_choice = input("Enter choice: ").strip()
                if semester_choice.isdigit() and 1 <= int(semester_choice) <= len(semesters):
                    semester_choice = int(semester_choice) - 1
                    semester = semesters[semester_choice]
                    break
                else:
                    print("❌ Invalid choice. Please enter a valid number.")

            print("\nSelect Department:")
            departments = ["Computer Science", "Information Technology", "Software Engineering", "Electrical Engineering"]
            for i, dept in enumerate(departments, 1):
                print(f"{i}. {dept}")
            
            while True:
                department_choice = input("Enter choice: ").strip()
                if department_choice.isdigit() and 1 <= int(department_choice) <= len(departments):
                    department_choice = int(department_choice) - 1
                    department = departments[department_choice]
                    break
                else:
                    print("❌ Invalid choice. Please enter a valid number.")

            print("\nSelect Shift:")
            shifts = ["Morning", "Evening"]
            for i, shift in enumerate(shifts, 1):
                print(f"{i}. {shift}")
            
            while True:
                shift_choice = input("Enter choice: ").strip()
                if shift_choice.isdigit() and 1 <= int(shift_choice) <= len(shifts):
                    shift_choice = int(shift_choice) - 1
                    shift = shifts[shift_choice]
                    break
                else:
                    print("❌ Invalid choice. Please enter a valid number.")

            # Step 2: Get subject selection
            print("\nSelect Subject:")
            # Collect all unique subjects for the selected semester, department, and shift
            all_subjects = set()
            for student_id, student_data in self.student_db.items():
                if (student_data.get("semester") == semester and
                    student_data.get("department") == department and
                    student_data.get("shift") == shift):
                    all_subjects.update(student_data.get("subjects", []))

            if not all_subjects:
                print("⚠️ No subjects found for the selected semester, department, and shift.")
                return

            # Display all unique subjects
            print("Available Subjects:")
            for i, subject in enumerate(all_subjects, 1):
                print(f"{i}. {subject}")

            # Allow the user to select multiple subjects
            while True:
                selected_indices = input("\nEnter the numbers of subjects (e.g., 1 2 3): ").strip().split()
                if all(index.isdigit() and 1 <= int(index) <= len(all_subjects) for index in selected_indices):
                    selected_subjects = [list(all_subjects)[int(index) - 1] for index in selected_indices]
                    break
                else:
                    print("❌ Invalid subject selection. Please enter valid numbers.")

            # Step 3: Filter and display attendance records
            print(f"\n📋 Attendance for {', '.join(selected_subjects)} in {semester}-{department}-{shift}:\n")

            # Lists to store marked and unmarked students
            marked_students = {subject: [] for subject in selected_subjects}
            unmarked_students = {subject: [] for subject in selected_subjects}

            # Get current date
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Iterate through all students in the selected semester, department, and shift
            for student_id, student_data in self.student_db.items():
                if (student_data.get("semester") == semester and
                    student_data.get("department") == department and
                    student_data.get("shift") == shift):

                    student_name = student_data.get("name", "Unknown")
                    for subject in selected_subjects:
                        if (student_id in self.attendance_db and
                            current_date in self.attendance_db[student_id] and
                            subject in self.attendance_db[student_id][current_date]):
                            # Student has marked attendance for this subject
                            marked_students[subject].append((student_id, student_name))
                        else:
                            # Student has not marked attendance for this subject
                            unmarked_students[subject].append((student_id, student_name))

            # Display attendance for each selected subject
            for subject in selected_subjects:
                print(f"\n📖 Subject: {subject}")
                if marked_students[subject]:
                    print("✅ Marked Students:")
                    for student_id, student_name in marked_students[subject]:
                        print(f"👤 {student_name} (ID: {student_id})")
                else:
                    print("❌ No students have marked attendance for this subject today.")

                if unmarked_students[subject]:
                    print("\n❌ Unmarked Students:")
                    for student_id, student_name in unmarked_students[subject]:
                        print(f"👤 {student_name} (ID: {student_id})")
                else:
                    print("\n✅ All students have marked attendance for this subject today.")

            # Step 4: Generate CSV Report
            csv_filename = f"attendance_report_{semester}_{department}_{shift}.csv"
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Student ID", "Student Name", "Subject", "Attendance Status"])
                for subject in selected_subjects:
                    for student_id, student_name in marked_students[subject]:
                        writer.writerow([student_id, student_name, subject, "Marked"])
                    for student_id, student_name in unmarked_students[subject]:
                        writer.writerow([student_id, student_name, subject, "Unmarked"])

            print(f"\n📄 CSV report generated: {csv_filename}")

        else:
            print("⚠️ Invalid choice or feature not implemented yet.")

    def re_register_face(self):
        """Re-register a student's face by updating their face embedding in the database."""
        
        # Step 1: Get the student ID
        student_id = input("Enter Student ID: ").strip()
        if student_id not in self.student_db:
            print("❌ Student not found in the database.")
            return

        # Step 2: Capture a new face image
        print("📸 Capturing new face image for re-registration...")
        image = self.capture_image()
        if image is None:
            print("❌ Error: Failed to capture image. Please try again.")
            return

        # Step 3: Extract face embeddings
        embeddings, _ = self.extract_embeddings(image)
        if not embeddings:
            print("❌ Error: No face detected! Please try again.")
            return

        # Step 4: Update the student's face embedding in the database
        new_embedding = embeddings[0]  # Use the first detected face
        self.student_db[student_id]["embedding"] = new_embedding.tolist()
        self.save_student_database()

        print(f"✅ Face re-registered successfully for Student ID: {student_id}.")

    def detect_unauthorized_faces(self):
        """Detect unauthorized faces (faces not in the database) and terminate automatically."""
        
        # Initialize InsightFace
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return

        print("🎥 Starting unauthorized face detection. Detecting faces...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame.")
                break

            # Detect faces using RetinaFace
            faces = RetinaFace.detect_faces(frame)
            if not faces:
                cv2.imshow("Unauthorized Face Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Extract embeddings using InsightFace
            face_results = face_app.get(frame)
            if not face_results:
                cv2.imshow("Unauthorized Face Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Process each detected face
            for face in face_results:
                bbox = face.bbox.astype(int)  # Get bounding box coordinates
                embedding = face.embedding  # Get face embedding

                # Match the face with the database
                student_id, student_name = self.match_student(embedding)

                if student_id:
                    # Fetch student details
                    student_data = self.student_db[student_id]
                    semester = student_data.get("semester", "N/A")
                    department = student_data.get("department", "N/A")
                    shift = student_data.get("shift", "N/A")

                    # Draw green rectangle and label for authorized faces
                    color = (0, 255, 0)  # Green
                    label = f"Authorized: {student_name}"
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Display the frame with the authorized face
                    cv2.imshow("Unauthorized Face Detection", frame)
                    cv2.waitKey(2000)  # Wait for 2 seconds to show the result

                    # Show student details
                    print("\n✅ Authorized Face Detected:")
                    print(f"👤 Name: {student_name}")
                    print(f"🆔 ID: {student_id}")
                    print(f"📚 Semester: {semester}")
                    print(f"🏢 Department: {department}")
                    print(f"🌅 Shift: {shift}")

                    # Release the camera and close the window
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # Terminate the program

                else:
                    # Draw red rectangle and label for unauthorized faces
                    color = (0, 0, 255)  # Red
                    label = "Unauthorized"
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Display the frame with the unauthorized face
                    cv2.imshow("Unauthorized Face Detection", frame)
                    cv2.waitKey(2000)  # Wait for 2 seconds to show the result

                    # Show unauthorized message
                    print("\n❌ Unauthorized Face Detected.")

                    # Release the camera and close the window
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # Terminate the program

            # Display the frame
            cv2.imshow("Unauthorized Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()


    def get_semester_dept_shift(self):
        """Prompt user to select semester, department, shift (reusable)."""
        print("\nSelect Semester:")
        semesters = [f"Semester {i}" for i in range(1, 9)]
        for i, sem in enumerate(semesters, 1):
            print(f"{i}. {sem}")
        semester = semesters[int(input("Choice: ")) - 1]

        print("\nSelect Department:")
        departments = ["Computer Science", "IT", "Software Eng", "Electrical Eng"]
        for i, dept in enumerate(departments, 1):
            print(f"{i}. {dept}")
        department = departments[int(input("Choice: ")) - 1]

        print("\nSelect Shift:")
        shift = "Morning" if input("Shift (1=Morning, 2=Evening): ") == "1" else "Evening"
        return semester, department, shift

    def add_subjects_to_semester(self):
        """Add subjects to a semester-department-shift (max 8, no duplicates)."""
        # Step 1: Get semester, department, shift
        semester, department, shift = self.get_semester_dept_shift()  # (New helper function)
        
        # Step 2: Fetch existing subjects (if any)
        key = f"{semester}-{department}-{shift}"
        existing_subjects = self.subjects_db.get(key, [])
        
        # Step 3: Add new subjects
        print(f"\nCurrent subjects: {existing_subjects}")
        while len(existing_subjects) < 8:
            subject = input("Enter subject (or 'done'): ").strip()
            if subject.lower() == "done":
                break
            if subject in existing_subjects:
                print("⚠️ Subject already exists!")
                continue
            existing_subjects.append(subject)
            print(f"✅ Added: {subject}")
        
        # Step 4: Save
        self.subjects_db[key] = existing_subjects
        self.save_subjects_db()
        print(f"\n📖 Updated subjects for {key}: {existing_subjects}")
        
    def add_student_from_camera(self):
        """Add a new student with face capture, auto-assigning subjects from their semester."""
        # Step 1: Capture face image
        print("\n📸 Capturing face for student registration...")
        image = self.capture_image()
        if image is None:
            print("❌ Face capture cancelled or failed.")
            return False

        # Step 2: Extract face embedding
        embeddings, _ = self.extract_embeddings(image)
        if not embeddings:
            print("❌ No face detected! Please try again.")
            return False

        # Step 3: Check if student already exists
        new_embedding = np.array(embeddings[0])
        existing_student_id, existing_name = self.match_student(new_embedding)
        if existing_student_id:
            print(f"⚠️ Student already exists: {existing_name} (ID: {existing_student_id})")
            return False

        # Step 4: Get semester, department and shift
        print("\nSelect student's academic details:")
        try:
            semester, department, shift = self.get_semester_dept_shift()
        except (ValueError, IndexError):
            print("❌ Invalid selection!")
            return False

        # Step 5: Get subjects for this semester (auto-assigned)
        semester_key = f"{semester}-{department}-{shift}"
        subjects = self.subjects_db.get(semester_key, [])
        
        if not subjects:
            print(f"❌ No subjects found for {semester_key}!")
            print("Please add subjects first using the subject management option.")
            return False

        # Step 6: Get student ID and name
        while True:
            student_id = input("\nEnter Student ID: ").strip()
            if not student_id:
                print("❌ ID cannot be empty!")
                continue
            if student_id in self.student_db:
                print("❌ Student ID already exists!")
                continue
            break

        student_name = input("Enter Student Name: ").strip()
        while not student_name:
            print("❌ Name cannot be empty!")
            student_name = input("Enter Student Name: ").strip()

        # Step 7: Add student to database
        self.student_db[student_id] = {
            "name": student_name,
            "semester": semester,
            "department": department,
            "shift": shift,
            "subjects": subjects,  # Auto-assigned from semester
            "embedding": new_embedding.tolist()
        }
        
        self.save_student_database()
        
        # Step 8: Show confirmation
        print("\n✅ Student Added Successfully!")
        print(f"Name: {student_name}")
        print(f"ID: {student_id}")
        print(f"Program: {semester}, {department}, {shift}")
        print(f"Enrolled Subjects: {', '.join(subjects)}")
        
        return True

    def add_multiple_students_from_camera(self):
        """Add multiple students with face capture, auto-assigning subjects from their semester."""
        # Step 1: Get academic details once for all students
        print("\n📝 First, select the academic program for all students:")
        try:
            semester, department, shift = self.get_semester_dept_shift()
        except (ValueError, IndexError):
            print("❌ Invalid selection!")
            return False

        # Step 2: Verify subjects exist for this program
        semester_key = f"{semester}-{department}-{shift}"
        subjects = self.subjects_db.get(semester_key, [])
        if not subjects:
            print(f"❌ No subjects found for {semester_key}!")
            print("Please add subjects first using the subject management option.")
            return False

        print(f"\n🔔 All students will be enrolled in: {', '.join(subjects)}")

        # Step 3: Initialize counters
        added_count = 0
        duplicate_count = 0
        failed_count = 0

        # Step 4: Continuous student addition
        while True:
            print("\n" + "="*40)
            print(f"👥 Current Batch Stats: {added_count} added | {duplicate_count} duplicates | {failed_count} failed")
            print("="*40)
            
            choice = input("\nAdd new student? (yes/no): ").strip().lower()
            if choice != 'yes':
                break

            # Step 5: Face capture and processing
            print("\n📸 Capturing face for new student...")
            image = self.capture_image()
            if image is None:
                print("⚠️ Face capture cancelled or failed.")
                failed_count += 1
                continue

            embeddings, _ = self.extract_embeddings(image)
            if not embeddings:
                print("❌ No face detected! Please try again.")
                failed_count += 1
                continue

            # Step 6: Check for duplicates
            new_embedding = np.array(embeddings[0])
            existing_id, existing_name = self.match_student(new_embedding)
            if existing_id:
                print(f"⚠️ Duplicate found: {existing_name} (ID: {existing_id})")
                duplicate_count += 1
                continue

            # Step 7: Get student details
            while True:
                student_id = input("\nEnter Student ID: ").strip()
                if not student_id:
                    print("❌ ID cannot be empty!")
                    continue
                if student_id in self.student_db:
                    print("❌ ID already exists!")
                    continue
                break

            student_name = input("Enter Student Name: ").strip()
            while not student_name:
                print("❌ Name cannot be empty!")
                student_name = input("Enter Student Name: ").strip()

            # Step 8: Add student to database
            self.student_db[student_id] = {
                "name": student_name,
                "semester": semester,
                "department": department,
                "shift": shift,
                "subjects": subjects,
                "embedding": new_embedding.tolist()
            }
            added_count += 1
            print(f"\n✅ Added {student_name} (ID: {student_id})")

        # Step 9: Finalize and save
        if added_count > 0:
            self.save_student_database()
            print("\n" + "="*40)
            print(f"🎉 Batch Addition Complete!")
            print(f"✅ Successfully added: {added_count} students")
            print(f"⚠️ Skipped duplicates: {duplicate_count}")
            print(f"❌ Failed attempts: {failed_count}")
            print(f"📚 All students enrolled in: {', '.join(subjects)}")
        else:
            print("\n⚠️ No students were added.")

        return added_count > 0

    def update_subjects(self):
        """Manage subjects (update names or delete) for a semester-department-shift."""
        # Step 1: Select academic program
        print("\n" + "="*40)
        print("📚 SUBJECT MANAGEMENT SYSTEM")
        print("="*40)
        print("\n📝 Select academic program to manage subjects:")

        semester, department, shift = self.get_semester_dept_shift()
        program_key = f"{semester}-{department}-{shift}"
        
        # Step 2: Verify subjects exist
        current_subjects = self.subjects_db.get(program_key, [])
        if not current_subjects:
            print(f"\n❌ No subjects found for {program_key}")
            return False

        # Step 3: Display subjects with actions
        print(f"\nCurrent subjects for {program_key}:")
        for i, subject in enumerate(current_subjects, 1):
            print(f"{i}. {subject}")
        
        print("\nActions:")
        print("1. Update subject name")
        print("2. Delete subject")
        print("0. Cancel")
        
        # Step 4: Get action choice
        while True:
            try:
                action = int(input("\nChoose action (0-2): "))
                if action in (0, 1, 2):
                    break
                print("❌ Invalid choice! Please enter 0, 1, or 2")
            except ValueError:
                print("❌ Please enter a number")

        if action == 0:
            print("\n❌ Operation canceled.")
            return False

        # Step 5: Get subject to modify
        while True:
            try:
                sub_choice = int(input("\nEnter subject number to modify: "))
                if 1 <= sub_choice <= len(current_subjects):
                    target_subject = current_subjects[sub_choice-1]
                    break
                print(f"❌ Please enter a number between 1-{len(current_subjects)}")
            except ValueError:
                print("❌ Please enter a number")

        # UPDATE SUBJECT FLOW
        if action == 1:
            # Step 6: Get new subject name
            while True:
                new_subject = input(f"\nEnter new name for '{target_subject}': ").strip()
                if not new_subject:
                    print("❌ Subject name cannot be empty!")
                    continue
                if new_subject in current_subjects:
                    print("⚠️ Subject already exists in this program!")
                    continue
                if len(new_subject) > 50:
                    print("⚠️ Subject name too long (max 50 characters)")
                    continue
                break

            # Step 7: Confirm update
            confirm = input(f"\n⚠️ Change '{target_subject}' to '{new_subject}'? (yes/no): ").lower()
            if confirm != 'yes':
                print("\n❌ Update canceled.")
                return False

            # Step 8: Update in subjects database
            current_subjects[current_subjects.index(target_subject)] = new_subject
            self.subjects_db[program_key] = current_subjects

            # Step 9: Update in all student records
            updated_students = 0
            for student_id, student_data in self.student_db.items():
                if (student_data.get("semester") == semester and
                    student_data.get("department") == department and
                    student_data.get("shift") == shift):
                    if target_subject in student_data["subjects"]:
                        student_data["subjects"].remove(target_subject)
                        student_data["subjects"].append(new_subject)
                        updated_students += 1

            # Step 10: Update attendance records
            updated_attendance = 0
            for student_id, dates in self.attendance_db.items():
                if student_id in self.student_db:  # Only for existing students
                    student_data = self.student_db[student_id]
                    if (student_data.get("semester") == semester and
                        student_data.get("department") == department and
                        student_data.get("shift") == shift):
                        for date, subjects in dates.items():
                            if target_subject in subjects:
                                subjects[new_subject] = subjects.pop(target_subject)
                                updated_attendance += 1

            # Step 11: Save all changes
            self.save_subjects_db()
            self.save_student_database()
            self.save_attendance()

            # Step 12: Show results
            print("\n" + "="*40)
            print("✅ UPDATE COMPLETE")
            print("="*40)
            print(f"Program: {program_key}")
            print(f"Changed: '{target_subject}' → '{new_subject}'")
            print(f"Updated {updated_students} student records")
            print(f"Modified {updated_attendance} attendance records")
            print("\nUpdated subjects list:")
            for i, subject in enumerate(self.subjects_db[program_key], 1):
                print(f"{i}. {subject}")

        # DELETE SUBJECT FLOW
        elif action == 2:
            # Step 6: Confirm deletion
            confirm = input(f"\n⚠️ PERMANENTLY delete '{target_subject}'? (yes/no): ").lower()
            if confirm != 'yes':
                print("\n❌ Deletion canceled.")
                return False

            # Step 7: Remove from subjects database
            current_subjects.remove(target_subject)
            self.subjects_db[program_key] = current_subjects

            # Step 8: Remove from student records
            updated_students = 0
            for student_id, student_data in self.student_db.items():
                if (student_data.get("semester") == semester and
                    student_data.get("department") == department and
                    student_data.get("shift") == shift):
                    if target_subject in student_data["subjects"]:
                        student_data["subjects"].remove(target_subject)
                        updated_students += 1

            # Step 9: Remove from attendance records
            updated_attendance = 0
            for student_id, dates in self.attendance_db.items():
                if student_id in self.student_db:  # Only for existing students
                    student_data = self.student_db[student_id]
                    if (student_data.get("semester") == semester and
                        student_data.get("department") == department and
                        student_data.get("shift") == shift):
                        for date in list(dates.keys()):
                            if target_subject in dates[date]:
                                del dates[date][target_subject]
                                updated_attendance += 1
                                if not dates[date]:  # Remove date if empty
                                    del dates[date]

            # Step 10: Save all changes
            self.save_subjects_db()
            self.save_student_database()
            self.save_attendance()

            # Step 11: Show results
            print("\n" + "="*40)
            print("✅ DELETION COMPLETE")
            print("="*40)
            print(f"Program: {program_key}")
            print(f"Deleted subject: '{target_subject}'")
            print(f"Removed from {updated_students} student records")
            print(f"Cleared {updated_attendance} attendance records")
            print("\nRemaining subjects:")
            if self.subjects_db[program_key]:
                for i, subject in enumerate(self.subjects_db[program_key], 1):
                    print(f"{i}. {subject}")
            else:
                print("No subjects remaining")

        return True

from FacialAttendance import FacialAttendance

class AttendanceSystem:
    def __init__(self):
        self.facial_attendance = FacialAttendance()  # Create an instance

    def run(self):
        """ Main menu for the attendance system """
        while True:
            print("\n1. Add Subjects")
            print("2. Add Single Student")
            print("3. Add Multiple Student")
            print("4. Delete Entire Class")
            print("5. Delete Single Student")
            print("6. Mark Sigle Student attendance")
            print("7. Mark Entire Class attendance")
            print("8. View Attendance")
            # print("8. update attendance")
            # print("9. export attendance")
            print("9. Detect Unauthorized Faces")
            print("10. Re Register Face")
            print("11. Update Subject")
            print("12. Exit")

            choice = input("Enter your choice: ")

            if choice == "1":
                self.facial_attendance.add_subjects_to_semester()
            elif choice == "2":
                self.facial_attendance.add_student_from_camera()
            elif choice == "3":
                self.facial_attendance.add_multiple_students_from_camera()
            elif choice == "4":
                self.facial_attendance.delete_entire_semester()
            elif choice == "5":
                self.facial_attendance.delete_student_from_face()   
            elif choice == "6":
                self.facial_attendance.mark_single_attendance_from_face()
            elif choice == "7":
                self.facial_attendance.mark_entire_semester_attendance_with_faces()
            elif choice == "8":
                self.facial_attendance.view_attendance()
            elif choice == "9":
                self.facial_attendance.detect_unauthorized_faces()   
            elif choice == "10":
                id = input("enter your id ")
                self.facial_attendance.re_register_face(id)
            elif choice == "9":
                self.facial_attendance.update_subjects() 
            elif choice == "12":
                print("Exiting...")
                break
            else:
                print("Invalid choice! Try again.")

if __name__ == "__main__":
    system = AttendanceSystem()
    system.run()

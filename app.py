############################################# IMPORTING ################################################
import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

############################################# VARIABLES ################################################
app = Flask(__name__)

nimgs = 10
datetoday = date.today().strftime("%d-%m-%Y")
datetoday2 = date.today().strftime("%d-%m-%Y")

if not os.path.isdir('./Attendance'):
    os.makedirs('./Attendance')

if not os.path.isdir('./faces'):
    os.makedirs('./faces')

student_file = 'StudentDetails.csv'


############################################# UTILITY: RENDER HOME PAGE ################################
def render_home_page(message=None):
    """Helper to render home.html with attendance + total user info + optional message."""
    if os.path.isfile(f'Attendance-{datetoday2}.csv'):
        df = pd.read_csv(f'Attendance-{datetoday2}.csv')
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        l = len(df)
    else:
        names = []
        rolls = []
        times = []
        l = 0

    if os.path.isdir('./faces'):
        totalreg = len(os.listdir('./faces'))
    else:
        totalreg = 0

    return render_template(
        'home.html',
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=totalreg,
        mess=message
    )


############################################# FACE EXTRACTION ################################################
def extract_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return []

    extracted_faces = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (50, 50))
        extracted_faces.append(face_img)

    return extracted_faces


############################################# TRAIN THE MODEL ################################################
def train_model():
    faces = []
    labels = []

    if not os.path.isdir('./faces'):
        return None

    for userfolder in os.listdir('./faces'):
        folder_path = os.path.join('./faces', userfolder)
        if not os.path.isdir(folder_path):
            continue

        for imgname in os.listdir(folder_path):
            img_path = os.path.join(folder_path, imgname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces.append(gray.flatten())
            labels.append(userfolder)

    if len(faces) == 0:
        return None

    faces = np.array(faces)
    labels = np.array(labels, dtype=str)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'face_recognition_model.pkl')
    return "Model Trained"


############################################# HOME ROUTE ################################################
@app.route('/')
def home():
    return render_home_page()


############################################# ADD NEW USER ################################################
@app.route('/add', methods=['POST'])
def add():
    # Get form values and clean them
    newusername = request.form.get('newusername', '').strip()
    newuserid = request.form.get('newuserid', '').strip()

    if newusername == '' or newuserid == '':
        return render_home_page("⚠ Name and ID are required!")

    # Load or create StudentDetails.csv
    if os.path.isfile(student_file):
        df = pd.read_csv(student_file)
    else:
        df = pd.DataFrame(columns=['Roll', 'Name'])

    # Ensure Roll column is treated as string
    if 'Roll' not in df.columns:
        df['Roll'] = []
    df['Roll'] = df['Roll'].astype(str)

    # Check for duplicate ID
    if newuserid in df['Roll'].values:
        return render_home_page("⚠ User ID already exists!")

    # Append new user
    df.loc[len(df)] = [newuserid, newusername]
    df.to_csv(student_file, index=False)

    # Create user folder for face images
    userpath = os.path.join('./faces', newuserid)
    if not os.path.isdir(userpath):
        os.makedirs(userpath)

    # Capture images from webcam
    cam = cv2.VideoCapture(0)
    count = 0

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if len(faces) > 0:
            face = faces[0]
            img_path = os.path.join(userpath, f'{newuserid}_{count}.jpg')
            cv2.imwrite(img_path, face)
            count += 1
            cv2.putText(frame, f"Images: {count}/{nimgs}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face Not Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Add User - Press Q", frame)

        if cv2.waitKey(1) == ord('q') or count == nimgs:
            break

    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Train / update the model after adding user
    train_model()

    return render_home_page("🎉 User Added Successfully!")


############################################# START ATTENDANCE ################################################
@app.route('/start')
def start():
    # Check if model exists
    if not os.path.isfile('face_recognition_model.pkl'):
        return render_home_page("⚠ Train the model first by adding a user!")

    # Check if student details exist
    if not os.path.isfile(student_file):
        return render_home_page("⚠ No registered users found! Please add a user first.")

    knn = joblib.load('face_recognition_model.pkl')
    cam = cv2.VideoCapture(0)

    # Load today's attendance file or create empty
    att_file = f'Attendance-{datetoday2}.csv'
    if os.path.isfile(att_file):
        df_att = pd.read_csv(att_file)
    else:
        df_att = pd.DataFrame(columns=['Name', 'Roll', 'Time'])

    # Load student details
    student_df = pd.read_csv(student_file)
    if 'Roll' not in student_df.columns:
        student_df['Roll'] = []
    student_df['Roll'] = student_df['Roll'].astype(str)

    registered = []

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if len(faces) > 0:
            # Convert face to grayscale and flatten
            face_gray = cv2.cvtColor(faces[0], cv2.COLOR_BGR2GRAY)
            face = face_gray.flatten().reshape(1, -1)

            # Predict ID
            try:
                pred = knn.predict(face)[0]  # string label (userfolder)
            except Exception:
                pred = "Unknown"

            pred_str = str(pred)

            # Match with student details
            match = student_df[student_df['Roll'] == pred_str]
            if not match.empty:
                student_name = match['Name'].values[0]
            else:
                student_name = pred_str

            # Mark attendance once per user
            if pred_str not in registered and student_name != "Unknown":
                time_now = datetime.now().strftime("%H:%M:%S")
                df_att.loc[len(df_att)] = [student_name, pred_str, time_now]
                df_att.to_csv(att_file, index=False)
                registered.append(pred_str)

            cv2.putText(frame, student_name, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Attendance - Press Q", frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return render_home_page()


############################################# DELETE ATTENDANCE ################################################
@app.route('/delete_attendance', methods=['POST'])
def delete_attendance():
    att_file = f'Attendance-{datetoday2}.csv'
    if os.path.isfile(att_file):
        os.remove(att_file)
        return render_home_page("🧹 Today's attendance cleared!")
    else:
        return render_home_page("⚠ No attendance file found for today.")


############################################# MAIN ################################################
if __name__ == '__main__':
    app.run(debug=True)

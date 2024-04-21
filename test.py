import cv2
import dlib
import numpy as np
import tensorflow as tf
from collections import Counter
import time

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotion_model = tf.keras.models.load_model(r'C:\Users\shilp\OneDrive\Documents\Luminar\Internship\Emotion_model\emotion_model.h5')

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

emotion_list = []
start_time = time.time()
duration = 20  # in seconds

while (time.time() - start_time) < duration:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        emotion_list.append(emotion_dict[maxindex])

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Determine the most frequent emotion
if emotion_list:
    most_common_emotion = Counter(emotion_list).most_common(1)[0][0]
    print(emotion_list)
    print("Most common emotion:", most_common_emotion)
else:
    print("No emotions detected.")

import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

model = load_model("model/gesture_model.h5")
labels = np.load("model/labels.npy")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

prediction_history = []
sentence = []
last_added = ""

while True:

    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    landmarks = []

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:
            for lm in hand.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

        while len(landmarks) < 126:
            landmarks.append(0)

        if len(landmarks) > 126:
            landmarks = landmarks[:126]

        arr = np.array(landmarks).reshape(1,126)

        pred = model.predict(arr)

        classID = np.argmax(pred)
        confidence = pred[0][classID]

        prediction_history.append(classID)

        if len(prediction_history) > 10:
            prediction_history.pop(0)

        if prediction_history.count(classID) > 7 and confidence > 0.8:

            text = labels[classID]

            if text != last_added:

                sentence.append(text)
                last_added = text

        else:
            text = ""

    else:
        text = "No Hand"

    cv2.putText(frame,f"Gesture: {text}",(30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.putText(frame,"Sentence: " + " ".join(sentence),(30,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(255,0,0),2)

    cv2.imshow("ISL Recognition",frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        sentence = []

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer

#Load the model
model = TFSMLayer("model/", call_endpoint="serving_default")

# Image formatting
def preprocess_frame(frame):
    image = cv2.resize(frame, (224, 224))
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1 #Resnet style normalization
    image = tf.expand_dims(image, axis=0)
    return image

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    input_data = preprocess_frame(frame)

    # Predict using the model
    output = model(input_data)
    output_key = list(output.keys())[0]
    prediction = output[output_key].numpy()[0][0]

    label = "Cooking" if prediction < 0.5 else "Lock In"
    color = (0, 255, 0) if label == "Cooking" else (0, 0, 255)

    # Show prediction on the frame
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("FocusPal", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



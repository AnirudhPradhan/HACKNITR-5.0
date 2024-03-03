from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO
import cv2
import os
import time
import base64
from inference_sdk import InferenceHTTPClient

# app = Flask(__name__)

# Initialize the InferenceHTTPClient

app = Flask(__name__)
socketio = SocketIO(app)


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="QchEUFcH0DjxlJBZA3Am"
)

def infer_and_draw(image_path):
    result = CLIENT.infer(image_path, model_id="malpractice_prevention/4")

    image = cv2.imread(image_path)

    # Draw rectangles on the image
    for prediction in result['predictions']:
        if prediction['class'] != 'Correct posture':
            x, y, w, h = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
            color = (0, 255, 0)  # Green 
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            text = f"{prediction['class']} ({prediction['confidence']:.2f})"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
last_eye_detected_time = time.time()
cheating_count = 0

@app.route('/')
def main():
    return render_template('index.html')


@app.route('/corp')
def corp():
    return render_template('index2.html')


@app.route('/mal', methods=['GET', 'POST'])
def malprac():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            # image_path = './malpractice_prevention-4/train/images/457307_jpg.rf.7da220953395629ddfb2b0c9026e609d.jpg'
            # file.save(image_path)
            image_path = os.path.join(os.getcwd(), 'uploaded_image.jpg')
            file.save(image_path)    

            result_image = infer_and_draw(image_path)

            _, buffer = cv2.imencode('.jpg', result_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template('result.html', image_base64=image_base64)

    return render_template('index_m.html')


@app.route('/proctor')
def index():
    return render_template('index_p.html')

def detect_cheating(frame):
    global last_eye_detected_time, cheating_count

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        last_eye_detected_time = time.time()
        cheating_popup_displayed = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) > 0:
            last_eye_detected_time = time.time()
    

    if time.time() - last_eye_detected_time > 5 :
        socketio.emit('cheating_alert', {'message': f'This person is cheating! - {cheating_count}'})
        cheating_count += 1
        if cheating_count > 20:
            cap.release()
            cv2.destroyAllWindows()

    if len(faces) > 1:
        socketio.emit('cheating_alert', {'message': f'Multiple faces detected! - {cheating_count}'})
        cheating_count += 1
        if cheating_count > 20:
            cap.release()
            cv2.destroyAllWindows()


    return frame

def generate_frames():
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = detect_cheating(frame)
        time.sleep(0.1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == '__main__':
    app.run(debug=True)

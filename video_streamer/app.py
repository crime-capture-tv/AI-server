from flask import Flask, render_template, Response
import cv2
from threading import Thread
import numpy as np

from video_streamer import VideoStreamer


app = Flask(__name__)

url1 = "http://192.168.0.30:8000/stream.mjpeg"
url2 = "http://192.168.0.17:8000/stream.mjpeg"
network_storage = '\\\\192.168.0.26\\crimecapturetv\\suspicion-video'
request_server_url = "http://192.168.0.12:8080/api/v1/stores/1/videos?storeNo=1"

streamer = VideoStreamer(url1, url2, network_storage, request_server_url, request=True, plot_box=False)

# Start the threads outside of the route or function
thread1 = Thread(target=streamer.stream_video, args=(url1, 0))
thread2 = Thread(target=streamer.stream_video, args=(url2, 1))
thread3 = Thread(target=streamer.webcam_capture, args=(2,))

thread1.start()
thread2.start()
thread3.start()

@app.route('/')
def index():
    return render_template('index.html')  # 간단한 HTML 페이지를 반환

def gen_frames():
    while True:
        frame_ = streamer.process_frames()  # Note: We renamed the start() method to process_frames()
        if frame_[0] is not None and frame_[1] is not None and frame_[2] is not None:
            h1, w1 = frame_[0].shape[:2]
            h2, w2 = frame_[1].shape[:2]
            h3, w3 = frame_[2].shape[:2]
        else:
            h1, w1, h2, w2, h3, w3 = 480, 640, 480, 640, 480, 640  # Default dimensions, you can change this

        height = max(h1, h2, h3)
        width = max(w1, w2, w3)

        black_frame = np.zeros((height, width, 3), dtype=np.uint8)

        def get_frame_or_signal(frame):
            if frame is None:
                frame_with_text = black_frame.copy()
                cv2.putText(frame_with_text, 'No Signal', (width // 4, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return frame_with_text
            else:
                return cv2.resize(frame, (width, height))

        resized_frame0 = get_frame_or_signal(frame_[0])
        resized_frame1 = get_frame_or_signal(frame_[1])
        resized_frame2 = get_frame_or_signal(frame_[2])

        img_path = 'logo.png'
        img = cv2.imread(img_path)
        img = cv2.resize(img, (width, height))

        upper_frame = np.hstack((resized_frame0, resized_frame1))
        lower_frame = np.hstack((resized_frame2, img))
        both_frames = np.vstack((upper_frame, lower_frame))

        ret, buffer = cv2.imencode('.jpg', both_frames)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # MJPEG 스트림 형식

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False, threaded=True)

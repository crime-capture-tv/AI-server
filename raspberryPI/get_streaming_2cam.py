import cv2
import requests
import numpy as np
from threading import Thread
import time


exit_signal = False

def stream_video(url, position):
    global exit_signal
    frame_count = 0
    fps = 0.0
    start_time = time.time()
    try:    
        stream = requests.get(url, stream=True, timeout=5)  
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to stream at {url}")
        return
    byte_stream = b""

    while True:
        for chunk in stream.iter_content(chunk_size=1024):
            byte_stream += chunk
            a = byte_stream.find(b'\xff\xd8')
            b = byte_stream.find(b'\xff\xd9')

            if a != -1 and b != -1:
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time

                jpg = byte_stream[a:b+2]
                byte_stream = byte_stream[b+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if img is not None:
                    frame[position] = img
                    cv2.putText(img, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if elapsed_time > 1:  # 1 second interval
                    # print(f"Stream {position} FPS: {fps}")
                    # Reset frame count and start time for next FPS calculation cycle
                    frame_count = 0
                    start_time = time.time()

                if exit_signal:
                    return

def webcam_capture(position):
    global exit_signal
    frame_count = 0 
    fps = 0.0
    start_time = time.time()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame_img = cap.read()
        if ret:
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            cv2.putText(frame_img, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame[position] = frame_img

            if elapsed_time > 1:  # 1 second interval
                # print(f"webcam {position} FPS: {fps}")
                # Reset frame count and start time for next FPS calculation cycle
                frame_count = 0
                start_time = time.time()
        if exit_signal:
            return

frame = [None, None, None]
save_signal = False


cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)

url1 = "http://192.168.0.30:8000/stream.mjpeg"
url2 = "http://192.168.0.11:8000/stream.mjpeg"

thread1 = Thread(target=stream_video, args=(url1, 0))
thread2 = Thread(target=stream_video, args=(url2, 1))
thread3 = Thread(target=webcam_capture, args=(2,))

thread1.start()
thread2.start()
thread3.start()

while True:
    if frame[0] is not None and frame[1] is not None and frame[2] is not None:
        h1, w1 = frame[0].shape[:2]
        h2, w2 = frame[1].shape[:2]
        h3, w3 = frame[2].shape[:2]
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
    
    resized_frame0 = get_frame_or_signal(frame[0])
    resized_frame1 = get_frame_or_signal(frame[1])
    resized_frame2 = get_frame_or_signal(frame[2])

    img_path = 'logo2.png'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height))
    
    upper_frame = np.hstack((resized_frame0, resized_frame1))
    lower_frame = np.hstack((resized_frame2, img))
    both_frames = np.vstack((upper_frame, lower_frame))
    
    cv2.imshow('Stream', both_frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit_signal = True
        break

thread1.join()
thread2.join()
thread3.join()

cv2.destroyAllWindows()
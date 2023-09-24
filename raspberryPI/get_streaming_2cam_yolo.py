import cv2
import requests
import numpy as np
from threading import Thread
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

exit_signal = False

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out0 = None
out1 = None
recording0 = False
recording1 = False
recoding_signal = False

def stream_video(url, position, video_out_dir):
    global exit_signal, out0, out1, recording0, recording1, recoding_signal
    model = YOLO('yolov8n.pt')

    frame_count = 0
    fps = 0.0
    start_time = time.time()
    predict_signal = False

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
                    cv2.putText(img, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if predict_signal:
                        results = model.predict(img, classes=[0], verbose=False)

                        if len(results[0].boxes) == 0:
                            recoding_signal = False
                        else:
                            recoding_signal = True

                        for r in results:
                            annotator = Annotator(img)
                            boxes = r.boxes
                            for box in boxes:
                                b = box.xyxy[0]
                                c = box.cls
                                annotator.box_label(b, model.names[int(c)], color=(0, 0, 255))

                        frame_ = annotator.result()  
                        frame[position] = frame_

                        predict_signal = False

                    else:
                        frame[position] = img

                    if recoding_signal:  
                        if not recording0 and position == 0:
                            recording0 = True
                            out0 = cv2.VideoWriter(f'{video_out_dir}/output{position}.avi', fourcc, 20.0, (frame[0].shape[1], frame[0].shape[0]))
                        if recording0 and position == 0:
                            out0.write(frame[0])

                        if not recording1 and position == 1:
                            recording1 = True
                            out1 = cv2.VideoWriter(f'{video_out_dir}/output{position}.avi', fourcc, 20.0, (frame[1].shape[1], frame[1].shape[0]))
                        if recording1 and position == 1:
                            out1.write(frame[1])

                        # print('recording...')

                    else:
                        if recording0 and position == 0:
                            recording0 = False
                            out0.release()
                            out0 = None

                        if recording1 and position == 1:
                            recording1 = False
                            out1.release()
                            out1 = None

                    if elapsed_time > 1:
                        frame_count = 0
                        start_time = time.time()
                        predict_signal = True

                if exit_signal:
                    return

def webcam_capture(position):
    global exit_signal
    model = YOLO('yolov8n.pt')

    frame_count = 0 
    fps = 0.0
    start_time = time.time()
    predict_signal = False
    exceed_start_time = None
    exceed_end_time = None

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame_img = cap.read()
        if ret:
            frame_count += 1
            elapsed_time = time.time() - start_time
            if predict_signal:
                # 예측을 위한 코드
                results = model.predict(frame_img, classes=[0], verbose=False)  # class 0은 '사람' 클래스에 대한 것이라고 가정합니다.

                for r in results:
                    annotator = Annotator(frame_img)
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0]
                        c = box.cls
                        box_width = b[2] - b[0]
                        box_height = b[3] - b[1]
                        # box_width가 400을 초과하면 시작 시간을 기록
                        if box_width > 400 and exceed_start_time is None:
                            exceed_start_time = time.time()

                        # box_width가 400 미만이고 시작 시간이 이미 기록되어 있다면 종료 시간을 기록
                        if box_width < 400 and exceed_start_time is not None:
                            exceed_end_time = time.time()

                            duration = exceed_end_time - exceed_start_time
                            print(f"Duration when BBox Width was greater than 400: {duration:.2f} seconds")
                            
                            exceed_start_time = None
                            exceed_end_time = None

                        print(f"BBox Width: {box_width:.2f}, BBox Height: {box_height:.2f}")  # bounding box의 크기를 출력합니다.
                        annotator.box_label(b, model.names[int(c)], color=(0, 0, 255))

                frame_img = annotator.result()
                predict_signal = False

            frame[position] = frame_img

            if elapsed_time > 1:  # 1 second interval
                frame_count = 0
                start_time = time.time()
                predict_signal = True  # 1초마다 예측을 수행하기 위해 신호를 True로 설정합니다.

        if exit_signal:
            return

frame = [None, None, None]
save_signal = False


cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)

url1 = "http://192.168.0.30:8000/stream.mjpeg"
url2 = "http://192.168.0.85:8000/stream.mjpeg"

video_out_dir = 'output'

thread1 = Thread(target=stream_video, args=(url1, 0, video_out_dir))
thread2 = Thread(target=stream_video, args=(url2, 1, video_out_dir))
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

if out0:
    out0.release()
if out1:
    out1.release()
    
cv2.destroyAllWindows()
import cv2
import requests
import numpy as np
from threading import Thread
import time
import os
from ultralytics import YOLO

URL = "http://192.168.0.12:8080/api/v1/stores/1/videos?storeNo=1"

headers = {
    "accept": "*/*",
    "Content-Type": "application/json"
}
# 영상 저장위한 설정들(web)
file_path_0 = 'W:/suspicion-video/cctv-01/record_0.mp4'
file_path_1 = 'W:/suspicion-video/cctv-02/record_1.mp4'

fps = 12
fourcc_0 = cv2.VideoWriter_fourcc(*'MP4V')             # 인코딩 포맷 문자
fourcc_1 = cv2.VideoWriter_fourcc(*'MP4V')             # 인코딩 포맷 문자

width = 640
height = 480
size = (int(width), int (height))                       # 프레임 크기
out_0 = cv2.VideoWriter(file_path_0, fourcc_0, fps, size)     # VideoWriter 객체 
out_1 = cv2.VideoWriter(file_path_1, fourcc_1, fps, size)     # VideoWriter 객체 


exit_signal = False
check_0 = -1
check_1 = -1
request_check0 = -2
request_check1 = -2
time_list1 = []
time_list2 = ['0000-00-00-00-00-00']
time_check1 = ''
time_check2 = ''
time_check3 = ''
time_check4 = ''

def stream_video(url, position, out):
    global exit_signal
    global check_0
    global check_1
    global time_list1
    global request_check0
    global request_check1
    global time_check1
    global time_check2
    global time_check3
    global time_check4


    model = YOLO('yolov8n.pt') 

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
            time_1 = time.time()
            if a != -1 and b != -1:
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time

                jpg = byte_stream[a:b+2]
                byte_stream = byte_stream[b+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if img is not None:
                    results = model.predict(img,                       # 웹캠에서 불러오는 frame
                                classes=[0],                 # 인식가능한 객체종류중 사람(0)만 인식하기
                                verbose=False
                                #   tracker="bytetrack.yaml",  # 트래킹방식
                                #   tracker="botsort.yaml",    # 트래킹방식
                                #   conf=0.5,                    # conf정도에 따라서 객체인식여부
                                #   iou=0.5                      # 영역정확도에 따라서 객체인식 여부
                                )
                    annotated_frame = results[0].plot()
                    if str(round(time_1,1))[-1] == '0':
                        # print(f'1초마다 체크 - {position}')
                        if results[0].boxes:
                            # print(f'{position} 박스확인')
                            if position == 0:
                                check_0 = 1
                            elif position == 1:
                                check_1 = 1
                            p_time = time.localtime(time_1)             
                            time_list1.append(time.strftime('%Y-%m-%d-%H-%M-%S', p_time))
                        else:
                            if time_list1:
                                # print(f'{position}박스사라짐')
                                if position == 0:
                                    check_0 = 0
                                elif position == 1:
                                    check_1 = 0

                    if check_0==1 or check_1==1:
                        print('둘다 녹화중')                                         # check포인트가 1이면 영상저장
                        out.write(img)
                    elif check_0==0 and check_1==0:
                        # print('녹화 중단-----------------------------')
                        out.release()                               # 저장된 out을 영상으로 저장
                        if time_list1:
                            if position == 0:
                                request_check0 += 2
                            elif position == 1:
                                request_check1 += 2
                            if time_check1 == '':
                                time_check1 = time_list1[0]
                            if time_check2 == '':
                                time_check2 = time_list2[-1]
                            try:
                                if position == 0:
                                    os.rename('W:/suspicion-video/cctv-01/record_0.mp4', 'W:/suspicion-video/cctv-01/' + 'C0' + '_' + f'{time_check1}_{time_check2}.mp4')
                                elif position == 1:
                                    os.rename('W:/suspicion-video/cctv-02/record_1.mp4' , 'W:/suspicion-video/cctv-02/' + 'C1' + '_' + f'{time_check1}_{time_check2}.mp4')
                            except Exception as e:
                                # print(f"Error renaming file for position {position}: {e}")
                                # print('1')
                                pass

                    # img = results[0].plot()
                    frame[position] = annotated_frame
                    # out.write(img)
                    cv2.putText(annotated_frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if elapsed_time > 1:  # 1 second interval
                    # print(f"Stream {position} FPqS: {fps}")
                    # Reset frame count and start time for next FPS calculation cycle
                    frame_count = 0
                    start_time = time.time()

                if exit_signal:
                    out.release()
                    if time_list1:
                        if position == 0:
                            request_check0 += 2
                        elif position == 1:
                            request_check1 += 2 
                        if time_check1 == '':
                            time_check1 = time_list1[0]
                        if time_check2 == '':
                            time_check2 = time_list2[-1]
                        try:
                            if position == 0:
                                os.rename('W:/suspicion-video/cctv-01/record_0.mp4', 'W:/suspicion-video/cctv-01/' + 'C0' + '_' + f'{time_check1}_{time_check2}.mp4')
                            elif position == 1:
                                os.rename('W:/suspicion-video/cctv-02/record_1.mp4' , 'W:/suspicion-video/cctv-02/' + 'C1' + '_' + f'{time_check1}_{time_check2}.mp4')
                        except Exception as e:
                            # print(f"Error renaming file for position {position}: {e}")
                            # print('1')
                            pass
                    return


def webcam_capture(position):
    global exit_signal
    global time_list1
    global request_check0
    global request_check1
    global time_check1
    global time_check2
    global time_check3
    global time_check4
    global URL
    global headers

    model = YOLO('yolov8n.pt')
    frame_count = 0 
    fps = 0.0
    start_time = time.time()

    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while True:
        ret, frame_img = cap.read()
        time_1 = time.time()
        if ret:
            results = model.predict(frame_img,                     
                                classes=[0],                
                                verbose = False
                                )
            annotated_frame = results[0].plot()
            
            if str(round(time_1,1))[-1] == '0':
                try:
                    boxes =     results[0].boxes.xywh.cpu()
                    for box in boxes:
                        x,y,w,h = box
                        if ((width*height)*0.5) <= w*h:
                            p_time = time.localtime(time.time())
                            time_list2.append(time.strftime('%Y-%m-%d-%H-%M-%S', p_time))
                except:
                    pass

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            cv2.putText(annotated_frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame[position] = annotated_frame
        
            if elapsed_time > 1:  # 1 second interval
                # print(f"webcam {position} FPS: {fps}")
                # Reset frame count and start time for next FPS calculation cycle
                frame_count = 0
                start_time = time.time()
            if request_check0 == 0 and request_check1 == 0:
                if len(time_list2) >= 2:
                    print(time.time())
                    print('request11111111~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    
                    data = {f"suspicionVideoPath01": f"\\\\192.168.0.26\\crimecapturetv\\suspicion-video\\cctv-01\\C0_{time_check1}_{time_check2}.mp4",
                            f"suspicionVideoPath02": f"\\\\192.168.0.26\\crimecapturetv\\suspicion-video\\cctv-02\\C1_{time_check1}_{time_check2}.mp4",
                            "stayStartTime"                   : f"{time_list2[1]}",
                            "stayEndTime"                     : f"{time_list2[-1]}"
                    }
                    print(data)
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')
                    response = requests.post(URL, json=data, headers=headers)
                    print('*'*40)
                    print(response.status_code)
                    print(response.request.body.decode())
                    print(time.time())
                    print('*'*50)

                elif len(time_list2) == 1:
                    print(time.time())
                    print('request11111111~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    data = {f"suspicionVideoPath01": f"\\\\192.168.0.26\\crimecapturetv\\suspicion-video\\cctv-01\\C0_{time_check1}_{time_check2}.mp4",
                            f"suspicionVideoPath02": f"\\\\192.168.0.26\\crimecapturetv\\suspicion-video\\cctv-02\\C1_{time_check1}_{time_check2}.mp4",
                            "stayStartTime"                   : f"{time_list2[0]}",
                            "stayEndTime"                     : f"{time_list2[0]}"
                    }
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    response = requests.post(URL, json=data, headers=headers)
                    print('*'*40)
                    print(response.status_code)
                    print(response.request.body.decode())
                    print(time.time())
                    print('*'*50)
        
        if exit_signal:
            if request_check0 == -2:
                if len(time_list2) >= 2:
                    print(time.time())
                    print('request11111111~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    data = {f"suspicionVideoPath01": f"\\\\192.168.0.26\\crimecapturetv\\suspicion-video\\cctv-01\\C0_{time_check1}_{time_check2}.mp4",
                            f"suspicionVideoPath02": f"\\\\192.168.0.26\\crimecapturetv\\suspicion-video\\cctv-02\\C1_{time_check1}_{time_check2}.mp4",
                            "stayStartTime"                   :   f"{time_list2[1]}",
                            "stayEndTime"                     :   f"{time_list2[-1]}"
                    }
                    print(data)
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')
                    response = requests.post(URL, json=data, headers=headers)
                    print('*'*40)
                    print(response.status_code)
                    print(response.request.body.decode())
                    print(time.time())
                    print('*'*50)

                elif len(time_list2) == 1:
                    print(time.time())
                    print('request11111111~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    data = {f"suspicionVideoPath01": f"\\\\192.168.0.26\\crimecapturetv\\suspicion-video\\cctv-01\\C0_{time_check1}_{time_check2}.mp4",
                            f"suspicionVideoPath02": f"\\\\192.168.0.26\\crimecapturetv\\suspicion-video\\cctv-02\\C1_{time_check1}_{time_check2}.mp4",
                            "stayStartTime"                   : f"{time_list2[0]}",
                            "stayEndTime"                     : f"{time_list2[0]}"
                    }
                    print(data)
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    response = requests.post(URL, json=data, headers=headers)
                    print('*'*40)
                    print(response.status_code)
                    print(response.request.body.decode())
                    print(time.time())
                    print('*'*50)
                    
            return 


frame = [None, None, None]
save_signal = False

cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)

url1 = "http://192.168.0.30:8000/stream.mjpeg"
url2 = "http://192.168.0.85:8000/stream.mjpeg"

thread1 = Thread(target=stream_video, args=(url1, 0, out_0))
thread2 = Thread(target=stream_video, args=(url2, 1, out_1))
thread3 = Thread(target=webcam_capture, args=(2,))

thread1.start()
thread2.start()
thread3.start()

while True:
    # if request_check0 == 0 and request_check1 == 0:
        # print('2')
        # print('request_check~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
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

print('time_list1', time_list1)
print('time_list2', time_list2)

thread1.join()
thread2.join()
thread3.join()

cv2.destroyAllWindows()


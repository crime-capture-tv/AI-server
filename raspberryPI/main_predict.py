from ultralytics import YOLO
import cv2
import time
import json
import os
import shutil

def createFolder(directory):                         # 폴더가 없다면 만들기
    if not os.path.exists(directory):
        os.makedirs(directory)


model = YOLO('yolov8n.pt')                           # 모델 불러오기

cap = cv2.VideoCapture(0)                            # 웹캠 연결

# 영상 저장위한 설정들
file_path = 'Z:/suspicion-video/record.mp4'
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'MP4V')             # 인코딩 포맷 문자
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = (int(width), int (height))                       # 프레임 크기
out = cv2.VideoWriter(file_path, fourcc,fps, size)     # VideoWriter 객체 


track_history = dict()                                  # 시간저장 위한 딕셔너리객체생성
check_1 = -1                                           # 영상저장 위한 check포인트
time_list = []

while cap.isOpened():
    ret, frame = cap.read()                             # 다음 프레임 읽기
    time_1 = time.time()
    if ret:
        cv2.imshow('camera', frame)                  # 영상 재생
        # print('time_1', time_1)

        if str(round(time_1,1))[-1] == '0':
            print('11')
            results = model.predict(frame,                # 웹캠에서 불러오는 frame
                                classes=[0],              # 인식가능한 객체종류중 사람(0)만 인식하기
                                #   tracker="bytetrack.yaml", # 트래킹방식
                                #   tracker="botsort.yaml",
                                conf=0.5,                 # conf정도에 따라서 객체인식여부
                                iou=0.5                   # 영역정확도에 따라서 객체인식 여부
                                )
            # print(results)
            if results[0].boxes:                            # 사람인식함
                print('22')

                check_1 = 1                                 # 영상저장위한 check_1 변경
                # 시간 저장
                p_time = time.localtime(time_1)             
                time_list.append(time.strftime('%Y-%m-%d-%H-%M-%S', p_time))
            else:
                if time_list:
                    print('33')
                    check_1 = 0                          
                
        if check_1 == 1:                                         # check포인트가 1이면 영상저장
            out.write(frame)
        elif check_1 == 0:                               # 영상저장이 완료되면
            out.release()                               # 저장된 out을 영상으로 저장
            # 영상의 처음시간, 마지막시간 변환
            if time_list:
                # record_first_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time_list[0]))
                # record_last_time  = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time_list[1]))
                
                # 영상이름을 처음시간,마지막시간으로 변경
                try:
                    os.rename('Z:/suspicion-video/record.mp4', f'Z:/suspicion-video/{time_list[0]}_{time_list[1]}.mp4')
                    # filename = f'{time_list[0]}_{time_list[1]}.mp4'
                    # src = rf'./yolov8/'
                    # dir = rf'Z:/suspicion-video/'
                    # shutil.move(src + filename, dir + filename)
                except:
                    pass

        if cv2.waitKey(1) != -1:
            try:
                out.release()                               # 저장된 out을 영상으로 저장
            except:
                pass
            if time_list:                            # 1ms 동안 키 입력 대기
                print('44', time_list)

                try:
                    os.rename('Z:/suspicion-video/record.mp4', f'Z:/suspicion-video/{time_list[0]}_{time_list[1]}.mp4')
                    time.sleep(0.5)
                    # filename = f'{time_list[0]}_{time_list[1]}.mp4'
                    # src = rf'./yolov8/'
                    # dir = rf'Z:/suspicion-video/'
                    # shutil.move(src + filename, dir + filename)
                    time.sleep(0.5)

                except:
                    pass
                
                # 폴더가 없다면 생성
                createFolder('./json')
                # name1, name2, name3 = time_list[0].split(':')        # 변수명 분리
                file_path = f'./json/{time_list[0]}_{time_list[-1]}.json'   # 저장할 위치
                time.sleep(1)                                            # 파일 만들어질 시간주기
                with open(file_path, 'w') as f:
                    json.dump(time_list,f)                             # json파일 저장
            break
    else:
        print('no frame')
        break

cap.release()                                                                # 자원 반납
cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import time
import json
import os

def createFolder(directory):                         # 폴더가 없다면 만들기
    if not os.path.exists(directory):
        os.makedirs(directory)
        
model = YOLO('yolov8n.pt')                           # 모델 불러오기

cap = cv2.VideoCapture(0)                            # 웹캠 연결

# 영상 저장위한 설정들
file_path = 'record.mp4'
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'DIVX')             # 인코딩 포맷 문자
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = (int(width), int (height))                       # 프레임 크기
out = cv2.VideoWriter(file_path, fourcc, fps, size)     # VideoWriter 객체 


track_history = dict()                                  # 시간저장 위한 딕셔너리객체생성
check_1 = 0                                             # 영상저장 위한 check포인트

while cap.isOpened():
    ret, frame = cap.read()                             # 다음 프레임 읽기

    if ret:                                             # 모델 트래킹 시작
        results = model.track(frame,                    # 웹캠에서 불러오는 frame
                              persist=True,             # 객체인식때마다 부여되는 id유지여부
                              classes=[0],              # 인식가능한 객체종류중 사람(0)만 인식하기
                            #   tracker="bytetrack.yaml", # 트래킹방식
                            #   tracker="botsort.yaml",
                              conf=0.5,                 # conf정도에 따라서 객체인식여부
                              iou=0.5                   # 영역정확도에 따라서 객체인식 여부
                              )

    
        try:                                             # 화면에서 사람인식됬을경우
            check_1 = 1                                  # check포인트 1로 변경

            # 인식된 박스(box) x:box의 중심점, y: box의 중심점, w: box의 weight, h: box의 height
            boxes =     results[0].boxes.xywh.cpu()    
            track_ids = results[0].boxes.id.int().cpu().tolist()    # 인식된 박스의 id값 

            # 시간 저장
            p_time = time.localtime(time.time())
            for track_id in track_ids:
                if track_id not in track_history:           # 딕셔너리내 id가 없다면 만들고 시간저장
                    track_history[track_id] = [time.strftime('%Y%m%d_%H:%M:%S', p_time)]

                else:                                       # 딕셔너리내 id가 이미 있다면 시간저장
                    track_history[track_id].append(time.strftime('%Y%m%d_%H:%M:%S', p_time))
        except:
            check_1 = 0                                     # 추적되는 사람이 없다면 check포인트 0으로 변경
            pass

        finally:
            annotated_frame = results[0].plot()             # results값을 화면에 표시할수 있게 변환


        cv2.imshow('camera', annotated_frame)               # 영상 재생
        
        if check_1 == 1:                                    # check포인트가 1이면 영상저장
            out.write(annotated_frame) 

        if cv2.waitKey(1) != -1:                            # 1ms 동안 키 입력 대기
            print('111', track_history)                     # 저장된 시간 확인해보기

            # 폴더가 없다면 생성
            createFolder('./json')

            if track_history:           # 추적된 사람의 시각이 저장되어있고 트래킹이 끝났을경우 json형태로 내보내기 
                track_history2 = dict()                     # 처음시각, 마지막시각 저장
                for nums in track_history:
                    first_time = track_history[nums][0]     # 각 id의 처음시간
                    last_time = track_history[nums][-1]     # 각 id의 마지막시간
                    track_history2[nums] = [first_time,last_time]   # track_history2 딕셔너리에 저장하기
                
                name1, name2, name3 = track_history[1][0].split(':')        # 변수명 분리
                file_path = f'./json/ID111_{name1+name2+name3}.json'           # 저장할 위치
                time.sleep(1)                                               # 파일 만들어질 시간주기
                with open(file_path, 'w') as f:
                    json.dump(track_history2,f)                             # json파일 저장
            break                                                           # 아무 키라도 입력이 있으면 중지
    else:
        print('no frame')
        break

cap.release()                                                                # 자원 반납
cv2.destroyAllWindows()




from ultralytics import YOLO
import cv2
import time
import json
import os

#폴더없다면 만들기
def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 모델 불러오기
model = YOLO('yolov8n.pt')  

# 웹캠 연결
cap = cv2.VideoCapture(0)               

# 영상 저장위한 설정
# file_path = 'record.mp4'
# fps = 3
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')            # 인코딩 포맷 문자
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# size = (int(width), int (height))                   # 프레임 크기
# out = cv2.VideoWriter(file_path, fourcc, fps, size) # VideoWriter 객체 

# 시간저장 위한 딕셔너리객체생성
track_history = dict()

# 화면에서 일정비율 이상 인식될시 시간저장위한 딕셔너리 객체생성
track_box = dict()


while cap.isOpened():
    ret, frame = cap.read()           # 다음 프레임 읽기

    if ret:
        # 모델 트래킹 시작
        results = model.track(frame,                       # 웹캠에서 불러오는 frame
                              persist=True,                # 객체인식때마다 부여되는 id유지여부
                              classes=[0],                 # 인식가능한 객체종류중 사람(0)만 인식하기
                            #   tracker="bytetrack.yaml",  # 트래킹방식
                            #   tracker="botsort.yaml",    # 트래킹방식
                              conf=0.5,                    # conf정도에 따라서 객체인식여부
                              iou=0.5                      # 영역정확도에 따라서 객체인식 여부
                              )

        # 화면에서 사람인식됬을경우
        try:
            # 인식된 박스 x:box의 중심점, y: box의 중심점, w: box의 weight, h: box의 height
            boxes =     results[0].boxes.xywh.cpu()
            # 인식된 박스의 id값 
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # 저장할 현재시각 설정
            p_time = time.localtime(time.time())


            # 시간 저장 및 화면의 일정비율 이상시 별도로 시간리스트저장
            for box, track_id in zip(boxes, track_ids):
                # x:box의 중심점, y: box의 중심점, w: box의 weight, h: box의 height
                x,y,w,h = box

                # id딕셔너리안에 id없다면 만들고 시간저장
                if track_id not in track_history:
                    track_history[track_id] = [time.strftime('%Y%m%d_%H:%M:%S', p_time)]
                    # box딕셔너리안에 id없을것이므로 생성
                    track_box[track_id] = []
                # 딕셔너리내 id가 이미 있다면 시간저장
                else:
                    track_history[track_id].append(time.strftime('%Y%m%d_%H:%M:%S', p_time))
                    # 만약 id 크기가 일정퍼센트 이상이라면 시간저장
                    if ((width*height)*0.5) <= w*h:
                        track_box[track_id].append(time.strftime('%Y%m%d_%H:%M:%S', p_time))
                # 인식된 객체가 화면의 일정비율(0.5)이상일경우 시간저장
                if ((width*height)*0.5) <= w*h:
                    track_box[track_id].append(time.strftime('%Y%m%d_%H:%M:%S', p_time))
                

        except:
            pass

        finally:
            # results 값을 화면에 표시할수 잇게 변환
            annotated_frame = results[0].plot()

        # 영상 재생
        cv2.imshow('camera', annotated_frame)   
        
        # 1ms 동안 키 입력 대기
        if cv2.waitKey(1) != -1:    

            # 저장된 시간 확인해보기
            print('111', track_history)
            print('222', track_box)

            # 만약 폴더가 없다며 만들기
            createFolder('./json')
            # 추적된 사람의 시각이 저장되어있고 트래킹이 끝났을경우 json형태로 내보내기 ////
            if track_history:
                # 처음시각, 마지막시각 저장
                track_history2 = dict()
                for nums in track_history:
                    # 각 id의 처음시간
                    first_time = track_history[nums][0]
                    # 각 id의 마지막시간
                    last_time = track_history[nums][-1]
                    track_history2[nums] = [first_time,last_time]
                
                # json 저장
                name1, name2, name3 = track_history[1][0].split(':')
                file_path = f'./json/ID222_time_{name1+name2+name3}.json'
                time.sleep(1)
                with open(file_path, 'w') as f:
                    json.dump(track_history2,f)
                track_history = dict()

            # 추적된 사람의 시각이 저장되어있고 트래킹이 끝났을경우 json형태로 내보내기 ////
            if track_box:
                # 처음시각, 마지막시각 저장
                track_box2 = dict()
                for nums in track_box:
                    # 각 id의 처음시간
                    first_time = track_box[nums][0]
                    # 각 id의 마지막시간
                    last_time = track_box[nums][-1]
                    track_box2[nums] = [first_time,last_time]
                
                # 변수명 분리
                name1, name2, name3 = track_box[1][0].split(':')
                # 저장할 위치
                file_path = f'./json/BOX222_time_{name1+name2+name3}.json'
                # 파일 만들어질 시간주기
                time.sleep(1)
                # json파일 저장
                with open(file_path, 'w') as f:
                    json.dump(track_box2,f)
            
            # 아무 키라도 입력이 있으면 중지
            break                   
    else:
        print('no frame')
        break

# 자원 반납
cap.release()                           
cv2.destroyAllWindows()




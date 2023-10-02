import cv2
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO
import shutil


class Preprocessing():
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    
    def file_load(self, input_file):
        print('Preprocessing start ...')
        self.input_file = input_file
        self.input_file_path = '/'.join(input_file.split('/')[:-1])
        self.input_file_name = input_file.split('/')[-1].split('.')[0]


    def input_video_data(self):
        cap = cv2.VideoCapture(self.input_file)

        if not cap.isOpened():
            print("Error: Could not open video file.")
        else:
            # 비디오의 해상도 얻기
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 비디오의 프레임 속도 얻기
            fps = cap.get(cv2.CAP_PROP_FPS)

        # 비디오 캡처 객체를 닫기
        cap.release()
        return {'Resolution': [frame_width, frame_height], 'fps': fps}


    def regulate_resolution_fps(self, resolution, target_fps):
        self.regulate_file_name = f'{self.input_file_name}_regulate.mp4'
        output_file = os.path.join(self.input_file_path, self.regulate_file_name)

        cap = cv2.VideoCapture(self.input_file)

        # 원본 비디오의 프레임 속도 얻기
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 비디오 작성자 객체를 준비
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, target_fps, (resolution['width'], resolution['height']))

        frame_count = 0

        while True:
            ret, frame = cap.read() 
            if not ret:
                break
            
            # 해상도 변경
            frame_resized = cv2.resize(frame, (resolution['width'], resolution['height']))
            
            # 프레임 속도가 목표보다 낮은 경우, 몇몇 프레임을 복제하여 삽입
            if original_fps < target_fps:
                repeat_frame_count = target_fps // original_fps
                for _ in range(repeat_frame_count):
                    out.write(frame_resized)

            # 프레임 속도가 목표보다 높은 경우, 몇몇 프레임을 건너뛰기
            elif original_fps > target_fps:
                if frame_count % (original_fps // target_fps) != 0:
                    frame_count += 1
                    continue
                out.write(frame_resized)

            # 프레임 속도가 목표와 동일한 경우, 모든 프레임을 삽입
            else:
                out.write(frame_resized)
            
            frame_count += 1

        # 자원 해제
        cap.release()
        out.release()
        print(f'Regulate video / resolution : ({resolution["width"]} x {resolution["height"]}),  fps : {target_fps}')


    def segment_video(self, segment_duration=4, step_duration=2):
        # self.segment_dir_name = f'{self.input_file_path}/{self.input_file_name}_segment'
        self.segment_dir_name = os.path.join(self.input_file_path, f'{self.input_file_name}_segment')
        input_file = os.path.join(self.input_file_path, self.regulate_file_name)
        output_dir = self.segment_dir_name

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(input_file)

        # 비디오 프레임 레이트 얻기
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 각 세그먼트의 프레임 수 계산
        frames_per_segment = fps * segment_duration

        # 스텝별 프레임 수 계산
        frames_per_step = fps * step_duration

        # 비디오의 총 프레임 수 얻기
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 세그먼트의 수 계산
        num_segments = (total_frames - frames_per_segment) // frames_per_step + 1

        # 각 세그먼트를 반복하여 프레임을 캡처하고 저장
        for segment_idx in range(num_segments):
            segment_name = f"segment_{segment_idx:03d}.mp4"
            segment_path = os.path.join(output_dir, segment_name)

            # 비디오 쓰기 객체 생성
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(segment_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

            # 세그먼트의 시작 프레임 설정
            start_frame = segment_idx * frames_per_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_idx in range(frames_per_segment):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()

        cap.release()
        print(f"Video segmented into {num_segments} segments of {segment_duration} seconds each.")


    def crop_and_save_video(self):
        print('cropping video ...')
        input_dir = self.segment_dir_name
        self.crop_dir_name = f'{self.input_file_path}/{self.input_file_name}_crop'

        video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

        if not os.path.exists(self.crop_dir_name):
            os.makedirs(self.crop_dir_name)
        else:
            shutil.rmtree(self.crop_dir_name)
            os.makedirs(self.crop_dir_name)

        for crop_idx, video_file in enumerate(video_files):
            input_video_path = os.path.join(input_dir, video_file)
            crop_name = f"crop_{crop_idx:03d}.mp4"
            output_video_path = os.path.join(self.crop_dir_name, crop_name)

            cap = cv2.VideoCapture(input_video_path)

            # 원본 영상의 FPS 가져오기
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (224, 224))

            last_cropped_frame = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(frame, conf=0.5, classes=0, verbose=False, device='cuda')
                try:
                    cords = results[0].boxes[0].xyxy.tolist()
                    if len(results) > 0:
                        center_x = (cords[0][0] + cords[0][2]) / 2
                        center_y = (cords[0][1] + cords[0][3]) / 2
                        cropped_frame = frame[int(center_y) - 112:int(center_y) + 112, int(center_x) - 112:int(center_x) + 112]
                        last_cropped_frame = cropped_frame
                        out.write(cropped_frame)
                    elif last_cropped_frame is not None:
                        # 검출이 실패한 경우, 마지막으로 크롭된 프레임 사용
                        out.write(last_cropped_frame)
                except:
                    out.write(last_cropped_frame)

            cap.release()
            out.release()
        
        print('crop done')


    def delete_short_videos(self, min_duration=1):
        output_dir = self.crop_dir_name
        # 출력 디렉터리에서 모든 파일 가져오기
        for segment_name in os.listdir(output_dir):
            segment_path = os.path.join(output_dir, segment_name)
            
            # 파일이 비디오 파일인지 확인
            if os.path.isfile(segment_path) and segment_name.endswith('.mp4'):
                # 비디오 캡처 객체 생성
                cap = cv2.VideoCapture(segment_path)
                
                # 비디오의 FPS와 프레임 수를 가져오기
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 비디오 캡처 객체 해제
                cap.release()
                
                # FPS가 0인 경우 또는 비디오 길이가 최소 지속 시간 미만인 경우 삭제
                if fps == 0 or (frame_count / max(fps, 1)) < min_duration:
                    os.remove(segment_path)
                    print(f"Deleted: {segment_name} (Duration: {(frame_count / max(fps, 1)):.2f} seconds)")
                

if __name__ == '__main__':
    input_file = 'data/multi_cam/insert_2.mp4'

    preprocessing = Preprocessing()

    # get_data = preprocessing.input_video_data(input_file)

    resolution = {'width': 640, 'height': 480}
    target_fps = 30

    preprocessing.file_load(input_file)
    preprocessing.regulate_resolution_fps(resolution, target_fps)
    preprocessing.segment_video(segment_duration=4, step_duration=2)
    preprocessing.crop_and_save_video()
    preprocessing.delete_short_videos(min_duration=2.0)

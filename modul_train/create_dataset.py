import os
import shutil
import cv2
import numpy as np
import random
from ultralytics import YOLO

class CreateDataset():
    def __init__(self):
        self.model = YOLO("yolov8s.pt")
    
    def dir_create_and_seperate(self, label_list, src_folder, dic_folder):
        for label in label_list:
            # 원본 폴더 지정
            src_folder_label = f"{src_folder}/{label}"

            # 대상 폴더 지정
            train_folder = f"{dic_folder}/train/{label}"
            val_folder = f"{dic_folder}/val/{label}"
            test_folder = f"{dic_folder}/test/{label}"

            # 대상 폴더가 존재하지 않으면 생성
            for folder in [train_folder, val_folder, test_folder]:
                if not os.path.exists(folder):
                    os.makedirs(folder)

            # 파일 리스트 얻기
            files = [f for f in os.listdir(src_folder_label) if os.path.isfile(os.path.join(src_folder_label, f))]
            random.shuffle(files)  # 파일 리스트를 무작위로 섞기

            # 분할 비율 설정
            train_ratio = 0.7
            val_ratio = 0.15
            test_ratio = 0.15

            # 각 세트에 대한 파일 수 계산
            num_files = len(files)
            train_count = int(num_files * train_ratio)
            # train_count = 200
            val_count = int(num_files * val_ratio)
            # val_count = 30
            test_count = num_files - train_count - val_count

            # 파일을 해당 폴더로 복사
            for i, file in enumerate(files):
                src = os.path.join(src_folder_label, file)
                if i < train_count:
                    dst = os.path.join(train_folder, file)
                elif i < train_count + val_count:
                    dst = os.path.join(val_folder, file)
                else:
                    dst = os.path.join(test_folder, file)
                
                shutil.copy(src, dst)
            print(f'Seperate done ({label})  -  train_count : {train_count}, val_count : {val_count}, test_count : {test_count}')


    def get_new_file_name(self, label, dst, file_name):
        index = 0
        base_name, ext = os.path.splitext(file_name)
        new_file_name = file_name

        while os.path.exists(os.path.join(dst, new_file_name)):
            # new_file_name = f"{base_name.split('_')[0]}_aug_{index}{ext}"
            new_file_name = f'{label}_aug_{index}.mp4'
            index += 1

        return new_file_name


    def augment_video(self, label_list, dic_folder, aug_size):
        for label in label_list:
            print(f'augment start ({label})')
            path = f'{dic_folder}/train/{label}'
            file_list = os.listdir(path)

            output_dir = f'{dic_folder}/train/{label}'

            for i in range(1, aug_size):
                for file_name in file_list:
                    input_video = os.path.join(path, file_name)
                    new_file_name = self.get_new_file_name(label, output_dir, file_name)
                    output_video = os.path.join(output_dir, new_file_name)

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir) 
                    
                    angle = round(random.uniform(-10, 10), 2)
                    brightness_factor = round(random.uniform(-30, 30), 2)
                    red = round(random.uniform(-30, 30), 2)
                    blue = round(random.uniform(-30, 30), 2)
                    green = round(random.uniform(-30, 30), 2)

                    # 비디오 캡처 객체를 생성
                    cap = cv2.VideoCapture(input_video)

                    # 비디오 작성자 객체를 준비
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_video, fourcc, 30.0, (640, 480))

                    while True:
                        ret, frame = cap.read()
                        
                        if not ret:
                            break

                        # 밝기 조절
                        if brightness_factor >= 0:
                            frame = cv2.add(frame, np.ones(frame.shape, dtype='uint8') * np.uint8(brightness_factor))
                        else:
                            frame = cv2.subtract(frame, np.ones(frame.shape, dtype='uint8') * np.uint8(abs(brightness_factor)))

                        # 색상 조절
                        blue_, green_, red_ = cv2.split(frame)
                        red_ = cv2.add(red_, red)  # 빨간색 채널을 높이기
                        blue_ = cv2.subtract(blue_, blue)  # 파란색 채널을 낮추기
                        green_ = cv2.add(green_, green)  
                        frame = cv2.merge((blue_, green_, red_))
                        # 프레임의 크기를 얻습니다
                        (h, w) = frame.shape[:2]

                        # 회전 중심을 프레임의 중심으로 설정
                        center = (w / 2, h / 2)

                        # 회전 매트릭스를 얻습니다
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)

                        # 회전을 수행
                        rotated = cv2.warpAffine(frame, M, (w, h))

                        # 결과 비디오에 프레임을 쓰기
                        out.write(rotated)

                    # 자원 해제
                    cap.release()
                    out.release()
                print(f'Augment done  -  {label} / {i}')


    def video_crop(self, input_dataset, output_dataset):
        type_list = os.listdir(input_dataset)
        for type in type_list:
            input_dataset_type = os.path.join(input_dataset, type)
            output_dataset_type = os.path.join(output_dataset, type)

            label_list = os.listdir(input_dataset_type)
            for label in label_list:
                print(f"Croping...   {type}-{label}")

                input_dataset_type_label = os.path.join(input_dataset_type, label)
                output_dataset_type_label = os.path.join(output_dataset_type, label)
                if not os.path.exists(output_dataset_type_label):
                    os.makedirs(output_dataset_type_label)

                file_names = os.listdir(input_dataset_type_label)
                for file_name in file_names:
                    input_file = os.path.join(input_dataset_type_label, file_name)
                    output_file = os.path.join(output_dataset_type_label, file_name)

                    cap = cv2.VideoCapture(input_file)

                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_file, fourcc, fps, (224, 224))

                    last_cropped_frame = None

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results = self.model.predict(frame, conf=0.5, classes=0, verbose=False)
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
            print(f"Crop done") 


    def delete_short_videos(self, input_dataset, min_duration=2.0):
        print(input_dataset)
        type_list = os.listdir(input_dataset)
        for type in type_list:
            input_dataset_type = os.path.join(input_dataset, type)

            label_list = os.listdir(input_dataset_type)
            for label in label_list:
                print(f"file checking ...   {type}-{label}")
                input_dataset_type_label = os.path.join(input_dataset_type, label)

                file_names = os.listdir(input_dataset_type_label)
                for file_name in file_names:
                    file_ = os.path.join(input_dataset_type_label, file_name)
                    
                    # 파일이 비디오 파일인지 확인
                    if os.path.isfile(file_) and file_name.endswith('.mp4'):
                        # 비디오 캡처 객체 생성
                        cap = cv2.VideoCapture(file_)
                        
                        # 비디오의 FPS와 프레임 수를 가져오기
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # 비디오 캡처 객체 해제
                        cap.release()
                        
                        # FPS가 0인 경우 또는 비디오 길이가 최소 지속 시간 미만인 경우 삭제
                        if fps == 0 or (frame_count / max(fps, 1)) < min_duration:
                            os.remove(file_)
                            print(f"Deleted: {file_name} (Duration: {(frame_count / max(fps, 1)):.2f} seconds)")
                    


if __name__ == '__main__':
    label_list = ['catch', 'put', 'normal', 'insert']

    src_dir = f"origin_data/640x480_fps30"
    aug_size = 6
    dic_dir = f"data_set/data_set_aug{aug_size}"


    create_dataset = CreateDataset()

    create_dataset.dir_create_and_seperate(label_list, src_dir, dic_dir)

    create_dataset.augment_video(label_list, dic_dir, aug_size)

    create_dataset.video_crop(dic_dir, f"{dic_dir}_crop")

    create_dataset.delete_short_videos(f"{dic_dir}_crop", min_duration=2.0)
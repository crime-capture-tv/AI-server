from videoMAE_result import MultiVideoClassification
from preprocessing import Preprocessing
from process import Process
import os
from typing import Union
from fastapi import FastAPI
import shutil


app = FastAPI()

preprocessing = Preprocessing()

model_ckpt = "./models/20230914/checkpoint-426"
video_classification = MultiVideoClassification(model_ckpt)


@app.get('/')
def main():
    return 'classification api'


@app.get('/classification')
def preprocess(recordedAt: Union[str, None] = None, suspicionVideoPath: Union[str, None] = None):
    # 네트워크 경로와 로컬 경로를 지정합니다.
    print('공유폴더에서 영상 가져오기')
    # network_path = r'\\192.168.0.42\crimecapturetv\suspicion-video\test.mp4'
    # network_path = suspicionVideoPath.replace('\\\\', '/')
    network_path = suspicionVideoPath
    network_file_name = network_path.split('\\')[-1]
    local_path = f'output/{network_file_name}'
    # 파일을 복사합니다.
    shutil.copy(network_path, local_path)
    print('공유폴더에서 영상 가져오기 완료')

    input_file = local_path

    # Preprocessing 
    resolution = {'width': 640, 'height': 480}
    target_fps = 30

    preprocessing.file_load(input_file)
    preprocessing.regulate_resolution_fps(resolution, target_fps)
    preprocessing.segment_video(segment_duration=4, step_duration=2)
    preprocessing.crop_and_save_video()
    preprocessing.delete_short_videos(min_duration=2.0)


    input_file_path = '/'.join(input_file.split('/')[:-1])
    input_file_name = input_file.split('/')[-1].split('.')[0]
    video_path = os.path.join(input_file_path, f'{input_file_name}_crop')
    segment_video = os.listdir(video_path)

    start_time="20230912_14:44:14"
    duration = 4

    video_classification.load_videos(video_path)
    results = video_classification.predict(start_time, duration)
    print(results)


    first_catch_detected = False  # 첫 번째 'catch' 감지를 위한 플래그

    catchvideopath = os.path.join(input_file_path, f'{input_file_name}_segment')

    for key, value in results.items():
        # 첫 번째로 'catch'가 감지되었는지 확인
        if not first_catch_detected and value == 'catch':
            first_catch_detected = True
            # 해당 video를 공유폴더에 복사
            filename = f"hilight_{key}.mp4"
            filename = filename.replace(':', '')
            print('catchvideo : ', key)
            print('catchvideopath : ', catchvideopath)
            catchvideo = key
    person_data = results

    counter_time = {
        'startTime': '20230915_21:57:40',
        'endTime': '20230915_21:57:59'
    }

    thief, catch_time = Process.check_behavior(person_data, counter_time)  # 두 개의 값을 반환받습니다.

    resultText = ''

    if thief == 1:
        print('손놈이다!')
        resultText = {
            'result': '손놈이다!',
            'first_catch_time': catch_time,
            'suspicionVideoPath': '라즈베리 파이에서 받아온 원본 경로',
            'highlightVideoPath': None,
            'crimeType': 'normal'
        }
    else:
        print(f'도동놈이다! 첫 catch 시간: {catch_time}')
        network_path = fr'\\192.168.0.42\crimecapturetv\hilight-video\{filename}'
        print('하이라이트 저장 시작')
        # shutil.copy(os.path.join(catchvideopath, catchvideo), network_path)
        # with open(os.path.join(catchvideopath, catchvideo), 'rb') as src_file:
        #     with open(network_path, 'wb') as dst_file:
        #         dst_file.write(src_file.read())
        print('하이라이트 저장 완료')
        resultText = {
            'result': '도동놈이다!',
            'first_catch_time': catch_time,
            'suspicionVideoPath': '라즈베리 파이에서 받아온 원본 경로',
            'highlightVideoPath': network_path,
            'crimeType': 'theft'
        }

    return resultText
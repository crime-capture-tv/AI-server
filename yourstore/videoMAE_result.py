from torchvision.io import read_video
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToPILImage, ToTensor
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from datetime import datetime, timedelta
from IPython.display import Image
import torch
import os
import imageio


class OneVideoClassification():
    def __init__(self, model_ckpt):
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            ignore_mismatched_sizes=True)

        self.mean = self.image_processor.image_mean
        self.std = self.image_processor.image_std
        if "shortest_edge" in self.image_processor.size:
            height = width = self.image_processor.size["shortest_edge"]
        else:
            height = self.image_processor.size["height"]
            width = self.image_processor.size["width"]
        self.resize_to = (height, width)
        self.num_frames_to_sample = self.model.config.num_frames

       
    def load_video(self, video_file):
        self.video_file = video_file
        self.video_file_path = '/'.join(video_file.split('/')[:-1])
        self.video_file_name = video_file.split('/')[-1].split('.')[0]
    
        video, _, _ = read_video(video_file, pts_unit="sec")
        self.video = video.permute(3, 0, 1, 2)

        return self.video


    def transform_video(self, video):
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
        transformed_video = []
        for frame in video:
            frame = frame / 255.0
            frame = ToPILImage()(frame)
            frame = Resize(self.resize_to, antialias=True)(frame)
            frame = ToTensor()(frame)
            frame = Normalize(self.mean, self.std)(frame)
            transformed_video.append(frame)
        transformed_video = torch.stack(transformed_video)

        return transformed_video.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)


    def preprocessing(self):
        val_transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(self.transform_video),  # 수정된 부분
                        ]
                    ),
                ),
            ])
        self.video_tensor = val_transform({"video": self.video})["video"]

        return self.video_tensor


    def run_inference(self, model, video):
        perumuted_sample_test_video = video.permute(1, 0, 2, 3)
        inputs = {"pixel_values": perumuted_sample_test_video.unsqueeze(0)}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = model.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        return logits

    def predict(self):
        self.preprocessing()
        logits = self.run_inference(self.model, self.video_tensor)
        predicted_class_idx = logits.argmax(-1).item()

        return self.model.config.id2label[predicted_class_idx]


class MultiVideoClassification():
    def __init__(self, model_ckpt):
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            ignore_mismatched_sizes=True)

        self.mean = self.image_processor.image_mean
        self.std = self.image_processor.image_std
        if "shortest_edge" in self.image_processor.size:
            height = width = self.image_processor.size["shortest_edge"]
        else:
            height = self.image_processor.size["height"]
            width = self.image_processor.size["width"]
        self.resize_to = (height, width)
        self.num_frames_to_sample = self.model.config.num_frames

       
    def load_videos(self, video_dir):
        self.video_dir = video_dir
        self.video_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
        video_file = os.path.join(self.video_dir, self.video_list[0])
        video, _, _ = read_video(video_file, pts_unit="sec")
        self.video = video.permute(3, 0, 1, 2)

        return self.video


    def transform_video(self, video):
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
        transformed_video = []
        for frame in video:
            frame = frame / 255.0
            frame = ToPILImage()(frame)
            frame = Resize(self.resize_to, antialias=True)(frame)
            frame = ToTensor()(frame)
            frame = Normalize(self.mean, self.std)(frame)
            transformed_video.append(frame)
        transformed_video = torch.stack(transformed_video)

        return transformed_video.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)


    def preprocessing(self):
        val_transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(self.transform_video),  # 수정된 부분
                        ]
                    ),
                ),
            ])
        self.video_tensor = val_transform({"video": self.video})["video"]

        return self.video_tensor


    def run_inference(self, model, video):
        perumuted_sample_test_video = video.permute(1, 0, 2, 3)
        inputs = {"pixel_values": perumuted_sample_test_video.unsqueeze(0)}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = model.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        return logits

    def predict(self, date, duration, format_str):
        result_dic = {}
        for video_name in self.video_list:
            video_file = os.path.join(self.video_dir, video_name)
            video, _, _ = read_video(video_file, pts_unit="sec")
            self.video = video.permute(3, 0, 1, 2)
    
            self.preprocessing()
            logits = self.run_inference(self.model, self.video_tensor)
            predicted_class_idx = logits.argmax(-1).item()
            result = self.model.config.id2label[predicted_class_idx]

            video_index = int(video_name.split('.')[0].split('_')[1])
            seg_sec = video_index * duration
            new_date = date + timedelta(seconds=seg_sec)
            key_time = new_date.strftime(format_str)

            result_dic[key_time] = [video_index, result]

        return result_dic


if __name__ == '__main__':
    
    model_ckpt = "./models/20230914/checkpoint-426"   # best
    # model_ckpt = "./models/20230916/checkpoint-599"

    video_path = 'data/multi_cam/insert_2_crop'

    start_time="20230912-14-44-14"
    duration = 4

    format_str = "%Y%m%d-%H-%M-%S"

    date=datetime.strptime(start_time,format_str)

    # classification_videos = os.listdir(video_path)

    # video_classification = OneVideoClassification(model_ckpt)
    # data_dic = {}
    # for video_name in classification_videos:
    #     print(video_name)
    #     video_classification.load_video(os.path.join(video_path, video_name))
    #     result = video_classification.predict()
    #     print("Predicted class:", result)

    #     key_time = date.strftime("%Y%m%d_%H:%M:%S")
    #     date = date + timedelta(seconds=duration)

    #     data_dic[key_time] = result

    # print(data_dic)

    video_classification = MultiVideoClassification(model_ckpt)
    video_classification.load_videos(video_path)
    result = video_classification.predict(date, duration, format_str)
    print(result)
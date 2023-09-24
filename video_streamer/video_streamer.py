import cv2
import shutil
import requests
import requests
import numpy as np
from threading import Thread
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os
import json


class VideoStreamer:
    def __init__(self, url1, url2, network_storage, request_server_url, video_out_dir='output', img_path='logo2.png', request=False, plot_box=True):
        # Camera URLs
        self.url1 = url1
        self.url2 = url2
        self.network_storage = network_storage  # Network storage location
        self.request_server_url = request_server_url  # Server URL for requests
        self.request = request  # Boolean to decide if to send requests
        self.plot_box = plot_box  # Boolean to decide if bounding boxes should be plotted

        # Formatting for time
        self.format_str = "%Y-%m-%d-%H-%M-%S"

        # Various flags and data storage variables
        self.exit_signal = False
        self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.out = [None, None]
        self.recording = False
        self.recoding_signal = False
        self.video_out_dir = video_out_dir
        self.img_path = img_path
        self.frame = [None, None, None]
        self.stay_time = {'stayStartTime': None, 'stayEndTime': None}
        self.rec_start_time = None
        self.rec_end_time = None
        self.last_recoding_end_time = None
        self.send_request_signal = False
        self.last_request_time = None
        self.last_print_time = None
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')

    def stream_video(self, url, position):
        # Stream video from a given URL and position
        frame_count = 0
        fps = 0.0
        start_time = datetime.now()
        predict_signal = False

        # Try to start the video stream
        try:    
            stream = requests.get(url, stream=True, timeout=5)  
        except requests.exceptions.RequestException as e:
            print(f"Could not connect to stream at {url}")
            return

        byte_stream = b""   # Buffer for streaming data

        while True:
            # Loop through the bytes in the stream
            for chunk in stream.iter_content(chunk_size=1024):
                byte_stream += chunk
                # Checking for start and end of jpg data
                a = byte_stream.find(b'\xff\xd8')
                b = byte_stream.find(b'\xff\xd9')

                if a != -1 and b != -1:
                    frame_count += 1
                    elapsed_time = datetime.now() - start_time
                    fps = frame_count / elapsed_time.total_seconds()

                    jpg = byte_stream[a:b+2]
                    byte_stream = byte_stream[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if img is not None:
                        # Process image and display FPS
                        cv2.putText(img, f"CCTV{position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, f"FPS: {fps:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

                        if predict_signal:
                            # Run object detection using YOLO
                            results = self.model.predict(img, classes=[0], verbose=False, device='cpu')

                            # Check if any objects are detected
                            self.recoding_signal = len(results[0].boxes) != 0

                            # Annotate detected objects on the frame
                            for r in results:
                                annotator = Annotator(img)
                                boxes = r.boxes
                                for box in boxes:
                                    b = box.xyxy[0]
                                    c = box.cls
                                    if self.plot_box:
                                        annotator.box_label(b, self.model.names[int(c)], color=(0, 0, 255))

                            frame_ = annotator.result()  
                            self.frame[position] = frame_

                            predict_signal = False

                        else:
                            self.frame[position] = img

                        # Start recording if an object is detected
                        if self.recoding_signal and not self.recording:
                            if not self.last_recoding_end_time:
                                self.rec_start_time = datetime.now()
                                self.recording = True
                                for i in range(2):
                                    if self.frame[i] is not None:
                                        self.out[i] = cv2.VideoWriter(f'{self.video_out_dir}/output{i}.mp4', self.fourcc, 24.0, (self.frame[i].shape[1], self.frame[i].shape[0]))
                            else:
                                time_since_last_record = (datetime.now() - self.last_recoding_end_time).total_seconds()
                                if time_since_last_record > 3:  # 예를 들어, 마지막 녹화가 5초 이내라면 녹화를 다시 시작하지 않음
                                    self.rec_start_time = datetime.now()
                                    self.recording = True
                                    for i in range(2):  # 두 개의 비디오 출력을 위해 반복
                                        if self.frame[i] is not None:
                                            self.out[i] = cv2.VideoWriter(f'{self.video_out_dir}/output{i}.mp4', self.fourcc, 24.0, (self.frame[i].shape[1], self.frame[i].shape[0]))
                        
                        # Write frames to video if recording is active
                        if self.recording:
                            # Overlay "Recoding" text on the frame
                            text_width, text_height = cv2.getTextSize("recoding", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                             # Write frames to output video
                            cv2.putText(img, "Recoding", (img.shape[1] - text_width - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if self.frame[0] is not None:
                                if self.out[0] is not None:
                                    self.out[0].write(self.frame[0])
                            if self.frame[1] is not None:
                                if self.out[1] is not None:
                                    self.out[1].write(self.frame[1])
                            
                            # Print recording status every 2 seconds
                            if self.last_print_time is None or (datetime.now() - self.last_print_time).total_seconds() >= 2:
                                print("recoding ... ")
                                self.last_print_time = datetime.now()

                        if self.recording and self.rec_start_time is not None:
                            time_elapsed_since_recording = (datetime.now() - self.rec_start_time).total_seconds()

                            # Stop recording if conditions are met
                            if not self.recoding_signal and self.recording and time_elapsed_since_recording >= 2:
                                self.rec_end_time = datetime.now()
                                self.last_recoding_end_time = self.rec_end_time
                                self.recording = False
                                self.send_request_signal = True
                        
                        # Reset frame count and signal object detection every second
                        if elapsed_time.total_seconds() > 1:
                            frame_count = 0
                            start_time = datetime.now()
                            predict_signal = True

                    # Exit the loop if signaled
                    if self.exit_signal:
                        return

    def webcam_capture(self, position):
        frame_count = 0 
        fps = 0.0
        start_time = datetime.now()
        predict_signal = False
        exceed_start_time = None
        exceed_end_time = None

        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame_img = cap.read()
            if ret:
                frame_count += 1
                elapsed_time = datetime.now() - start_time

                # Displaying FPS and camera label on frame
                fps = frame_count / elapsed_time.total_seconds()
                cv2.putText(frame_img, f"CAM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_img, f"FPS: {fps:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

                # Indicate if processing is ongoing on frame
                if exceed_start_time is not None and exceed_end_time is None:
                    cv2.putText(frame_img, "Calculating ...", (frame_img.shape[1]-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Object prediction using YOLO model
                if predict_signal:
                    results = self.model.predict(frame_img, classes=[0], verbose=False, device='cpu')    # Assuming class 0 is 'human'.

                    for r in results:
                        annotator = Annotator(frame_img)
                        boxes = r.boxes
                        for box in boxes:
                            b = box.xyxy[0]
                            c = box.cls
                            box_width = b[2] - b[0]
                            box_height = b[3] - b[1]

                            # Record time when box width exceeds a threshold
                            if box_width > 400 and exceed_start_time is None:
                                exceed_start_time = datetime.now()

                            # Record end time when box width drops below the threshold
                            if box_width < 400 and exceed_start_time is not None:
                                exceed_end_time = datetime.now()

                                duration = (exceed_end_time - exceed_start_time).total_seconds()

                                print(f"Duration when BBox Width was greater than 400: {duration:.2f} seconds")

                                if duration > 4 :
                                    self.stay_time['stayStartTime'] = exceed_start_time.strftime(self.format_str)
                                    self.stay_time['stayEndTime'] = exceed_end_time.strftime(self.format_str)
                                
                                # Reset the start and end time
                                exceed_start_time = None
                                exceed_end_time = None

                            # Annotate bounding boxes on the frame if necessary
                            if self.plot_box:
                                annotator.box_label(b, self.model.names[int(c)], color=(0, 0, 255))
                    if self.plot_box:
                        frame_img = annotator.result()
                    predict_signal = False

                # Update the current frame
                self.frame[position] = frame_img

                # Reset counters and flag for object prediction every second
                if elapsed_time.total_seconds() > 1:
                    frame_count = 0
                    start_time = datetime.now()
                    predict_signal = True

            # Exit the loop if signaled
            if self.exit_signal:
                return

    def save_data(self):
        # Save video files and remove them if their duration is below a set threshold.

        self.filename_list = []

        # Iterate over both outputs
        for idx in range(2):
            if self.out[idx]:
                filename = f'{self.video_out_dir}/output{idx}.mp4'
                self.out[idx].release()
                self.out[idx] = None

                # Check the duration of the video
                vid = cv2.VideoCapture(filename)
                fps = vid.get(cv2.CAP_PROP_FPS)
                total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration = total_frames / fps
                vid.release()

                # Remove video if its duration is less than 5 seconds
                if video_duration < 5:
                    os.remove(filename)
                    print(f"The video duration of {video_duration:.2f} seconds is too short. {filename} has been deleted.")
                else:
                    print(f"Video saved to {filename}")
                    print(self.stay_time)
                    self.filename_list.append(filename)

        # If both videos are saved, proceed to copy and send request
        if len(self.filename_list) == 2:
            self.copy_and_request()

    def copy_and_request(self):
        # Copy saved video files to a target directory and send a request to a server.
        
        target_path_list = []

        # Copy video files to the desired location
        for idx, filename in enumerate(self.filename_list):
            new_name = f'cctv-{idx:02d}\\C{idx}_{self.rec_start_time.strftime(self.format_str)}_{self.rec_end_time.strftime(self.format_str)}.mp4'
            target_path = os.path.join(self.network_storage, new_name)
            target_path_list.append(target_path)
            shutil.copy(filename, target_path)
        
        print(f"File copied to {self.network_storage}")

        # Send request to another server with video paths and stay time
        headers = {
                "accept": "*/*",
                "Content-Type": "application/json"
            }
        data = {
                f"suspicionVideoPath01": f"{target_path_list[0]}",
                f"suspicionVideoPath02": f"{target_path_list[1]}",
                "stayStartTime": f"{self.stay_time['stayStartTime']}",
                "stayEndTime": f"{self.stay_time['stayEndTime']}"
            }
        json_string = json.dumps(data)
        print('send request ... ')
        response = requests.post(self.request_server_url, data=json_string, headers=headers)

        # Print result of request
        if response.status_code == 200:
            print("Successfully sent request to the server.")
        else:
            # print(f"Failed to send request. Status code: {response.status_code}. Response: {response.text}")
            print(f"Failed to send request. Status code: {response.status_code}.")

    def start(self):
        # Start threads to handle video streams
        thread1 = Thread(target=self.stream_video, args=(self.url1, 0))
        thread2 = Thread(target=self.stream_video, args=(self.url2, 1))
        thread3 = Thread(target=self.webcam_capture, args=(2,))
        
        thread1.start()
        thread2.start()
        thread3.start()

        while True:
            # Check if frames exist and get their dimensions
            if self.frame[0] is not None and self.frame[1] is not None and self.frame[2] is not None:
                h1, w1 = self.frame[0].shape[:2]
                h2, w2 = self.frame[1].shape[:2]
                h3, w3 = self.frame[2].shape[:2]
            else:
                # Default frame dimensions
                h1, w1, h2, w2, h3, w3 = 480, 640, 480, 640, 480, 640

            # Determine max dimensions for display window
            height = max(h1, h2, h3)
            width = max(w1, w2, w3)
            
            # Create a black frame for "No Signal" display
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Check and initiate data save request
            if self.send_request_signal and self.request:
                self.send_request_signal = False
                print(self.last_request_time)
                if self.last_request_time is None or (datetime.now() - self.last_request_time).total_seconds() >= 2:
                    self.last_request_time = datetime.now()
                    thread4 = Thread(target=self.save_data)
                    thread4.start()
                else:
                    print('thread is not ready')
            
            # Function to get the appropriate frame or a 'No Signal' display
            def get_frame_or_signal(frame):
                if frame is None:
                    frame_with_text = black_frame.copy()
                    cv2.putText(frame_with_text, 'No Signal', (width // 4, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    return frame_with_text
                else:
                    return cv2.resize(frame, (width, height))
            
            # Resize frames for display
            resized_frame0 = get_frame_or_signal(self.frame[0])
            resized_frame1 = get_frame_or_signal(self.frame[1])
            resized_frame2 = get_frame_or_signal(self.frame[2])

            # Read and resize the logo image
            img_path = 'logo.png'
            img = cv2.imread(img_path)
            img = cv2.resize(img, (width, height))
            
            # Stack frames horizontally and vertically to prepare the final display
            upper_frame = np.hstack((resized_frame0, resized_frame1))
            lower_frame = np.hstack((resized_frame2, img))
            both_frames = np.vstack((upper_frame, lower_frame))
            
            # Show the final frame
            cv2.imshow('Stream', both_frames)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.exit_signal = True
                break

        # Wait for all threads to finish
        thread1.join()
        thread2.join()
        thread3.join()
        # Clean up windows
        cv2.destroyAllWindows()

    def process_frames(self):
        return self.frame

if __name__ == "__main__":
    url1 = "http://192.168.0.30:8000/stream.mjpeg"
    url2 = "http://192.168.0.17:8000/stream.mjpeg"
    network_storage = '\\\\192.168.0.26\\crimecapturetv\\suspicion-video'
    request_server_url = "http://192.168.0.5:8080/api/v1/stores/1/videos?storeNo=1"

    streamer = VideoStreamer(url1, url2, network_storage, request_server_url, request=True, plot_box=False)
    streamer.start()
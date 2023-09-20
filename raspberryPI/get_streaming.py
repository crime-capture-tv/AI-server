import cv2
import requests
import numpy as np

url = "http://192.168.0.72:8000/stream.mjpeg"

stream = requests.get(url, stream=True)
byte_stream = b""

for chunk in stream.iter_content(chunk_size=1024):
    byte_stream += chunk
    a = byte_stream.find(b'\xff\xd8')
    b = byte_stream.find(b'\xff\xd9')
    
    if a != -1 and b != -1:
        jpg = byte_stream[a:b+2]
        byte_stream = byte_stream[b+2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is not None:
            cv2.imshow('Stream', img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

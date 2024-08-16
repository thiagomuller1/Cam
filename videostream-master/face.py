# Import the necessary packages
import cv2
import os

# URL da câmera RTSP (sem usuário e senha)
URL = 'rtsp://192.168.1.15//live/ch00_1'

# Só roda se for ffmpeg
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# Caminho para o arquivo de cascade Haar usado para detecção de faces
path_cascade = "haarcascade_frontalface_alt2.xml"
# Definindo o detector de faces
face_cascade = cv2.CascadeClassifier(path_cascade)
ds_factor = 0.6

class VideoCamera(object):
    def __init__(self):
        # Capturando o vídeo da URL especificada
        self.video = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
    
    def __del__(self):
        # Liberando a câmera
        self.video.release()
    
    def get_frame(self):
        # Extraindo frames
        ret, frame = self.video.read()
        if not ret:
            return None

        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break

        # Codificando o frame bruto do OpenCV para JPG e retornando
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

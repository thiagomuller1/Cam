#!/usr/bin/env python

# Exemplo simples de como rodar stream de uma câmera IP

import cv2
import os

# URL da câmera RTSP (sem usuário e senha)
URL = 'rtsp://192.168.1.15//live/ch00_1'
print('Conectando com: ' + URL)

# Só roda se for ffmpeg
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# Captura o vídeo da URL especificada
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Sem frame")
        break
    
    cv2.imshow('VIDEO', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

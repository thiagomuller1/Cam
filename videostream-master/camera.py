import cv2
import time
import datetime
import os

# Caminho onde as imagens capturadas serão salvas
IMG_PATH = "/home/engecorp/videostream/captured_img/"
time_grava = 3.0  # grava um frame a cada x segundos

# URL da câmera RTSP (sem usuário e senha)
CAMERA_URL = "rtsp://192.168.1.15//live/ch00_1"

# Função para detectar movimento
def motion_detection(URL):
    video_capture = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
    
    first_frame = None
    startTime = time.time()

    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        text = 'Normal'

        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_frame = cv2.GaussianBlur(greyscale_frame, (21, 21), 0)
        blur_frame = cv2.blur(gaussian_frame, (5, 5))
        greyscale_image = blur_frame
     
        if first_frame is None:
            first_frame = greyscale_image
            continue

        frame_delta = cv2.absdiff(first_frame, greyscale_image)
        thresh = cv2.threshold(frame_delta, 100, 255, cv2.THRESH_BINARY)[1]
        dilate_image = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) > 800:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = 'Movimento detectado'
        
        cv2.putText(frame, 'Status: %s' % (text),
                    (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.putText(frame, datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
       
        cv2.imshow('Security Feed', frame)
        cv2.imshow('Threshold (foreground mask)', dilate_image)
        cv2.imshow('Frame Delta', frame_delta)

        if (text == 'Movimento detectado') and ((time.time() - startTime) > time_grava):
            cv2.imwrite(os.path.join(IMG_PATH, str(time.time()) + ".jpg"), frame)
            first_frame = None
            startTime = time.time()

        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('Iniciando...')
    motion_detection(CAMERA_URL)

"""
Primer proyecto
Procesamiento digital de Imágenes

@author: Carlos Alfredo Pinto Hernández
"""

import cv2                           #cv2.__version__ = 4.7.0
import matplotlib.pyplot as plt
import numpy as np

def upload_video():
    #Carga el video y retorna una lista de todos los frames y el valor del fds
    
    cap = cv2.VideoCapture("Video_MAS_prueba2.mp4") 
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
     
    # Read until video is completed
    # Crea list para guardar los frames
    frames=[]

    fds = cap.get(cv2.CAP_PROP_FPS) #frames por segundo

    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
     
      if ret == True:

        # Se guarda cada frame en la lista
        frames.append(frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
     
      # Break the loop
      else: 
        break
     
    # When everything done, release the video capture object
    cap.release()
     
    # Closes all the frames
    cv2.destroyAllWindows()
    return frames, fds


frames, fds = upload_video()

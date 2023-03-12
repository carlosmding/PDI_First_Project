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
    
    cap = cv2.VideoCapture("Video_MAS_v2.mp4") 
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


def close_operation(frame):
  # Realiza las operaciones de dilatacion y erodación de una imagen 
  # con el objetivo de obtener la esfera del péndulo para luego calcular el centro de masa  
  
  imagen_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  var, imagen_B = cv2.threshold(imagen_gray,120,255, cv2.THRESH_BINARY)
  element=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

  #Se realiza el proceso de dilatación y erosión 3 veces cada una
  dilate_1=cv2.dilate(imagen_B, element)
  dilate_2=cv2.dilate(dilate_1, element)
  dilate_3=cv2.erode(dilate_2,element)
  erode_1=cv2.erode(dilate_3,element)
  erode_2=cv2.erode(erode_1,element)
  erode_3=cv2.erode(erode_2,element)
  imagen_final=cv2.erode(erode_3,element)
  
  return imagen_final

def find_mass_center(frame):
  # Halla el centro de masa de una imagen luego de realizar las operaciones básicas con close_operation()
  # Se reduce el tamaño de la imagen para que solo tenga en cuenta filas desde 0 hasta la 500 y todas las columnas
  # Retorna las posiciones en x, y del centro de masa
  
  imagen = frame[:500,:]
  th=close_operation(imagen)
  contornos, arch = cv2.findContours(th, 1, 2)
  M=cv2.moments(contornos[0])
  cx = int(M['m10']/M['m00'])
  cy = int(M['m01']/M['m00'])
  return cx, cy

def find_all_center_mass(frames):
  # Crea un vector con todos los centros de masas 
  centers_mass = []
  for frame in frames:
    centers_mass.append(find_mass_center(frame))
  return centers_mass

def paint_vel_ace(imag, vel, ac, cx2, cy2):
  #Retorna una imagen con el texto de la velocidad y acelaración calculada 
  font = cv2.FONT_ITALIC
  tamañoLetra = 0.5
  colorLetra = (0,0,255)
  grosorLetra = 1
  texto=f"v={vel} cm/s a={ac} cm/s2"
  cv2.putText(imag, texto, (cx2, cy2-50), font, tamañoLetra, colorLetra, grosorLetra)
  return imag

def calculate_vel(x1, x2, y1, y2, fds):
  # Calcula la velocidad con dos vectores de posiciones (x,y)
  # Convertir pixeles a cms, cada 8 frames es un cms
  vel=(np.sqrt(((x2-x1)/8)**2 + ((y2-y1)/8)**2))/(1/fds)
  return vel.round(2)

def upload_video_and_paint():
    #Carga video de prueba desde el Drive
    cap = cv2.VideoCapture("Video_MAS_v2.mp4") 
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
     
    # Read until video is completed
    # Se cargan listas para velocidades y aceleraciones
    velocities=[]
    acelarations=[]

    fds = cap.get(cv2.CAP_PROP_FPS) #frames por segundo
    first_time=True

    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        # Display the resulting frame
        
        # la primera vez no va a dibujar ni cm ni velocidad ni acelaración porque no tiene valores iniciales
        if first_time:
          cx1, cy1 = find_mass_center(frame)
          cm =cv2.circle(frame,(cx1,cy1), 2,(0,0,255),-1)
          vel1=0
          first_time=False # se cambia bandera

        else:
          cx2, cy2 = find_mass_center(frame) # Halla el centro de masa
          cm =cv2.circle(frame,(cx2,cy2), 2,(0,0,255),-1) # Dibuja el centro de masa
          
          # Calcular velocidad y acelaración y se guardan en sus listas
          vel =calculate_vel(cx1,cx2,cy1,cy2,fds)
          velocities.append(vel)
          ac= ((np.abs(vel1-vel))/fds).round(2)
          acelarations.append(ac)
        
          # Se inicializan los parametros de método putText
          imag = paint_vel_ace(cm, vel, ac, cx2, cy2)
          cv2.imshow("Velocidad y Aceleracion del Centro de Masa", imag)

          #Se actualizan variables
          cx1 = cx2
          cy1 = cy2
          vel1 = vel
        
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
    return fds, velocities, acelarations

def minimus(centers_mass):
  # Halla los valores minimo de variable x y el máximo de la variable y
  min_x=centers_mass[0][0]
  max_y=centers_mass[0][1]
  for point in centers_mass:
    if point[0]< min_x:
      min_x=point[0]
    elif point[1]> max_y:
      max_y=point[1]
  return min_x, max_y

def transforms_var_x(centers):
  # Transforma las variables usando los minimos y máximos
  # Dado que 8 frames son un cms, se hace la conversión a cms
  
  min_x, max_y =minimus(centers)
  var_x=[]
  var_y=[]

  for point in centers:
      var_x.append((point[0]- min_x)/8)
      var_y.append(np.abs(max_y - point[1] )/8)
  
  return var_x, var_y

def graphic_position_x(var, fds):
  # Dibuja gráfica de posición en x vs tiempo
  tiempo=list(i/fds for i in range(len(var)))
  plt.plot(tiempo,var, color="g")
  plt.xlabel('Time (s)')
  plt.ylabel('Posición en X (cms)')
  plt.title('Posición en X del Centro de Masa')
  plt.grid()
  plt.show()

def graphic_position_y(var, fds):
  # Dibuja gráfica de posición en y vs tiempo
  tiempo=list(i/fds for i in range(len(var)))
  plt.plot(tiempo,var, color="g")
  plt.xlabel('Time (s)')
  plt.ylabel('Posición en Y (cms)')
  plt.title('Posición en Y del Centro de Masa')
  plt.grid()
  plt.show()

def graphic_vel_time(vel, fds):
  # Dibuja gráfica de velocidad vs tiempo
  tiempo=list(i/fds for i in range(len(vel)))
  plt.plot(tiempo,vel, color="g")
  plt.xlabel('Time (s)')
  plt.ylabel('Velocidad (cms/s)')
  plt.title('Velocidad del Centro de Masa')
  plt.grid()
  plt.show()
  
def graphic_ace_time(ace, fds):
  # Dibuja gráfica de aceleración vs tiempo
  tiempo=list(i/fds for i in range(len(ace)))
  plt.plot(tiempo,ace, color="g")
  plt.xlabel('Time (s)')
  plt.ylabel('Aceleración (cms/s2)')
  plt.title('Aceleración del Centro de Masa')
  plt.grid()
  plt.show()

def main():
    frames, fds=upload_video()
    centers_mass = find_all_center_mass(frames)
    var_x, var_y = transforms_var_x(centers_mass)
    graphic_position_x(var_x, fds)
    graphic_position_y(var_y, fds)
     
    fds, vel, ace = upload_video_and_paint()
    graphic_vel_time(vel, fds)
    graphic_ace_time(ace, fds)
    
    #Con la regla se hace el ajuste, se evidencia que un cms equivale a 8 frames
    frame=frames[0]
    foto=frame[620:,19:27]
    cv2.imshow("Escala",foto)
    cv2.destroyAllWindows()

main()


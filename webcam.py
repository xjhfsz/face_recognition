import cv2

webcam = cv2.VideoCapture(0)  # 0 = câmera padrão

while True:
    ret, frame = webcam.read()
    
    if not ret:
        break
    
    # Converter para cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Mostrar ambos os frames
    cv2.imshow('Webcam Colorida', frame)
    # cv2.imshow('Webcam Cinza', cinza)
    
    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
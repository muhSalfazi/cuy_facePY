import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml")

camera = cv2.VideoCapture(0)

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah dari RGB ke Grayscale
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    return faces

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def drawer_box(frame):
    faces = face_detection(frame)
    for (x, y, w, h) in faces:  # Perbaikan sintaks untuk loop
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

def main():
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        cv2.imshow("cuyFace AI", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Perbaikan penulisan cv2.waitKey()
            close_window()

if __name__ == '__main__':
    main()

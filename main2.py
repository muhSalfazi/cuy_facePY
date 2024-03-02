import cv2

# Memuat kaskade (cascade)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Fungsi untuk mendeteksi senyuman
def detect_smile(gray, frame):
    # Mendeteksi wajah dengan parameter yang disesuaikan
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # Menggambar persegi panjang di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Mengekstrak region of interest (ROI) di mana senyuman diharapkan
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Mendeteksi senyuman dalam ROI wajah
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        # Jika tidak ada senyuman yang terdeteksi, tandai sebagai tidak bahagia
        if len(smiles) == 0:
            cv2.putText(frame, 'Tidak Bahagia', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        for (sx, sy, sw, sh) in smiles:
            # Menggambar persegi panjang di sekitar senyuman
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            cv2.putText(frame, 'Senyum', (x + sx, y + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 200, 10), 2)
    return frame

# Menangkap video dari webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Menangkap frame per frame
    ret, frame = video_capture.read()
    # Mengonversi frame menjadi skala keabuan
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Mendeteksi senyuman
    canvas = detect_smile(gray, frame)
    # Menampilkan frame hasil
    cv2.imshow('face_UHUY', canvas)
    # Jika 'q' ditekan, hentikan perulangan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan tangkapan
video_capture.release()
cv2.destroyAllWindows()

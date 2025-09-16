import os
import cv2
import numpy as np
import time
from threading import Thread
import tensorflow as tf

# Video akışını yönetmek için VideoStream sınıfı
class VideoStream:
    def __init__(self, resolution=(255, 255), framerate=30):
        self.stream = cv2.VideoCapture(0)  # Web kamerasını aç
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])  # Genişlik ayarı
        ret = self.stream.set(4, resolution[1])  # Yükseklik ayarı
        (self.grabbed, self.frame) = self.stream.read()  # İlk kareyi oku
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()  # Güncelleme işlemi için bir iş parçacığı başlat
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()  # Akış durdurulursa kaynakları serbest bırak
                return
            (self.grabbed, self.frame) = self.stream.read()  # Yeni kareleri oku

    def read(self):
        return self.frame  # Son alınan kareyi döndür

    def stop(self):
        self.stopped = True  # Akışı durdur

# Modelin ve etiket dosyalarının tanımlanması
MODEL_NAME = "C:\\TFLite_model ALYA3\\custom_model_lite\\saved_model"
GRAPH_NAME = "C:\\TFLite_model ALYA3\\custom_model_lite\\detect.tflite"
LABELMAP_NAME = "C:\\TFLite_model ALYA3\\custom_model_lite\\labelmap.txt"
min_conf_threshold = 0.95  # Minimum güven eşiği
imW, imH = 1280, 720  # Görüntü boyutları
use_TPU = False  # TPU kullanımı

# Geçerli çalışma dizinini al
CWD_PATH = os.getcwd()

# Model dosyasının yolu
PATH_TO_CKPT = os.path.join(MODEL_NAME, GRAPH_NAME)

# Etiket haritası dosyasının yolu
PATH_TO_LABELS = os.path.join(MODEL_NAME, LABELMAP_NAME)

# TensorFlow Lite modelini yükle ve tensor'ları tahsis et
interpreter = tf.lite.Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Etiket haritasını yükle
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]  # Etiketleri listeye aktar
if labels[0] == '???':
    del(labels[0])  # İlk etiket geçersizse sil

# Model detaylarını al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]  # Yükseklik
width = input_details[0]['shape'][2]  # Genişlik

# Modelin kayan nokta kullanıp kullanmadığını kontrol et
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5  # Giriş normalizasyonu için ortalama
input_std = 127.5   # Giriş normalizasyonu için standart sapma

outname = output_details[0]['name']

# Çıktı indekslerini ayarla
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Kare hızı hesaplama için başlangıç
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Video akışını başlat
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)  # Kamera açılması için bekle

# Ana döngü
while True:
    t1 = cv2.getTickCount()  # Zamanı al

    frame1 = videostream.read()  # Görüntüyü oku

    frame = frame1.copy()  # Görüntüyü kopyala
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye çevir
    frame_resized = cv2.resize(frame_rgb, (width, height))  # Görüntüyü yeniden boyutlandır
    input_data = np.expand_dims(frame_resized, axis=0)  # Girdi veri boyutunu ayarla

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std  # Normalizasyon

    interpreter.set_tensor(input_details[0]['index'], input_data)  # Girdiyi modele ayarla
    interpreter.invoke()  # Modeli çalıştır

    # Çıktıları al
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    # Tespit edilen nesneler için döngü
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):  # Güven eşiğini kontrol et
            ymin = int(max(1, (boxes[i][0] * imH)))  # Alt sınır
            xmin = int(max(1, (boxes[i][1] * imW)))  # Sol sınır
            ymax = int(min(imH, (boxes[i][2] * imH)))  # Üst sınır
            xmax = int(min(imW, (boxes[i][3] * imW)))  # Sağ sınır

            # Sınırlara dikdörtgen çiz
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Etiket ve güveni yaz
            class_id = int(classes[i])
            label = labels[class_id] if class_id < len(labels) else 'N/A'
            cv2.putText(frame, f"{label}: {scores[i]:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # FPS değerini yaz
    cv2.putText(frame, f"FPS: {frame_rate_calc:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Nesne Tespiti', frame)  # Görüntüyü göster

    if cv2.waitKey(1) == ord('q'):  # 'q' tuşuna basılırsa çık
        break

    t2 = cv2.getTickCount()  # Zamanı al
    time1 = (t2 - t1) / freq  # Geçen süreyi hesapla
    frame_rate_calc = 1 / time1  # FPS'yi hesapla

cv2.destroyAllWindows()  # Pencereleri kapat
videostream.stop()  # Video akışını durdur

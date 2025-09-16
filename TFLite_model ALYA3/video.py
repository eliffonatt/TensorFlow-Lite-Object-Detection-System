import os
import tensorflow as tf
import cv2
import numpy as np

# Gerekli kütüphaneleri tanımla ve giriş argümanlarını ayarla
MODEL_NAME = "C:\TFLite_model ALYA3\custom_model_lite\saved_model"
GRAPH_NAME = "C:\TFLite_model ALYA3\custom_model_lite\detect.tflite"
PATH_TO_CKPT = os.path.join(MODEL_NAME, GRAPH_NAME)
LABELMAP_NAME= "C:\TFLite_model ALYA3\custom_model_lite\labelmap.txt"
min_conf_threshold = 0.5
imW, imH = 1280, 720
use_TPU = False

# Modeli yükle ve tensörleri ayarla
interpreter = tf.lite.Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()  

# Geçerli çalışma dizinini al
CWD_PATH = os.getcwd()

# Video dosyasının yolu
VIDEO_PATH = os.path.join(CWD_PATH,"C:\TFLite_model ALYA3\deneme01.mp4")

# Nesne tespiti için kullanılan modelin .tflite dosyasının yolu
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Etiket haritası dosyasının yolu
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Etiket haritasını yükle
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Eğer COCO "başlangıç modeli" kullanıyorsanız (https://www.tensorflow.org/lite/models/object_detection/overview),
# ilk etiket '???' olacaktır, bu yüzden silinmesi gerekir.
if labels[0] == '???':
    del(labels[0])

# Tensörleri yeniden ayarla
interpreter.allocate_tensors()

# Model detaylarını al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Kayan nokta model olup olmadığını kontrol et
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Çıktı katmanı ismini kontrol et, çünkü TF2 ve TF1 modellerinin çıktı sıralaması farklıdır.
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # Bu bir TF2 modeli
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # Bu bir TF1 modeli
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Video dosyasını aç
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

while(video.isOpened()):

    # Çerçeveyi al ve beklenen şekle [1xHxWx3] yeniden boyutlandır
    ret, frame = video.read()
    if not ret:
      print('Videonun sonuna gelindi!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Eğer kayan nokta modeli kullanıyorsanız, piksel değerlerini normalize et
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Gerçek algılama işlemini gerçekleştir, modeli görüntü ile çalıştır
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Algılama sonuçlarını al
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Tespit edilen nesnelerin sınır kutusu koordinatları
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Tespit edilen nesnelerin sınıf indeksleri
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Tespit edilen nesnelerin güven skorları

    # Non-Maximum Suppression (NMS) uygula
    selected_indices = tf.image.non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size=10,  # NMS tarafından seçilen maksimum kutu sayısı
        iou_threshold=0.5,   # NMS için IoU eşik değeri
        score_threshold=min_conf_threshold  # NMS için minimum güven skoru eşiği
    )

    selected_boxes = tf.gather(boxes, selected_indices)
    selected_classes = tf.gather(classes, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)

    # Algılama sonuçlarını döngüye al ve güven skoru eşiği üzerinde olan nesneler için tespit kutusu çiz
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Sınır kutusu koordinatlarını al ve kutuyu çiz
            # Yorumlayıcı bazen görüntü boyutlarının dışında koordinatlar döndürebilir, bu yüzden bunları max() ve min() ile sınırlamak gerekir
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

            # Etiket çiz
            object_name = labels[int(classes[i])] # Etiket dizisinden sınıf indeksini kullanarak nesne adını al
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Örnek: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Yazı tipi boyutunu al
            label_ymin = max(ymin, labelSize[1] + 10) # Etiketi pencerenin üst kısmına çok yakın yapmamaya dikkat et
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Etiket için beyaz kutu çiz
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Etiket metnini çiz

    # Sonuçlar çerçeve üzerinde çizildi, şimdi bunları ekranda gösterme zamanı
    cv2.imshow('Nesne algılama', frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) == ord('q'):
        break

# Temizlik yap
video.release()
cv2.destroyAllWindows()

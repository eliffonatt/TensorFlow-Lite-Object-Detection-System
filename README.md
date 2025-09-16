# TensorFlow Lite Nesne Tespit Sistemi

# (TensorFlow Lite Object Detection System)

## 📌 Proje Hakkında | About the Project
**🇹🇷 Türkçe**

-Bu proje, silahlı insansız hava aracı (SİHA) sistemleri için görüntü işleme tabanlı nesne tespiti amacıyla geliştirilmiştir. Projede iki farklı senaryo desteklenmektedir:

**Gerçek Zamanlı Tespit:** Canlı kamera görüntüsü üzerinden anlık nesne tespiti. 

**Video Dosyası Üzerinden Tespit:** Kayıtlı video üzerinde nesne tespiti.

-Model, yalnızca belirlenen sınıfları (dost unsur ve düşman unsur) algılar ve diğer nesneleri görmezden gelir.

## Projede kullanılan ana teknolojiler:

-TensorFlow Lite nesne tespit modeli

-OpenCV ile video ve kamera işleme

-Maksimum Olmayan Bastırma (NMS) ile daha doğru sonuçlar

-Yalnızca seçilen sınıflar için kutulama ve etiketleme

**🇬🇧 English**

-This project is developed for image processing-based object detection in Unmanned Combat Aerial Vehicles (UCAVs). It supports two different scenarios:

**Real-Time Detection:** Object detection from live camera input.

**Video File Detection:** Object detection on pre-recorded video files.

-The model detects only specific classes (friendly unit and enemy unit) and ignores all other objects.

## Key technologies used:

-TensorFlow Lite object detection model

-OpenCV for video and camera processing

-Non-Maximum Suppression (NMS) for more accurate results

-Bounding boxes and labeling only for selected classes

# ⚙️ Kurulum | Installation
**🇹🇷 Türkçe**

-Gerekli bağımlılıkları yükleyin:
```bash
pip install tensorflow opencv-python numpy
```

-Model dosyalarını (.tflite, labelmap.txt) ve test videosunu proje dizinine yerleştirin.

-video_detection.py veya realtime_detection.py dosyalarından ihtiyacınıza uygun olanı çalıştırın:
```bash
python video_detection.py
```

veya
```bash
python realtime_detection.py
```
**🇬🇧 English**

-Install the required dependencies:
```bash
pip install tensorflow opencv-python numpy
```

-Place the model files (.tflite, labelmap.txt) and the test video in the project directory.

-Run the script depending on your use case:
```bash
python video_detection.py
```

or
```bash
python realtime_detection.py
```

## 📊 Çalışma Mantığı | How It Works
**🇹🇷 Türkçe**

-Kamera veya video kaynağından kareler alınır.

-Kareler modele uygun şekilde yeniden boyutlandırılır.

-Model çalıştırılır ve sınıf skorları + sınırlayıcı kutular elde edilir.

-NMS uygulanarak en doğru tespitler seçilir.

-Sadece dost unsur ve düşman unsur sınıfları ekrana çizilir.

**🇬🇧 English**

-Frames are captured from camera or video input.

-Frames are resized to match the model’s input.

-The model runs and returns class scores + bounding boxes.

-NMS selects the most accurate detections.

-Only friendly unit and enemy unit classes are drawn on screen.

##📌 Notlar | Notes

-Türkçe etiketlerde dost unsur → Friendly Unit, düşman unsur → Enemy Unit olarak çevrilmiştir.

-desired_classes listesine yeni sınıf indeksleri eklenerek farklı nesneler de tespit edilebilir.

-Bu proje, SİHA sistemleri için nesne tespit altyapısı geliştirmek amacıyla hazırlanmıştır.


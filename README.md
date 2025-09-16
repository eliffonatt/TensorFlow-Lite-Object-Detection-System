# TensorFlow Lite Nesne Tespit Sistemi

# (TensorFlow Lite Object Detection System)

# 📌 Proje Hakkında | About the Project
** 🇹🇷 Türkçe **

-Bu proje, silahlı insansız hava aracı (SİHA) sistemleri için görüntü işleme tabanlı nesne tespiti amacıyla geliştirilmiştir. Projede iki farklı senaryo desteklenmektedir:

** Gerçek Zamanlı Tespit: ** Canlı kamera görüntüsü üzerinden anlık nesne tespiti. 

** Video Dosyası Üzerinden Tespit: ** Kayıtlı video üzerinde nesne tespiti.

-Model, yalnızca belirlenen sınıfları (dost unsur ve düşman unsur) algılar ve diğer nesneleri görmezden gelir.

# Projede kullanılan ana teknolojiler:

-TensorFlow Lite nesne tespit modeli

-OpenCV ile video ve kamera işleme

-Maksimum Olmayan Bastırma (NMS) ile daha doğru sonuçlar

-Yalnızca seçilen sınıflar için kutulama ve etiketleme

** 🇬🇧 English **

-This project is developed for image processing-based object detection in Unmanned Combat Aerial Vehicles (UCAVs). It supports two different scenarios:

** Real-Time Detection: ** Object detection from live camera input.

** Video File Detection: ** Object detection on pre-recorded video files.

-The model detects only specific classes (friendly unit and enemy unit) and ignores all other objects.

# Key technologies used:

-TensorFlow Lite object detection model

-OpenCV for video and camera processing

-Non-Maximum Suppression (NMS) for more accurate results

-Bounding boxes and labeling only for selected classes

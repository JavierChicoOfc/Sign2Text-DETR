# Using DETR Transformers for Basic Sign Language Estimation
More of a deep dive into training a DETR model from scratch and all the nuaces with getting object detection running. It was...fun. Anyway, here's a full walkthrough from me to you. Let me know how you go!

# To run

# For unicode characters generated in logger

chcp 65001 > $null
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

## solucionar opencv-headless error

Opencv-headless es una dependencia transitiva de alguna otra librería que se está usando en el proyecto. Esta versión de OpenCV no incluye soporte para interfaces gráficas, lo que causa problemas al intentar usar funciones que requieren una GUI, como `cv2.imshow()`.

Esta dependencia transitiva está causada por Albumentations.
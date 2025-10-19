# 🐍 Sign2Text - DETR

_Translate sign language into written words using machine learning._ 

---

**Note: this repo does not have git history beacause it was migrated from a private repo**

## 📖 Overview

**Sign2Text** is a project designed to **build a complete pipeline** for training and deploying a model that **translates sign language gestures into text.**  

**This repository aims to:**
- Provide a clean training pipeline for sign language recognition.  
- Offer tools for data preprocessing, training, and evaluation.  
- Demonstrate a minimal prototype suitable for training a sign2text model

The system is structured into **four main components**:

1. **Video**: captures video from camera using opencv  
2. **Generate Dataset**: orchestrates collection and labeling of gesture samples
3. **Training**: trains a model on the generated dataset
4. **Inference**: runs real-time sign recognition from live camera feed

---

## ⚙️ Installation

**Prerequisites:**  
- Python 3.12
- [uv](https://github.com/astral-sh/uv) (for dependency management)

Install dependencies
```
uv sync
```

## Usage

1. **Plan your signs:** Decide the set of signs (classes) you want to record and train the model with. Modify them in the [src/configuration/config.json](./src/configuration/config.json) file under the `classes` key. You must provide a color for each class for visualization during inference.
imension of each LSTM sample).

2. **Steps to train and execute in realtime the project:**
   1. **Record images** for each sign class using your webcam:
    
    ```bash
    uv run python src/main.py --script record_images --source webcam
    ```
    2. **Label the recorded images** using [Label Studio](https://labelstud.io/):
     
     ```bash
    uv run label-studio
    ```
    3. **Train the model** on the labeled dataset:
    
    ```bash
    uv run uv run python src/main.py --script train
    ```
    4. **Test the model** on test data or live webcam feed:
        
     ```bash
    uv run python src/main.py --script test --mode test
    ```
    5. **Run real-time sign recognition** from your webcam:
        
    ```bash
    python src/main.py --script realtime
    ```

3. **The main components can be accessed form main** in [main.py](./main.py):
    
    ```bash
    uv run python src/main.py --script <component> [additional arguments]
    ```

    Where `<component>` can be one of:
    - `record_images`: Capture images for each sign class using your webcam.
    - `train`: Train the sign language recognition model.
    - `test`: Run inference on test data or live webcam feed.
     - Use `--mode train` for inference on training data.
     - Use `--mode test` for inference on test data.
    - `realtime`: Real-time sign recognition from webcam.

## Credits

Credits are to mention the people who contributed to the project.

| GitHub                                                                 | LinkedIn                                                                                          |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| [Almudena Zhou Ramirez](https://almudenazhou.github.io/)               | [linkedin.com/in/almudena-zhou-ramirez-lopez](https://www.linkedin.com/in/almudena-zhou-ramirez-lopez/) |
| [Javier Chico García](https://github.com/JavierChicoOfc)               | [linkedin.com/in/javier-chico-garcía-ofc](https://www.linkedin.com/in/javier-chico-garc%C3%ADa-ofc/) |
| [Jose Manuel Pinto Lozano](https://github.com/JoseManuelPintoLozano)   | [linkedin.com/in/josemanuelpintolozano](https://www.linkedin.com/in/josemanuelpintolozano/)         |


This project was inspired by [nicknochnack repository](git clone https://github.com/nicknochnack/SignDETR)

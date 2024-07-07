# 🖐️ Sign Language Recognition System

![Sign Language Recognition](https://github.com/AaditSingh/Sign-Language-Recognition-System/blob/6bb7ed21b8499b042e8cebb9a16fc7a5d05fbb56/asl_sign.png)

## 📚 Project Overview

Sign Language Recognition System aims to bridge the communication gap between speech-impaired or deaf individuals and those who do not understand sign language. This system recognizes hand signs and gestures, converting them into readable text to facilitate communication.

## 🎯 Motivation

According to the 2011 Indian census, roughly 1.3 million people have hearing impairments, with the National Association of the Deaf estimating this number to be around 18 million. This significant number of individuals highlights the need for a system that can help them communicate effectively with those who do not understand sign language.

## ✨ Features

- **Sign Language Interpreter/Translator:** Facilitates communication between deaf-dumb and normal individuals without needing a human translator.
- **Sign Language Learning Tool:** Assists in teaching sign language to deaf-dumb children.
- **Human-Computer Interface:** Allows deaf-dumb individuals to interact with computer systems.
- **Interfacing in Virtual Environment:** Uses gestures to control various systems, including computers, music systems, virtual games, home appliances, and medical equipment.

## 📊 Dataset

We are using the “Sign Language MNIST” dataset, a public-domain, free-to-use dataset available on [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist). This dataset contains pixel information for around 1,000 images of each of 24 ASL letters, excluding J and Z.

## 🚧 Challenges

- **Real-Time Recognition:** The system needs to work in uncontrolled real-time environments, coping with messy backgrounds, moving objects, diverse lighting conditions, and various users.
- **Indian Sign Language Dataset:** There's a lack of standard datasets for Indian Sign Language (ISL) and issues with standardization.
- **Dynamic Gestures:** Recognizing dynamic hand gestures and differentiating between gestures with similar meanings remains challenging.
- **False Gestures:** Detecting false gestures, such as unwanted gestures between words in a sentence, is still a challenge.
- **Two-Hand Gestures:** Many ISL alphabets use two hands, making recognition difficult.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Sign-Language-Recognition-System.git
   cd Sign-Language-Recognition-System
   ```

2. Install the required dependencies:
   ```bash
   pip install keras
   pip install pandas
   pip install gTTS
   pip install opencv-python
   pip install mediapipe
   pip install numpy

   ```

3. Run the prediction script:
   ```bash
   python prediction.py
   ```

## 🚀 Usage

1. Load the dataset:
   - Ensure the `sign_mnist_train.csv` file is in the project directory.
   
2. Train the model (details can be found in the provided documentation).

3. Use the trained model to recognize sign language gestures through the `prediction.py` script.

## 🤝 Contributing

Contributions are welcome to this project. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## 📚 References

- "The Cognitive, Psychological, and Cultural Impact of Communication Barrier on Deaf Adults". Journal of Communication Disorders, Deaf Studies Hearing Aids 4 (2 2016). doi: 10.4172/2375-4427.1000164.
- Mihir Garimella. “Sign Language Recognition with Advanced Computer Vision”. [Towards Data Science](https://towardsdatascience.com/sign-language-recognition-with-advanced-computer-vision-7b74f20f3442).
- Vivek Bheda and Dianna Radpour. “Using Deep Convolutional Networks for Gesture Recognition in American Sign Language”. CoRR abs/1710.06836 (2017). [arXiv](http://arxiv.org/abs/1710.06836).

  ## 👤 Author
- **Aadit Singh**

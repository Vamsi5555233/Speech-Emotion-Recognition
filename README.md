# Speech Emotion Recognition using Machine Learning (MLP Classifier)

## 📌 Overview  
This project focuses on recognizing human emotions from speech audio using **Machine Learning techniques**.  
By extracting acoustic features such as **MFCCs, Chroma, and Mel Spectrograms** from audio signals, the project builds a classification model that can predict emotions like **Calm, Happy, Fearful, and Disgust**.  

The model is trained using **MLPClassifier (Multi-layer Perceptron)** from scikit-learn.  

---

## 🗂️ Dataset  
- **RAVDESS Dataset** (Ryerson Audio-Visual Database of Emotional Speech and Song)  
- Contains audio samples from professional actors expressing emotions.  
- Only 4 emotions are considered for this project:  
  - Calm  
  - Happy  
  - Fearful  
  - Disgust  

---

## ⚙️ Features Extracted  
Using **Librosa**, the following features are extracted from each `.wav` audio file:  
- **MFCC (Mel Frequency Cepstral Coefficients)** – captures timbral aspects of speech.  
- **Chroma** – represents pitch classes.  
- **Mel Spectrogram** – captures frequency distribution.  

---

## 🧠 Model  
- **MLPClassifier (Multi-layer Perceptron Neural Network)** from scikit-learn.  
- Hyperparameters:  
  - `hidden_layer_sizes=(300,)`  
  - `alpha=0.01`  
  - `batch_size=256`  
  - `learning_rate='adaptive'`  
  - `max_iter=500`  

---

## 📊 Results  
Evaluation metrics achieved on test data:  
- **Accuracy:** ~57%  
- **Precision:** ~63%  
- **Recall:** ~58%  
- **F1 Score:** ~58%  

Confusion Matrix visualization is included to show how well each emotion is classified.  

---

## 🛠️ Installation & Setup  

```bash
# 1. Clone the repository
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition

# 2. Install dependencies (Python 3.8+ recommended)
pip install numpy pandas scikit-learn librosa soundfile matplotlib seaborn

# 3. Run the project
jupyter notebook

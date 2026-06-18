# 🧠 MNIST Handwritten Digit Classification with Neural Networks

This project builds, trains, and evaluates a feed-forward neural network on the MNIST dataset using TensorFlow and Keras. It achieves **98.37% test accuracy** across 10 handwritten digit classes through a structured 13-stage pipeline that covers preprocessing, model design, evaluation, and hyperparameter tuning.

---

## 📊 Project Description

This project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) — 70,000 grayscale images of handwritten digits (0–9) — to train a fully connected neural network. The focus is on:

* Designing a robust feed-forward architecture with regularisation techniques,
* Training with adaptive callbacks to prevent overfitting,
* Evaluating performance through multiple metrics and visualisations,
* Comparing the effect of different hyperparameter configurations.

Key tasks include:

1. Loading and exploring the MNIST dataset,
2. Preprocessing images (flattening and normalisation),
3. Building and training a neural network with BatchNorm and Dropout,
4. Evaluating using accuracy, loss curves, confusion matrix, and classification report,
5. Tuning learning rate and batch size to compare model performance.

---

## 🧠 Features

### 📌 Model Architecture
* Three hidden Dense layers (256 → 128 → 64) with BatchNormalization, ReLU, and Dropout,
* Softmax output layer for 10-class probability prediction,
* Adam optimiser with SparseCategoricalCrossentropy loss.

### 📌 Training
* EarlyStopping to halt training when validation loss stops improving,
* ReduceLROnPlateau to adaptively shrink the learning rate during stagnation,
* 10% validation split monitored every epoch.

### 📌 Evaluation
* Accuracy and loss curves (train vs. validation),
* 10×10 confusion matrix heatmap,
* Per-class precision, recall, and F1-score,
* Visual grid of correct and incorrect predictions with confidence scores.

### 📌 Hyperparameter Tuning
* Three configurations tested across learning rate and batch size,
* Results saved to CSV and visualised as accuracy curves and bar charts.

---

## 📁 Project Structure

* `neural_network.py`: Full 13-stage pipeline — loading, preprocessing, training, evaluation, and tuning,
* `outputs/plots/`: All generated visualisation PNGs,
* `outputs/results/`: Classification report and hyperparameter results CSV,
* `README.md`: Project documentation (this file).

---

## 🚀 Getting Started

### ✅ Prerequisites

* Python 3.8 or higher
* Recommended: VS Code or Cursor

### 🛠️ Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/DataCrafter20/neural-network-mnist.git

# 2. Navigate to the project directory
cd neural-network-mnist

# 3. Install the dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

# 4. Run the pipeline
python neural_network.py
```

> No external dataset download needed — MNIST loads automatically through Keras on first run.

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **98.37 %** |
| Test Loss | 0.0628 |
| Epochs Trained | 25 / 30 (early stop) |
| Total Parameters | 244,554 |

### Hyperparameter Tuning Summary

| Config | Test Accuracy | Test Loss |
|---|---|---|
| LR=0.001 \| BS=64 | 98.07 % | 0.0667 |
| LR=0.001 \| BS=256 | 98.03 % | 0.0692 |
| LR=0.0005 \| BS=128 | 98.19 % | 0.0653 |

---

## 🧩 Future Enhancements

* Replace Dense layers with Convolutional layers (CNN) for higher accuracy,
* Add data augmentation (rotations, shifts) to improve generalisation,
* Extend to other image datasets beyond MNIST.

---

## 👤 Author

**Ndivhuwo Munyai**

* 📧 [nmunyai11@gmail.com](mailto:nmunyai11@gmail.com)
* 🔗 [github.com/DataCrafter20](https://github.com/DataCrafter20)
* 🔗 [linkedin.com/in/ndivhuwo-munyai-390a58337](https://linkedin.com/in/ndivhuwo-munyai-390a58337)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/DataCrafter20/neural-network-mnist/blob/main/LICENSE) file for details.

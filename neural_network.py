# ============================================================
# LEVEL 3 — TASK 3: NEURAL NETWORKS WITH TENSORFLOW / KERAS
# ============================================================

import os
from typing import Any
import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                          # saves plots without a display
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import seaborn             as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,                             # stops when val_loss stops improving
    ReduceLROnPlateau,                         # shrinks LR when training stalls
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# Fix random seeds so results are the same on every run
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# STAGE 1 — SETUP
# ============================================================

def create_folders():
    # Creates output folders if they do not already exist.
    # exist_ok=True means no error is raised if the folder is
    # already there from a previous run.
    os.makedirs("outputs/plots",   exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)


# ============================================================
# STAGE 2 — DATA LOADING
# ============================================================

def load_data():
    # --------------------------------------------------------
    # MNIST ships directly inside Keras so no CSV is needed.
    # Calling load_data() downloads the dataset on the first
    # run and caches it locally for every run after that.
    #
    # The dataset contains 70 000 grayscale images split into:
    #   Training set — 60 000 images the model learns from
    #   Test set     — 10 000 images held back for final scoring
    #
    # Each image is a 28×28 grid of pixels.
    # Each pixel is an integer from 0 (black) to 255 (white).
    # Each label is an integer from 0 to 9 the digit shown.
    # --------------------------------------------------------

    (X_train_raw, y_train), (X_test_raw, y_test) = keras.datasets.mnist.load_data()

    return X_train_raw, y_train, X_test_raw, y_test


def summarise_data(X_train_raw, y_train, X_test_raw, y_test):
    # --------------------------------------------------------
    # Prints a quick overview of what was loaded so we can
    # confirm the shapes, pixel range and class balance
    # before touching the data.
    # --------------------------------------------------------

    print("")
    print("=" * 60)
    print("  STAGE 2 — DATA SUMMARY")
    print("=" * 60)

    print(f"\n  Training images : {X_train_raw.shape[0]:,}  —  shape per image: {X_train_raw.shape[1:]}")
    print(f"  Test images     : {X_test_raw.shape[0]:,}  —  shape per image: {X_test_raw.shape[1:]}")
    print(f"  Pixel range     : {X_train_raw.min()} - {X_train_raw.max()}  (uint8)")
    print(f"  Classes         : {sorted(np.unique(y_train).tolist())}  (digits 0–9)")

    print("\n  --- Samples per Digit (Training Set) ---\n")
    unique, counts = np.unique(y_train, return_counts=True)
    for digit, count in zip(unique, counts):
        print(f"  Digit {digit} : {count:,}")

    print("\n" + "=" * 60)


# ============================================================
# STAGE 3 — DATASET VISUALISATION
# ============================================================

def plot_dataset_overview(X_train_raw, y_train):
    # --------------------------------------------------------
    # Before preprocessing or modelling we want to see the
    # raw data with our own eyes.  
    # This plot shows two things:
    #
    #   Left  — a bar chart of how many examples exist per
    #            digit class.  Roughly equal bars mean the
    #            model will not be biased toward any one class.
    #
    #   Right — a 2-row grid of sample images, two per digit,
    #            so we can see what the raw pixel data looks
    #            like and confirm the labels are correct.
    # --------------------------------------------------------

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("MNIST Dataset — Class Distribution & Sample Images",
                 fontsize=14, fontweight="bold", y=1.02)

    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- Left panel : class-frequency bar chart ---
    ax_bar = fig.add_subplot(outer[0])

    unique, counts = np.unique(y_train, return_counts=True)

    bars = ax_bar.bar(
        unique, counts,
        color="#2563EB",
        edgecolor="white",
        linewidth=0.7,
    )
    ax_bar.set_title("Training Set — Samples per Digit", fontsize=11)
    ax_bar.set_xlabel("Digit Class")
    ax_bar.set_ylabel("Number of Images")
    ax_bar.set_xticks(unique)

    # Place the exact count above every bar
    for bar, count in zip(bars, counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,  # horizontal centre of bar
            bar.get_height() + 40,               # just above the bar top
            f"{count:,}",
            ha="center", fontsize=7.5,
        )

    # --- Right panel : 2 sample images per digit (2 rows × 10 columns) ---
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 10,
        subplot_spec=outer[1],
        wspace=0.06,
        hspace=0.06,
    )

    for digit in range(10):
        indices = np.where(y_train == digit)[0]      # all indices for this digit
        for row in range(2):                          # show 2 examples per digit
            ax = fig.add_subplot(inner[row, digit])
            ax.imshow(X_train_raw[indices[row]], cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(str(digit), fontsize=9, pad=2)

    plt.tight_layout()

    output_path = "outputs/plots/01_dataset_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved: {output_path}")


# ============================================================
# STAGE 4 — PREPROCESSING
# ============================================================

def preprocess(X_train_raw, y_train, X_test_raw, y_test):
    # --------------------------------------------------------
    # Two operations are applied to the images before training.
    #
    # 1. Flatten
    #    Dense fully connected layers expect a 1-D feature
    #    vector, not a 2-D grid.  We reshape each 28×28 image
    #    into a single vector of 784 values.
    #
    # 2. Normalise
    #    Pixel values run from 0 to 255.  Dividing by 255
    #    scales them to the range [0.0, 1.0].  Smaller inputs
    #    keep the gradients stable during back-propagation and
    #    let the network converge faster.
    #
    # Labels are left as plain integers 0–9.
    # SparseCategoricalCrossentropy handles integer labels
    # directly no one-hot encoding is needed.
    # --------------------------------------------------------

    X_train = X_train_raw.reshape(-1, 784)            # (60000, 28, 28) -> (60000, 784)
    X_test  = X_test_raw.reshape(-1, 784)             # (10000, 28, 28) -> (10000, 784)

    X_train = X_train.astype("float32") / 255.0       # scale to [0.0, 1.0]
    X_test  = X_test.astype("float32")  / 255.0

    print(f"\n  X_train : {X_train.shape}  |  range [{X_train.min():.1f}, {X_train.max():.1f}]")
    print(f"  X_test  : {X_test.shape}   |  range [{X_test.min():.1f},  {X_test.max():.1f}]")

    return X_train, y_train, X_test, y_test


# ============================================================
# STAGE 5 — BUILD MODEL
# ============================================================

def build_model(learning_rate=0.001):
    # --------------------------------------------------------
    # Building a feed-forward fully connected neural network
    # using Keras Sequential layers stacked one after another.
    #
    # Architecture
    # ---------------------------------------------
    # Input          784 features  (one per pixel)
    # Hidden layer 1 Dense(256) -> BatchNorm -> ReLU -> Dropout(0.30)
    # Hidden layer 2 Dense(128) -> BatchNorm -> ReLU -> Dropout(0.30)
    # Hidden layer 3 Dense( 64) -> BatchNorm -> ReLU -> Dropout(0.20)
    # Output         Dense( 10) -> Softmax
    # ---------------------------------------------
    #
    # Why these design choices?
    #
    # BatchNormalization
    #   Normalises each layer's outputs to have near-zero mean
    #   and unit variance. This speeds up learning and makes
    #   the network less sensitive to the initial learning rate.
    #
    # ReLU activation
    #   f(x) = max(0, x). It keeps positive values unchanged
    #   and zeroes out negatives. It is the standard default
    #   for hidden layers because it is fast and rarely causes
    #   the vanishing gradient problem.
    #
    # Dropout
    #   Randomly zeroes a fraction of neurons during each
    #   training step. This prevents neurons from co-adapting
    #   and forces the network to learn multiple independent
    #   ways to recognise each digit, which reduces overfitting.
    #
    # Softmax output
    #   Converts the 10 raw output scores into probabilities
    #   that sum to 1.0 easy to read as per-class confidence.
    #
    # Adam optimiser
    #   Adapts the learning rate for each parameter individually
    #   using estimates of the first and second moments of the
    #   gradients. It is the most reliable default optimiser
    #   for feed-forward networks.
    #
    # SparseCategoricalCrossentropy loss
    #   Measures how surprised the model is by the correct
    #   class. "Sparse" means we pass integer labels directly
    #   instead of one-hot encoded vectors.
    # --------------------------------------------------------

    model = keras.Sequential(name="MNIST_FeedForward")

    model.add(layers.Input(shape=(784,)))              # declares input shape explicitly

    # Hidden layer 1
    model.add(layers.Dense(256))                       # 256 neurons, no activation yet
    model.add(layers.BatchNormalization())             # normalise this layer's outputs
    model.add(layers.Activation("relu"))               # apply ReLU after normalisation
    model.add(layers.Dropout(0.30))                   # randomly zero 30 % of neurons

    # Hidden layer 2
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.30))

    # Hidden layer 3
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.20))                   # slightly less dropout narrower layer

    # Output layer
    model.add(layers.Dense(10, activation="softmax")) # one probability per digit class

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# STAGE 6 — TRAIN MODEL
# ============================================================

def train_model(model, X_train, y_train, batch_size=128, epochs=30):
    # --------------------------------------------------------
    # We fit the model to the training data using two callbacks
    # that improve training quality and prevent overfitting.
    #
    # Validation split
    #   10 % of the training data is held back each epoch as a
    #   validation set.  The model never learns from it — we
    #   use it only to check whether the model is generalising
    #   or just memorising the training examples.
    #
    # EarlyStopping
    #   Monitors val_loss after each epoch.  If it has not
    #   improved for 5 consecutive epochs, training stops and
    #   the weights are restored to the best checkpoint found.
    #   This prevents the model from overfitting by running too
    #   many epochs.
    #
    # ReduceLROnPlateau
    #   If val_loss has not improved for 3 epochs, the learning
    #   rate is halved (multiplied by 0.5).  A smaller learning
    #   rate lets the optimiser take more careful steps near a
    #   good solution instead of overshooting it.
    # --------------------------------------------------------

    early_stop = EarlyStopping(
        monitor="val_loss",          # watch validation loss, not training loss
        patience=5,                  # stop after 5 epochs with no improvement
        restore_best_weights=True,   # roll back to the epoch with the lowest val_loss
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,                  # new_lr = current_lr × 0.5
        patience=3,                  # trigger after 3 stagnant epochs
        min_lr=1e-6,                 # never drop below this floor
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.10,       # 10% of training data -> validation set
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    epochs_ran = len(history.history["loss"])
    print(f"\n  Training complete — ran {epochs_ran} / {epochs} epochs.")

    return history


# ============================================================
# STAGE 7 — TRAINING CURVES
# ============================================================

def plot_training_curves(history):
    # --------------------------------------------------------
    # After training we plot accuracy and loss for both the
    # training set and the validation set over every epoch.
    #
    # What to look for:
    #   Both curves rising together -> model is learning well
    #   Train accuracy >> Val accuracy -> overfitting
    #   Both curves flat from the start -> learning rate too low
    #   Loss spiking or diverging -> learning rate too high
    #
    # The gap between train and val curves tells us how much
    # the model is generalising versus memorising the data.
    # --------------------------------------------------------

    epoch_range = range(1, len(history.history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — Accuracy & Loss", fontsize=14, fontweight="bold")

    # --- Accuracy ---
    ax1.plot(epoch_range, history.history["accuracy"],
             color="#2563EB", linewidth=2, label="Train")
    ax1.plot(epoch_range, history.history["val_accuracy"],
             color="#DC2626", linewidth=2, linestyle="--", label="Validation")
    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # --- Loss ---
    ax2.plot(epoch_range, history.history["loss"],
             color="#2563EB", linewidth=2, label="Train")
    ax2.plot(epoch_range, history.history["val_loss"],
             color="#DC2626", linewidth=2, linestyle="--", label="Validation")
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_path = "outputs/plots/02_training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved: {output_path}")


# ============================================================
# STAGE 8 — EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    # --------------------------------------------------------
    # We evaluate the trained model on the held-out test set —
    # 10 000 images the model has never seen during training.
    #
    # model.evaluate() returns the test loss and test accuracy.
    #
    # model.predict() returns a probability matrix of shape
    # (10000, 10) one row per image, one column per digit.
    # np.argmax picks the column with the highest probability,
    # giving us the model's predicted digit for each image.
    # --------------------------------------------------------

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n  Test Accuracy : {test_acc * 100:.2f} %")
    print(f"  Test Loss     : {test_loss:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)    # (10000, 10) probability matrix
    y_pred       = np.argmax(y_pred_probs, axis=1)     # (10000,) predicted digit per image

    return test_loss, test_acc, y_pred, y_pred_probs


# ============================================================
# STAGE 9 — CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(y_test, y_pred):
    # --------------------------------------------------------
    # A confusion matrix shows us exactly where the model is
    # making mistakes.  Each row is a true digit class and each
    # column is a predicted digit class.
    #
    # The diagonal cells top-left to bottom-right show
    # correctly classified images we want these to be as
    # large as possible.
    #
    # Off-diagonal cells reveal which digits get confused with
    # each other.  For example, a large number in row 4 /
    # column 9 would mean the model often mistakes 4s for 9s,
    # which makes intuitive sense given their similar shape.
    # --------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)              # 10×10 matrix of counts

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,                                    # prints count inside each cell
        fmt="d",                                       # integer format
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("True Digit")

    plt.tight_layout()

    output_path = "outputs/plots/03_confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved: {output_path}")


# ============================================================
# STAGE 10 — PREDICTION EXAMPLES
# ============================================================

def plot_prediction_examples(X_test, y_test, y_pred, y_pred_probs):
    # --------------------------------------------------------
    # Numbers alone do not tell the full story.  Seeing actual
    # images alongside the model's predictions and confidence
    # scores makes it much easier to understand what the
    # model has learned and where it struggles.
    #
    # Row 1 — ten correctly classified images
    #          The label shows the predicted digit and the
    #          model's confidence for that prediction.
    #
    # Row 2 — ten misclassified images
    #          The label shows both the wrong prediction and
    #          the true digit so we can see what fooled it.
    # --------------------------------------------------------

    correct_idx   = np.where(y_pred == y_test)[0][:10]   # first 10 correct predictions
    incorrect_idx = np.where(y_pred != y_test)[0][:10]   # first 10 wrong predictions

    fig, axes = plt.subplots(2, 10, figsize=(18, 4))
    fig.suptitle(
        "Prediction Examples — Row 1: Correct  |  Row 2: Mistakes",
        fontsize=12, fontweight="bold",
    )

    for col, idx in enumerate(correct_idx):
        ax = axes[0, col]
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        confidence = y_pred_probs[idx, y_pred[idx]] * 100   # softmax probability as %
        ax.set_title(f"✓ {y_pred[idx]}\n{confidence:.0f}%", fontsize=7, color="green")
        ax.axis("off")

    for col, idx in enumerate(incorrect_idx):
        ax = axes[1, col]
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        ax.set_title(
            f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}",
            fontsize=7, color="red",
        )
        ax.axis("off")

    plt.tight_layout()

    output_path = "outputs/plots/04_prediction_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved: {output_path}")


# ============================================================
# STAGE 11 — CLASSIFICATION REPORT
# ============================================================

def save_classification_report(y_test, y_pred, test_acc, test_loss):
    # --------------------------------------------------------
    # The classification report breaks down performance for
    # every individual digit class, giving us three metrics:
    #
    # Precision
    #   Of all the times the model predicted digit X, how
    #   often was it actually digit X?
    #   High precision -> few false positives.
    #
    # Recall
    #   Of all the real digit X images, how many did the model
    #   correctly identify?
    #   High recall -> few false negatives.
    #
    # F1-score
    #   The harmonic mean of precision and recall.
    #   It gives a single balanced score per class.
    #   Useful when you care equally about precision and recall.
    # --------------------------------------------------------

    report = classification_report(
        y_test, y_pred,
        target_names=[f"Digit {i}" for i in range(10)],
    )

    print("\n  --- Classification Report ---\n")
    print(report)

    output_path = "outputs/results/classification_report.txt"
    with open(output_path, "w") as f:
        f.write("MNIST Neural Network — Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy : {test_acc * 100:.2f} %\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n\n")
        f.write(report)

    print(f"  Report saved: {output_path}")


# ============================================================
# STAGE 12 — HYPERPARAMETER TUNING
# ============================================================

def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    # --------------------------------------------------------
    # Hyperparameters are settings we choose before training
    # begins the model does not learn them from data.
    # Choosing them well can meaningfully improve accuracy.
    #
    # We test three configurations that vary two hyperparameters:
    #
    # Learning rate
    #   Controls how large a step the optimiser takes after
    #   each batch.  Too high -> overshoots the optimal weights.
    #   Too low -> learns very slowly or gets stuck.
    #
    # Batch size
    #   How many images are processed before the weights are
    #   updated.  Smaller batches -> noisier but more frequent
    #   updates.  Larger batches -> smoother but less frequent.
    #
    # Configurations
    # ---------------------------------------------
    #  A  LR = 0.001,  Batch = 64   default LR, smaller batch
    #  B  LR = 0.001,  Batch = 256  default LR, larger batch
    #  C  LR = 0.0005, Batch = 128  halved LR,  medium batch
    # ---------------------------------------------
    #
    # Each config gets a completely fresh model no shared
    # weights and is trained for up to 20 epochs.
    # --------------------------------------------------------

    configs = [
        {"name": "LR=0.001  | BS=64",  "lr": 0.001,  "batch": 64},
        {"name": "LR=0.001  | BS=256", "lr": 0.001,  "batch": 256},
        {"name": "LR=0.0005 | BS=128", "lr": 0.0005, "batch": 128},
    ]

    summary_rows   = []         # one dict per config for the results CSV
    val_acc_curves = {}         # config name -> list of val_accuracy per epoch

    for cfg in configs:
        print(f"\n  -> Config: {cfg['name']}")

        model = build_model(learning_rate=cfg["lr"])   # fresh model for each config

        cb = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=0,                                 # silent we print our own summary
        )

        hist = model.fit(
            X_train, y_train,
            validation_split=0.10,
            epochs=20,                                 # capped at 20 for the tuning sweep
            batch_size=cfg["batch"],
            callbacks=[cb],
            verbose=0,                                 # suppress per-epoch lines
        )

        loss, acc  = model.evaluate(X_test, y_test, verbose=0)
        epochs_ran = len(hist.history["loss"])

        print(f"     Epochs ran    : {epochs_ran}")
        print(f"     Test Accuracy : {acc * 100:.2f} %")
        print(f"     Test Loss     : {loss:.4f}")

        summary_rows.append({
            "Config"        : cfg["name"],
            "Learning Rate" : cfg["lr"],
            "Batch Size"    : cfg["batch"],
            "Epochs Ran"    : epochs_ran,
            "Test Accuracy" : round(acc * 100, 2),
            "Test Loss"     : round(loss, 4),
        })

        val_acc_curves[cfg["name"]] = hist.history["val_accuracy"]

    # --- Save results CSV ---
    df_results = pd.DataFrame(summary_rows)
    csv_path   = "outputs/results/hyperparameter_results.csv"
    df_results.to_csv(csv_path, index=False)

    print("\n  --- Hyperparameter Comparison ---\n")
    print(df_results.to_string(index=False))
    print(f"\n  Results saved: {csv_path}")

    # --- Plot: validation accuracy curves per config ---
    colors = ["#2563EB", "#DC2626", "#16A34A"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for (name, curve), color in zip(val_acc_curves.items(), colors):
        ax.plot(range(1, len(curve) + 1), curve,
                label=name, color=color, linewidth=2)

    ax.set_title("Hyperparameter Tuning — Validation Accuracy per Config",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = "outputs/plots/05_hyperparameter_tuning.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved: {output_path}")

    # --- Plot: final test accuracy bar chart ---
    names = [r["Config"]        for r in summary_rows]
    accs  = [r["Test Accuracy"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.7)

    ax.set_title("Hyperparameter Comparison — Final Test Accuracy",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim([min(accs) - 1.5, 100])

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{acc:.2f} %",
            ha="center", fontsize=10, fontweight="bold",
            )

    plt.tight_layout()

    output_path = "outputs/plots/06_accuracy_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved: {output_path}")

    return df_results


# ============================================================
# STAGE 13 — SUMMARY REPORT
# ============================================================

def print_summary(test_acc, test_loss):
    # --------------------------------------------------------
    # Prints a clean final summary of what the pipeline produced.
    # --------------------------------------------------------

    print("")
    print("=" * 60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print("")
    print("  Dataset        : MNIST (70 000 images, 28x28 px)")
    print("  Architecture   : Feed-Forward Neural Network")
    print("                   Dense(256) -> Dense(128) -> Dense(64) -> Dense(10)")
    print("")
    print("  --- Model Performance (Test Set: 10 000 images) ---")
    print(f"  Test Accuracy  : {test_acc * 100:.2f} %")
    print(f"  Test Loss      : {test_loss:.4f}")
    print("")
    print("  --- Outputs ---")
    print("  outputs/plots/01_dataset_overview.png")
    print("  outputs/plots/02_training_curves.png")
    print("  outputs/plots/03_confusion_matrix.png")
    print("  outputs/plots/04_prediction_examples.png")
    print("  outputs/plots/05_hyperparameter_tuning.png")
    print("  outputs/plots/06_accuracy_comparison.png")
    print("  outputs/results/classification_report.txt")
    print("  outputs/results/hyperparameter_results.csv")
    print("")
    print("=" * 60)


# ============================================================
# PIPELINE RUNNER
# ============================================================
# Each stage is called in order.
# The output of one stage is passed as input to the next.
# ============================================================

def run_pipeline():

    print("")
    print("=" * 60)
    print("  NEURAL NETWORK — PIPELINE STARTING")
    print("=" * 60)

    # Stage 1 — Create output folders
    create_folders()

    # Stage 2 — Load and summarise data
    print("\n  [Stage 2] Loading MNIST dataset...")
    X_train_raw, y_train, X_test_raw, y_test = load_data()
    summarise_data(X_train_raw, y_train, X_test_raw, y_test)

    # Stage 3 — Visualise the raw dataset
    print("\n  [Stage 3] Plotting dataset overview...")
    plot_dataset_overview(X_train_raw, y_train)

    # Stage 4 — Preprocess images and labels
    print("\n  [Stage 4] Preprocessing...")
    X_train, y_train, X_test, y_test = preprocess(X_train_raw, y_train, X_test_raw, y_test)

    # Stage 5 — Build the neural network
    print("\n  [Stage 5] Building model...")
    model = build_model(learning_rate=0.001)
    model.summary()

    # Stage 6 — Train the model
    print("\n  [Stage 6] Training...")
    history = train_model(model, X_train, y_train, batch_size=128, epochs=30)

    # Stage 7 — Plot training curves
    print("\n  [Stage 7] Plotting training curves...")
    plot_training_curves(history)

    # Stage 8 — Evaluate on the test set
    print("\n  [Stage 8] Evaluating on test set...")
    test_loss, test_acc, y_pred, y_pred_probs = evaluate_model(model, X_test, y_test)

    # Stage 9 — Confusion matrix
    print("\n  [Stage 9] Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)

    # Stage 10 — Prediction examples
    print("\n  [Stage 10] Plotting prediction examples...")
    plot_prediction_examples(X_test, y_test, y_pred, y_pred_probs)

    # Stage 11 — Classification report
    print("\n  [Stage 11] Saving classification report...")
    save_classification_report(y_test, y_pred, test_acc, test_loss)

    # Stage 12 — Hyperparameter tuning
    print("\n  [Stage 12] Running hyperparameter tuning...")
    hyperparameter_tuning(X_train, y_train, X_test, y_test)

    # Stage 13 — Final summary
    print_summary(test_acc, test_loss)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()
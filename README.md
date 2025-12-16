# Image Classifier (MNIST)

Guide for reviewers of the single-notebook MNIST classifier. The notebook walks through loading and inspecting the MNIST handwritten digits dataset, training a simple convolutional neural network to recognize digits, and visualizing performance with metrics and plots.

- **What's here:** `image_classifier.ipynb` loads MNIST, visualizes samples, builds a small Keras CNN, trains 10 epochs, evaluates, and plots a confusion matrix plus learning curves.
- **Data:** MNIST is pulled automatically via `keras.datasets.mnist` (28x28 grayscale digits, 60k train/10k test).
- **Dependencies:** Python 3.x, TensorFlow/Keras, matplotlib, seaborn, scikit-learn, numpy. Install with `pip install tensorflow matplotlib seaborn scikit-learn`.
- **Run:** `jupyter notebook image_classifier.ipynb` (or `jupyter lab`). Execute cells top-to-bottom; GPU will speed up training but CPU works for this model.
- **Notebook flow:** Cells are grouped as 1) imports/helpers, 2) data load/visualization, 3) model build/summary, 4) training, 5) evaluation/plots. You can re-run sections independently; no hidden state.
- **Outputs:** Plots render inline (random sample grid, confusion matrix, loss/accuracy curves). Nothing is written to disk by default; save figures with `plt.savefig(...)` if needed.
- **Tuning quickstart:** Edit `EPOCHS`, `BATCH_SIZE`, or layer sizes in the model cell to experiment. Reducing epochs to 3–5 finishes in ~1–2 minutes on CPU; the default 10 epochs is still fast.
- **Model:** Two Conv2D+ReLU layers with max pooling, then Flatten → Dense(128, relu) → Dropout(0.5) → Dense(10, softmax). Optimizer: Adam. Loss: categorical cross-entropy. Batch size: 128. Epochs: 10.
- **Expected results:** Test accuracy around 99% (printed after evaluation). Confusion matrix should be strongly diagonal with minor slips (e.g., some 5↔3 or 9↔4). Learning curves should show declining loss without a widening train/val gap.
- **Review focus points:**
  - Confirm data normalization (`/255`) and label one-hot encoding.
  - Check for overfitting in plots; dropout is the main regularizer.
  - Validate evaluation uses the test set and predictions are argmaxed before confusion matrix.
  - Consider whether to save the model/weights or add reproducibility seeds if this were to be reused.

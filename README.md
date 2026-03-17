# budgie-audio-classifier
# Overview

This project focuses on classifying the wellness state of pet budgies based on their vocal sounds.
Audio signals are transformed into log-Mel spectrograms, which are then used as input to a CNN model for binary classification.

The dataset was manually collected from a pair of pet budgies and labeled based on a combination of:
- Behavioral observation
- Established knowledge of budgie vocalizations
- Contextual interpretation of their environment

# Dataset

Happy: 103 audio samples (15 seconds each)
Stressed: 97 audio samples (15 seconds each)

To improve model performance and increase dataset size, data augmentation was applied:

Audio was segmented into 3-second overlapping windows
This increased the number of training samples and improved generalization


# Methodology

- Audio preprocessing
- Conversion to log-Mel spectrograms
- CNN-based classification
- Training on augmented dataset

# Notes

Labels are based on observational and empirical criteria, not clinical validation
Dataset size is relatively small, so results should be interpreted accordingly

# Tech Stack

- Python
- TensorFlow / PyTorch (según uses)
- Librosa
- NumPy

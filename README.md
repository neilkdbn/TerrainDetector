# Terrain Classification and Adaptive Speed Control

This project classifies terrain images using a trained deep learning model and recommends rover speed based on terrain safety.

The system takes an input terrain image, predicts the terrain type, calculates the confidence score, assigns a risk level, and outputs the recommended rover speed in the terminal.

## Terrain Classes

The model classifies images into the following classes:
- Smooth Ground
- Gravel
- Sand
- Rock Field

## Speed Decision Logic

| Terrain Type | Risk Level | Recommended Rover Speed |
|--------------|------------|-------------------------|
| Smooth Ground | Safe | 70–100 km/h |
| Gravel | Moderate | 40–60 km/h |
| Sand | High | 20–40 km/h |
| Rock Field | Dangerous | 0–10 km/h |

## Features

- Image-based terrain classification
- Confidence score output
- Risk level mapping
- Recommended rover speed output
- Transfer learning using MobileNetV2
- Fine-tuning for improved performance

## Project Structure

```text
terrain-rover/
│
├── data/
├── models/
│   └── best_model.h5
├── utils/
│   ├── preprocess.py
│   ├── predict.py
│   └── decision.py
├── main.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
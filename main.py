import argparse
from utils.preprocess import preprocess_image
from utils.predict import load_trained_model, predict
from utils.decision import get_decision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="models/best_model.h5", help="Path to trained model")

    args = parser.parse_args()

    try:
        image = preprocess_image(args.image)
    except Exception:
        print("Invalid image path or unreadable image.")
        return

    try:
        model = load_trained_model(args.model)
    except Exception:
        print("Could not load model. Check model path.")
        return

    terrain, confidence = predict(model, image)
    risk, speed = get_decision(terrain)

    terrain_display = {
        "smooth": "Smooth Ground",
        "gravel": "Gravel",
        "sand": "Sand",
        "rock": "Rock Field"
    }.get(terrain, terrain)
    
    print()
    print(f"Terrain Detected: {terrain_display}")
    print(f"Confidence: {confidence * 100:.0f}%")
    print(f"Risk Level: {risk}")
    print(f"Recommended Rover Speed: {speed} km/h")

if __name__ == "__main__":
    main()
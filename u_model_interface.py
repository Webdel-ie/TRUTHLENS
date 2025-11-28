import pickle

# Load the full trained pipeline (TF-IDF + PAC model)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(text):
    pred = model.predict([text])[0]  # 1 = REAL, 0 = FAKE
    return "REAL" if pred == 1 else "FAKE"

if __name__ == "__main__":
    samples = [
        "NASA’s rover successfully collects samples from Mars to study signs of ancient life.",
        "Scientists claim humans can teleport within 5 years after secret experiments.",
        "World Health Organization reports decline in global smoking rates.",
        "Scientists discover a new planet that supports life."
    ]

    for s in samples:
        print("\nText:", s)
        print("Prediction:", predict(s))

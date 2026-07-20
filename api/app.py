import os
import torch
import numpy as np
from scipy.signal import hilbert
from flask import Flask, request, jsonify
import tempfile

app = Flask(__name__)

MODEL = None
DEVICE = None


def load_model():
    global MODEL, DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class AsymmetricPsiTNeuron(torch.nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.randn(out_f, in_f, dtype=torch.complex64) * 0.8)
            self.bias = torch.nn.Parameter(torch.zeros(out_f, dtype=torch.complex64))
            self.alpha = torch.nn.Parameter(torch.ones(out_f) * 1.5)
            self.beta = torch.nn.Parameter(torch.zeros(out_f))
            self.gamma = torch.nn.Parameter(torch.ones(out_f) * 2.0)

        def forward(self, x):
            z = torch.matmul(x, self.weights.t()) + self.bias
            R = torch.abs(z)
            theta = torch.angle(z)
            amp = torch.tanh(R)
            ang = torch.relu(theta + self.beta)
            phi = np.pi * self.gamma * torch.tanh(self.alpha * R) * torch.sin(ang)
            return amp * torch.complex(torch.cos(phi), torch.sin(phi))

    class NeuroFractalNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.channel_proj = torch.nn.Linear(256, 1, dtype=torch.complex64)
            self.psi1 = AsymmetricPsiTNeuron(64, 128)
            self.psi2 = AsymmetricPsiTNeuron(128, 64)
            self.norm1 = torch.nn.LayerNorm([256])
            self.norm2 = torch.nn.LayerNorm([128])
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(128, 32), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(16, 2),
            )

        def forward(self, x):
            if x.dim() == 3:
                x = x.squeeze(1)
            x = self.channel_proj(x).squeeze(-1)
            h = self.psi1(x)
            hr = torch.cat([torch.real(h), torch.imag(h)], dim=1)
            hr = self.norm1(hr)
            h = self.psi2(h)
            hr = torch.cat([torch.real(h), torch.imag(h)], dim=1)
            hr = self.norm2(hr)
            return self.classifier(hr)

    MODEL = NeuroFractalNet().to(DEVICE)
    model_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "psita_best.pt")
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in state:
            MODEL.load_state_dict(state["model_state_dict"])
        else:
            MODEL.load_state_dict(state)
        print("  Modele charge depuis checkpoints/psita_best.pt")
    else:
        print("  Modele non entraine charge (utilise les poids aleatoires)")
    MODEL.eval()


def process_eeg(eeg_data):
    n_channels = min(eeg_data.shape[0], 64)
    n_times = min(eeg_data.shape[1], 256)

    if eeg_data.shape[1] > n_times:
        indices = np.linspace(0, eeg_data.shape[1] - 1, n_times).astype(int)
        eeg_data = eeg_data[:, indices]

    eeg_data = eeg_data[:n_channels, :n_times]

    if np.iscomplexobj(eeg_data):
        complex_data = eeg_data.astype(np.complex64)
    else:
        complex_data = np.zeros((n_channels, n_times), dtype=np.complex64)
        for ch in range(n_channels):
            complex_data[ch] = hilbert(eeg_data[ch]).astype(np.complex64)

    if n_channels < 64:
        pad = np.zeros((64 - n_channels, n_times), dtype=np.complex64)
        complex_data = np.vstack([complex_data, pad])

    return torch.tensor(complex_data, dtype=torch.complex64).unsqueeze(0)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Ψ-NeuroFractal API",
        "version": "1.0.0",
        "author": "Gbane Assane",
        "description": "Detection precoce de maladies neurodegeneratives via reseaux de neurones complexes",
        "endpoints": {
            "POST /predict": "Analyse un signal EEG et retourne la prediction",
            "GET /health": "Verifie l'etat du service",
            "GET /docs": "Documentation de l'API",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else "unknown",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Modele non charge"}), 503

    if "eeg" not in request.files and "eeg" not in request.json:
        return jsonify({
            "error": "Aucun fichier EEG fourni",
            "usage": "Envoyez un fichier .npy ou .csv avec les donnees EEG",
            "format": "Shape attendue : (64 canaux, 256 points temporels)",
        }), 400

    try:
        if "eeg" in request.files:
            f = request.files["eeg"]
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                f.save(tmp.name)
                eeg_data = np.load(tmp.name)
                os.unlink(tmp.name)
        else:
            eeg_data = np.array(request.json["eeg"])

        tensor = process_eeg(eeg_data).to(DEVICE)

        with torch.no_grad():
            output = MODEL(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()

        result = {
            "prediction": "Pathologique" if pred == 1 else "Sain",
            "class": pred,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "sain": round(probs[0].item() * 100, 2),
                "pathologique": round(probs[1].item() * 100, 2),
            },
            "risk_level": (
                "Faible" if confidence < 0.6
                else "Modere" if confidence < 0.8
                else "Eleve"
            ),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/docs", methods=["GET"])
def docs():
    return jsonify({
        "title": "Ψ-NeuroFractal API Documentation",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": {
                "description": "Analyse un signal EEG et predit la presence de maladie neurodegenerative",
                "input": {
                    "format": "Fichier .npy ou JSON",
                    "shape": "(64, 256) - 64 canaux EEG, 256 points temporels",
                    "types_acceptes": ["numpy array (.npy)", "JSON 2D array"],
                },
                "output": {
                    "prediction": "Sain ou Pathologique",
                    "confidence": "Pourcentage de confiance (0-100)",
                    "risk_level": "Faible / Modere / Eleve",
                },
                "example_curl": "curl -X POST -F 'eeg=@eeg_data.npy' https://your-app.onrender.com/predict",
            }
        },
    })


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

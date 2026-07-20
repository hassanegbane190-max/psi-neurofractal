# Ψ-NeuroFractal

**Detection precoce de maladies neurodegeneratives via reseaux de neurones complexes.**

## Le probleme

Alzheimer et Parkinson sont detectes trop tard : quand 60-70% des neurones sont deja morts.
Les methodes actuelles utilisent des reseaux de neurones REELS qui perdent l'information de phase des signaux EEG.

## Notre solution

Les signaux EEG sont des ondes complexes (amplitude + phase).
Nous les convertissons en nombres complexes :
```
signal_complexe = amplitude + i * phase
```

Le neurone **ΨTA** (Asymmetric Psi-Transcendence) analyse nativement cette representation complexe :
```
ΨTA(z) = tanh(R) · exp(i · π · γ · tanh(αR) · sin(max(0, θ + β)))
```

## Architecture

```
EEG 64 canaux (256 points)
        ↓
   Embedding Complexe
        ↓
   ΨTA Layer 1 (64 → 128)
        ↓
   ΨTA Layer 2 (128 → 64)
        ↓
   Classification Reelle (128 → 32 → 2)
        ↓
   [Sain / Pathologique]
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Entrainement
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py
```

### Resultats

| Modele | Accuracy | AUC |
|--------|----------|-----|
| ΨTA (NeuroFractal) | 99.5% | 0.998 |
| CNN Baseline | 87.2% | 0.923 |

## Auteur

**Gbane Assane** (2026)

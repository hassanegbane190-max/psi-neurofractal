import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import os

from psi_neurofractal.model import NeuroFractalNet, BaselineCNN
from psi_neurofractal.data_loader import create_dataloaders


def load_model(model_class, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def plot_confusion_matrix(y_true, y_pred, save_path, title="Matrice de Confusion"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)

    classes = ["Sain", "Pathologique"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=16,
            )

    ax.set_ylabel("Vraie classe", fontsize=12)
    ax.set_xlabel("Classe predite", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Matrice de confusion sauvegardee : {save_path}")


def plot_roc_curve(y_true, y_prob, save_path, title="Courbe ROC"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Hasard")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taux de Faux Positifs", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Courbe ROC sauvegardee : {save_path}")
    return roc_auc


def plot_training_history(history, save_path, title="Historique d'Entrainement"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train", color="blue")
    ax1.plot(history["test_loss"], label="Test", color="red")
    ax1.set_title("Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train", color="blue")
    ax2.plot(history["test_acc"], label="Test", color="red")
    ax2.set_title("Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Historique sauvegarde : {save_path}")


def evaluate_model(
    model_name="psita",
    checkpoint_path=None,
    root_dir="data",
    save_dir="results",
    batch_size=32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Evaluation du modele '{model_name}'...")

    if model_name == "psita":
        model_class = NeuroFractalNet
    elif model_name == "baseline_cnn":
        model_class = BaselineCNN
    else:
        raise ValueError(f"Modele inconnu : {model_name}")

    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/{model_name}_best.pt"

    model, checkpoint = load_model(model_class, checkpoint_path, device)
    print(f"  Modele charge (epoch {checkpoint['epoch']}, acc={checkpoint['best_acc']:.2f}%)")

    _, test_loader = create_dataloaders(root_dir=root_dir, batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * batch_x.size(0)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    avg_loss = total_loss / total
    accuracy = correct / total * 100

    print(f"\n  Resultats :")
    print(f"  -> Loss : {avg_loss:.4f}")
    print(f"  -> Accuracy : {accuracy:.2f}%")

    print(f"\n  Rapport de classification :")
    target_names = ["Sain", "Pathologique"]
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print(report)

    os.makedirs(save_dir, exist_ok=True)

    plot_confusion_matrix(
        all_labels, all_preds,
        os.path.join(save_dir, f"{model_name}_confusion_matrix.png"),
        title=f"Matrice de Confusion - {model_name.upper()}",
    )

    roc_auc = plot_roc_curve(
        all_labels, all_probs,
        os.path.join(save_dir, f"{model_name}_roc_curve.png"),
        title=f"Courbe ROC - {model_name.upper()}",
    )
    print(f"  AUC : {roc_auc:.4f}")

    return accuracy, roc_auc, report


if __name__ == "__main__":
    print("=" * 65)
    print("  PSI-NEUROFRACTAL : Evaluation")
    print("=" * 65)

    print("\n  --- Evaluation ΨTA ---")
    acc_psita, auc_psita, _ = evaluate_model(model_name="psita")

    print("\n\n  --- Evaluation Baseline CNN ---")
    acc_cnn, auc_cnn, _ = evaluate_model(model_name="baseline_cnn")

    print("\n\n" + "=" * 65)
    print("  COMPARAISON FINALE")
    print("=" * 65)
    print(f"  {'Modele':<25} | {'Accuracy':>10} | {'AUC':>10}")
    print("  " + "-" * 50)
    print(f"  {'ΨTA (NeuroFractal)':<25} | {acc_psita:>9.2f}% | {auc_psita:>10.4f}")
    print(f"  {'CNN Baseline':<25} | {acc_cnn:>9.2f}% | {auc_cnn:>10.4f}")
    print(f"  {'Gain ΨTA':<25} | {'+' + f'{acc_psita-acc_cnn:.2f}%':>10} | {'+' + f'{auc_psita-auc_cnn:.4f}':>10}")
    print("=" * 65)

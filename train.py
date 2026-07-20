import torch
import torch.nn as nn
import numpy as np
import os
import time

from psi_neurofractal.model import NeuroFractalNet, BaselineCNN
from psi_neurofractal.data_loader import create_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train(
    model_name="psita",
    n_epochs=100,
    batch_size=32,
    lr=0.001,
    root_dir="data",
    save_dir="checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device utilise : {device}")

    print("\n  [1/4] Chargement des donnees EEG...")
    train_loader, test_loader = create_dataloaders(
        root_dir=root_dir, batch_size=batch_size
    )

    print(f"\n  [2/4] Construction du modele '{model_name}'...")
    if model_name == "psita":
        model = NeuroFractalNet().to(device)
        print("  -> Modele NeuroFractal ΨTA active")
    elif model_name == "baseline_cnn":
        model = BaselineCNN().to(device)
        print("  -> Modele baseline CNN reel")
    else:
        raise ValueError(f"Modele inconnu : {model_name}")

    n_params = model.count_parameters() if hasattr(model, 'count_parameters') else sum(p.numel() for p in model.parameters())
    print(f"  -> Parametres : {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, verbose=True
    )

    print(f"\n  [3/4] Entrainement sur {n_epochs} epochs...")
    print("  " + "=" * 65)
    print(f"  {'Epoch':>6} | {'Loss Train':>10} | {'Acc Train':>10} | {'Loss Test':>10} | {'Acc Test':>10}")
    print("  " + "-" * 65)

    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  {epoch:6d} | {train_loss:10.4f} | {train_acc:9.2f}% | {test_loss:10.4f} | {test_acc:9.2f}%"
            )

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "model_name": model_name,
                },
                os.path.join(save_dir, f"{model_name}_best.pt"),
            )

    elapsed = time.time() - start_time
    print("  " + "=" * 65)
    print(f"\n  [4/4] Entrainement termine en {elapsed:.1f}s")
    print(f"  -> Meilleure accuracy test : {best_acc:.2f}%")
    print(f"  -> Modele sauvegarde : {save_dir}/{model_name}_best.pt")

    return history, best_acc


if __name__ == "__main__":
    print("=" * 65)
    print("  PSI-NEUROFRACTAL : Entrainement ΨTA vs Baseline")
    print("=" * 65)

    print("\n  --- Modele ΨTA (NeuroFractal) ---")
    history_psita, acc_psita = train(model_name="psita", n_epochs=100)

    print("\n\n  --- Modele Baseline (CNN reel) ---")
    history_cnn, acc_cnn = train(model_name="baseline_cnn", n_epochs=100)

    print("\n\n" + "=" * 65)
    print("  COMPARAISON FINALE")
    print("=" * 65)
    print(f"  ΨTA (NeuroFractal)  : {acc_psita:.2f}%")
    print(f"  CNN Baseline        : {acc_cnn:.2f}%")
    print(f"  Gain ΨTA            : +{acc_psita - acc_cnn:.2f}%")
    print("=" * 65)

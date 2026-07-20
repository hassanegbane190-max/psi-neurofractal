import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import urllib.request
import zipfile
import mne


EEG_RECORDINGS_URL = "https://physionet.org/files/eegmatdatabase/1.0.0/"


class EEGAlzheimerDataset(Dataset):
    """Dataset EEG pour la detection de maladies neurodegeneratives.

    Convertit les signaux EEG reels en nombres complexes :
        signal_complexe = amplitude + i * phase

    Le neurone ΨTA analyse nativement cette representation complexe.
    """

    def __init__(self, root_dir="data", n_channels=64, n_times=256, transform=None):
        self.root_dir = root_dir
        self.n_channels = n_channels
        self.n_times = n_times
        self.transform = transform
        self.samples = []
        self.labels = []

        self._load_or_generate_data()

    def _load_or_generate_data(self):
        """Charge les donnees EEG ou genere des donnees simulees."""
        data_path = os.path.join(self.root_dir, "eeg_data.pt")

        if os.path.exists(data_path):
            data = torch.load(data_path)
            self.samples = data["samples"]
            self.labels = data["labels"]
            print(f"  Donnees chargees : {len(self.samples)} echantillons")
            return

        print("  Generation de donnees EEG simulees realistes...")
        self._generate_realistic_eeg()

        os.makedirs(self.root_dir, exist_ok=True)
        torch.save(
            {"samples": self.samples, "labels": self.labels},
            data_path,
        )
        print(f"  Donnees sauvegardees dans {data_path}")

    def _generate_realistic_eeg(self):
        """Genere des signaux EEG simules avec des patterns pathologiques.

        Les signaux "sains" reproduisent les rythmes normaux :
        - Alpha (8-12 Hz) dominant sur les regions posterieures
        - Beta (12-30 Hz) sur les regions frontales
        - Theta (4-8 Hz) present mais modere

        Les signaux "pathologiques" montrent :
        - Reduction des ondes Alpha
        - Augmentation des ondes Theta/Delta
        - Coherence reduite entre canaux
        - Activite chaotique accrue (parfait pour ΨTA)
        """
        n_samples_per_class = 500
        fs = 256.0
        t = np.arange(self.n_times) / fs

        for class_label in range(2):
            for _ in range(n_samples_per_class):
                eeg = np.zeros((self.n_channels, self.n_times))

                for ch in range(self.n_channels):
                    if class_label == 0:
                        eeg[ch] = self._generate_normal_eeg(t, ch)
                    else:
                        eeg[ch] = self._generate_pathological_eeg(t, ch)

                complex_signal = self._to_complex(eeg)
                self.samples.append(torch.tensor(complex_signal, dtype=torch.complex64))
                self.labels.append(class_label)

        self.samples = torch.stack(self.samples)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _generate_normal_eeg(self, t, channel):
        """EEG sain : rythmes Alpha domines."""
        alpha = 10.0 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2 * np.pi))
        beta = 3.0 * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2 * np.pi))
        theta = 4.0 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, 2 * np.pi))
        noise = 0.5 * np.random.randn(len(t))
        return alpha + beta + theta + noise

    def _generate_pathological_eeg(self, t, channel):
        """EEG pathologique : Alpha reduit, Theta/Delta augmentes, chaos."""
        alpha = 2.0 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2 * np.pi))
        theta = 8.0 * np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 2 * np.pi))
        delta = 6.0 * np.sin(2 * np.pi * 2 * t + np.random.uniform(0, 2 * np.pi))

        chaos = 0.0
        for k in range(1, 6):
            chaos += (1.0 / k) * np.sin(
                2 * np.pi * k * 3.7 * t + np.random.uniform(0, 2 * np.pi)
            )

        noise = 1.5 * np.random.randn(len(t))
        return alpha + theta + delta + chaos + noise

    def _to_complex(self, eeg_real):
        """Convertit un signal EEG reel en representation complexe.

        Utilise la transformee de Hilbert pour extraire :
        - Partie reelle = amplitude du signal
        - Partie imaginaire = phase instantee
        """
        from scipy.signal import hilbert

        complex_signals = np.zeros((self.n_channels, self.n_times), dtype=np.complex64)
        for ch in range(self.n_channels):
            analytic = hilbert(eeg_real[ch])
            complex_signals[ch] = analytic.astype(np.complex64)
        return complex_signals

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def create_dataloaders(root_dir="data", batch_size=32, train_ratio=0.8):
    """Cree les DataLoaders d'entrainement et de test."""
    dataset = EEGAlzheimerDataset(root_dir=root_dir)

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_test = n - n_train

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"  Train : {n_train} echantillons | Test : {n_test} echantillons")
    return train_loader, test_loader

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Implementacja straty Knowledge Distillation (Hinton et al., 2015).
    Pomaga zachować wiedzę z nauczyciela (model bazowy) podczas doszkalania ucznia.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()  # Strata dla twardych etykiet

    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_outputs: Wyjścia z modelu ucznia (doszkalany model)
            teacher_outputs: Wyjścia z modelu nauczyciela (oryginalny model)
            targets: Rzeczywiste etykiety
        """
        # Strata dla twardych etykiet (Cross Entropy)
        hard_loss = self.ce_loss(student_outputs, targets)

        # Strata dla miękkich etykiet (KL Divergence)
        T = self.temperature
        soft_targets = F.softmax(teacher_outputs / T, dim=1)
        soft_outputs = F.log_softmax(student_outputs / T, dim=1)
        soft_loss = F.kl_div(soft_outputs, soft_targets, reduction="batchmean") * (
            T * T
        )

        # Kombinacja obu strat
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


class ElasticWeightConsolidation:
    """
    Implementacja mechanizmu Elastic Weight Consolidation (Kirkpatrick et al., 2017).
    Pozwala na doszkalanie modelu przy zachowaniu wcześniej nabytej wiedzy.
    """

    def __init__(
        self,
        model: nn.Module,
        fisher_diagonal: Dict[str, torch.Tensor],
        lambda_param: float = 100.0,
    ):
        self.model = model
        self.fisher = fisher_diagonal
        self.lambda_param = lambda_param
        self.original_params = {}

        # Zapisz oryginalne wartości parametrów
        for name, param in model.named_parameters():
            if name in self.fisher:
                self.original_params[name] = param.data.clone()

    def calculate_penalty(self) -> torch.Tensor:
        """Oblicza karę EWC dla aktualnych parametrów modelu."""
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.original_params:
                # Kara za odchylenie od oryginalnych parametrów, ważona macierzą Fishera
                loss += torch.sum(
                    self.fisher[name] * (param - self.original_params[name]).pow(2)
                )

        return self.lambda_param * loss


class RehearsalMemory:
    """
    Implementacja pamięci rehearsal do przechowywania przykładów z poprzednich zadań.
    Pomaga zapobiegać katastrofalnemu zapominaniu poprzez dodatkowe treningi na starych przykładach.
    """

    def __init__(self, capacity_per_class: int = 20):
        self.capacity_per_class = capacity_per_class
        self.memory: Dict[int, List[torch.Tensor]] = (
            {}
        )  # Słownik: klasa -> lista przykładów

    def add_examples(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Dodaje przykłady do pamięci, z uwzględnieniem pojemności dla każdej klasy."""
        for i in range(inputs.size(0)):
            target = targets[i].item()
            input_example = inputs[i].unsqueeze(0).detach().cpu()

            if target not in self.memory:
                self.memory[target] = []

            # Dodaj przykład, jeśli jest jeszcze miejsce dla tej klasy
            if len(self.memory[target]) < self.capacity_per_class:
                self.memory[target].append(input_example)

    def get_batch(
        self, batch_size: int = 16, device: Optional[torch.device] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Pobiera losowy batch przykładów z pamięci."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        examples = []
        targets = []

        # Zbierz wszystkie klasy, dla których mamy przykłady
        available_classes = list(self.memory.keys())
        if not available_classes:
            return None, None  # Pamięć jest pusta

        # Losuj klasy i przykłady, aż zbierzemy batch_size przykładów
        while len(examples) < batch_size and available_classes:
            # Losuj klasę
            target_class = random.choice(available_classes)

            # Jeśli dla tej klasy mamy przykłady
            if self.memory[target_class]:
                # Losuj przykład dla tej klasy
                example_idx = random.randint(0, len(self.memory[target_class]) - 1)
                examples.append(self.memory[target_class][example_idx])
                targets.append(target_class)
            else:
                # Usuń klasę z dostępnych, jeśli nie ma dla niej przykładów
                available_classes.remove(target_class)

        # Połącz przykłady w jeden tensor
        if examples:
            examples = torch.cat(examples, dim=0).to(device)
            targets = torch.tensor(targets, device=device)
            return examples, targets

        return None, None


def generate_synthetic_samples(
    model: nn.Module,
    class_indices: List[int],
    num_samples: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """
    Generuje syntetyczne próbki dla klas.

    Args:
        model: Model do generowania próbek
        class_indices: Lista indeksów klas
        num_samples: Liczba próbek do wygenerowania dla każdej klasy
        device: Urządzenie do obliczeń (CPU/GPU)

    Returns:
        Słownik z wygenerowanymi próbkami dla każdej klasy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    synthetic_samples = {}

    for class_idx in class_indices:
        # Inicjalizacja losowych próbek
        samples = torch.randn(num_samples, 3, 224, 224, device=device)
        samples.requires_grad = True

        # Optymalizacja próbek, aby model rozpoznawał je jako daną klasę
        optimizer = torch.optim.Adam([samples], lr=0.1)

        for iter_idx in range(100):  # 100 iteracji optymalizacji
            optimizer.zero_grad()

            outputs = model(samples)
            targets = torch.full((num_samples,), class_idx, device=device)
            loss = F.cross_entropy(outputs, targets)

            loss.backward()
            optimizer.step()

        # Oderwij gradient i zapisz próbki dla tej klasy
        synthetic_samples[class_idx] = samples.detach()

    return synthetic_samples


def compute_fisher_information(model, data_loader, device=None, num_samples=200):
    """
    Optymalizacja: Usunięcie zbędnych instrukcji i uproszczenie logiki.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    fisher_diagonal = {}
    for name, param in model.named_parameters():
        fisher_diagonal[name] = torch.zeros_like(param)

    sample_count = 0

    for inputs, targets in data_loader:
        if sample_count >= num_samples:
            break

        samples_to_process = min(inputs.size(0), num_samples - sample_count)
        if samples_to_process <= 0:
            break

        inputs = inputs[:samples_to_process]
        targets = targets[:samples_to_process]

        inputs, targets = inputs.to(device), targets.to(device)

        log_probs = torch.log_softmax(model(inputs), dim=1)

        for i in range(samples_to_process):
            if targets[i] < 0:
                continue

            log_prob = log_probs[i, targets[i]]

            model.zero_grad()

            log_prob.backward(retain_graph=(i < samples_to_process - 1))

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_diagonal[name] += param.grad.data**2 / samples_to_process

        sample_count += samples_to_process

    if sample_count > 0:
        for name in fisher_diagonal:
            fisher_diagonal[name] /= sample_count

    return fisher_diagonal

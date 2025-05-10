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


def compute_fisher_information(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    num_samples: int,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Oblicza diagonalną macierz informacji Fishera dla parametrów modelu.
    
    Args:
        model: Model do obliczenia informacji Fishera
        data_loader: DataLoader z przykładami
        num_samples: Liczba próbek do użycia
        device: Urządzenie do obliczeń (CPU/GPU)
        
    Returns:
        Słownik z diagonalną macierzą informacji Fishera dla każdego parametru
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inicjalizuj słownik Fisher dla każdego parametru
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)
    
    # Ustaw model w tryb ewaluacji podczas zbierania danych
    model.eval()
    
    sample_count = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Przejdź przez dane i oblicz diagonalną macierz Fishera
    for inputs, targets in data_loader:
        if sample_count >= num_samples:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # Przetwórz tylko tyle próbek, ile potrzeba
        actual_batch_size = min(batch_size, num_samples - sample_count)
        if actual_batch_size < batch_size:
            inputs = inputs[:actual_batch_size]
            targets = targets[:actual_batch_size]
        
        # Dla każdej próbki oblicz gradient log-prawdopodobieństwa
        for i in range(actual_batch_size):
            model.zero_grad()
            
            # Forward pass dla pojedynczej próbki
            output = model(inputs[i:i+1])
            
            # Oblicz stratę dla poprawnej klasy
            log_prob = criterion(output, targets[i:i+1])
            
            # Backward pass
            log_prob.backward()
            
            # Dodaj kwadraty gradientów do macierzy Fishera
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.pow(2) / num_samples
            
            sample_count += 1
    
    return fisher

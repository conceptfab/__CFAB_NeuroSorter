Elementy warte pokazania na wykresach podczas doszkalania modeli
Podczas doszkalania modeli warto monitorować i wizualizować szereg metryk, które dostarczają cennych informacji o postępie i jakości procesu. Oto najważniejsze elementy, które warto przedstawić na wykresach:
1. Podstawowe metryki treningu

Funkcja straty (Loss) - Wykres przedstawiający zmianę funkcji straty zarówno dla zbioru treningowego, jak i walidacyjnego w kolejnych epokach
Dokładność (Accuracy) - Zmiana dokładności modelu na zbiorze treningowym i walidacyjnym

2. Metryki specyficzne dla doszkalania

Porównanie z modelem bazowym - Wykres pokazujący różnicę wydajności między modelem bazowym a doszkalanym na tych samych danych
Miary zapominania - Wykres pokazujący wydajność doszkalanego modelu na oryginalnym zbiorze danych w miarę postępu treningu
Wykres transferu wiedzy - Porównanie wydajności modelu na zadaniach z nowej domeny w zależności od ilości danych treningowych

3. Zaawansowane metryki

Dystans wag od inicjalizacji - Wykres Euklidesowego dystansu wag modelu od wag modelu bazowego w kolejnych epokach
Normy gradientów - Wykres średnich norm gradientów dla różnych warstw modelu
Współczynniki uczenia - Zmiany współczynnika uczenia (learning rate) w czasie, jeśli używany jest scheduler

4. Wizualizacje przestrzeni cech

t-SNE lub UMAP - Wizualizacja przestrzeni cech modelu przed i po doszkalaniu, pokazująca jak zmieniła się reprezentacja danych
Mapy aktywacji - Porównanie map aktywacji kluczowych warstw przed i po doszkalaniu dla tych samych danych wejściowych

5. Metryki skuteczności dla specyficznych zadań
Dla zadań klasyfikacji obrazów:

Macierz pomyłek (Confusion Matrix) - Wizualizacja macierzy pomyłek przed i po doszkalaniu
Krzywa ROC i AUC - Porównanie krzywych ROC i wartości AUC
Precision-Recall - Wykresy Precision-Recall dla poszczególnych klas

Dla modeli NLP:

Perpleksyjność (Perplexity) - Zmiany perpleksyjności modelu na zbiorze walidacyjnym
BLEU, ROUGE - Dla zadań generowania tekstu

6. Analizy specjalistyczne

Wykres efektywności danych - Zmiana wydajności modelu w zależności od ilości danych treningowych
Wrażliwość wag (Weight Sensitivity) - Analiza wrażliwości modelu na zmiany poszczególnych wag
Histogramy aktywacji - Dystrybucja aktywacji w warstwach przed i po doszkalaniu

7. Metryki wydajnościowe

Czas treningu - Czas trwania epok
Wykorzystanie pamięci - Zużycie pamięci GPU w trakcie treningu
Throughput - Liczba próbek przetwarzanych na sekundę

Przykładowy kod dla wykresów doszkalania
Oto przykładowy kod do generowania niektórych z wyżej wymienionych wykresów:
pythonimport matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

# 1. Wykres straty i dokładności
def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Wykres funkcji straty
    ax1.plot(history['train_loss'], label='Trening')
    ax1.plot(history['val_loss'], label='Walidacja')
    ax1.axvline(x=history['best_epoch'], color='r', linestyle='--', label='Najlepsza epoka')
    ax1.set_title('Funkcja straty')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Strata')
    ax1.legend()
    
    # Wykres dokładności
    ax2.plot(history['train_acc'], label='Trening')
    ax2.plot(history['val_acc'], label='Walidacja')
    ax2.axvline(x=history['best_epoch'], color='r', linestyle='--', label='Najlepsza epoka')
    ax2.set_title('Dokładność')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Dokładność')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 2. Porównanie z modelem bazowym
def plot_comparison_with_base(base_metrics, finetuned_metrics, metrics=['accuracy', 'f1', 'precision', 'recall'], save_path=None):
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    base_values = [base_metrics[m] for m in metrics]
    finetuned_values = [finetuned_metrics[m] for m in metrics]
    
    ax.bar(x - width/2, base_values, width, label='Model bazowy')
    ax.bar(x + width/2, finetuned_values, width, label='Model doszkolony')
    
    ax.set_ylabel('Wartość')
    ax.set_title('Porównanie wydajności modelu bazowego i doszkalanego')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Dodanie etykiet z procentową zmianą
    for i, (base, finetuned) in enumerate(zip(base_values, finetuned_values)):
        change = ((finetuned - base) / base) * 100
        color = 'green' if change > 0 else 'red'
        ax.annotate(f"{change:+.1f}%", 
                   xy=(i + width/2, finetuned), 
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   color=color, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 3. Wizualizacja macierzy pomyłek
def plot_confusion_matrices(base_model, finetuned_model, test_loader, class_names, device, save_path=None):
    # Funkcja do otrzymania predykcji modelu
    def get_predictions(model):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        return np.array(all_preds), np.array(all_targets)
    
    # Pobierz predykcje
    base_preds, targets = get_predictions(base_model)
    finetuned_preds, _ = get_predictions(finetuned_model)
    
    # Oblicz macierze pomyłek
    base_cm = confusion_matrix(targets, base_preds)
    finetuned_cm = confusion_matrix(targets, finetuned_preds)
    
    # Normalizuj macierze
    base_cm_norm = base_cm.astype('float') / base_cm.sum(axis=1)[:, np.newaxis]
    finetuned_cm_norm = finetuned_cm.astype('float') / finetuned_cm.sum(axis=1)[:, np.newaxis]
    
    # Wizualizacja
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(base_cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax1)
    ax1.set_title('Model bazowy')
    ax1.set_xlabel('Predykcja')
    ax1.set_ylabel('Rzeczywista klasa')
    
    sns.heatmap(finetuned_cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax2)
    ax2.set_title('Model doszkolony')
    ax2.set_xlabel('Predykcja')
    ax2.set_ylabel('Rzeczywista klasa')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 4. Wizualizacja t-SNE przestrzeni cech
def plot_tsne_features(base_model, finetuned_model, data_loader, device, layer_name='features', save_path=None):
    # Funkcja do ekstrakcji cech z określonej warstwy modelu
    def get_features(model, layer_name):
        features = []
        labels = []
        
        # Funkcja hook do przechwytywania aktywacji
        def hook_fn(module, input, output):
            features.append(output.cpu().numpy())
        
        # Zarejestruj hook na odpowiedniej warstwie
        for name, module in model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
        
        model.eval()
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                model(inputs)
                labels.extend(targets.numpy())
        
        # Usuń hook
        handle.remove()
        
        return np.vstack(features), np.array(labels)
    
    # Pobierz cechy z obu modeli
    base_features, labels = get_features(base_model, layer_name)
    finetuned_features, _ = get_features(finetuned_model, layer_name)
    
    # Przygotuj dane do t-SNE (redukuj wymiarowość, jeśli potrzeba)
    base_features_flat = base_features.reshape(base_features.shape[0], -1)
    finetuned_features_flat = finetuned_features.reshape(finetuned_features.shape[0], -1)
    
    # Wykonaj t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    base_tsne = tsne.fit_transform(base_features_flat)
    tsne = TSNE(n_components=2, random_state=42)
    finetuned_tsne = tsne.fit_transform(finetuned_features_flat)
    
    # Wizualizacja
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    scatter1 = ax1.scatter(base_tsne[:, 0], base_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax1.set_title('Przestrzeń cech modelu bazowego')
    ax1.set_xlabel('t-SNE Wymiar 1')
    ax1.set_ylabel('t-SNE Wymiar 2')
    
    scatter2 = ax2.scatter(finetuned_tsne[:, 0], finetuned_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax2.set_title('Przestrzeń cech modelu doszkalanego')
    ax2.set_xlabel('t-SNE Wymiar 1')
    ax2.set_ylabel('t-SNE Wymiar 2')
    
    # Dodanie legendy
    fig.colorbar(scatter1, ax=ax1, label='Klasa')
    fig.colorbar(scatter2, ax=ax2, label='Klasa')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
Te wykresy pomagają lepiej zrozumieć jak przebiega proces doszkalania, jakie są jego efekty i gdzie mogą występować problemy. Pozwalają też na lepszą dokumentację procesu i łatwiejsze porównanie różnych podejść do doszkalania.
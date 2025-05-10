to jest scieżka fine-tuningu:
panel ustawien - fine_tuning_task_config_dialog.py -> przygotowuje profil zadania
profil zadania - b2_classes_fine_tuning_5.json - zawiera inforamcje dla pliku wykonawczego
plik wykonawczy - fine_tuning.py
fine_tuning.py: przyklad loga:
--- Epoka 4/5 ---
C:\tools\python\Lib\site-packages\pyqtgraph\graphicsItems\ScatterPlotItem.py:888: RuntimeWarning: All-NaN slice encountered
self.bounds[ax] = (np.nanmin(d) - self.\_maxSpotWidth*0.7072, np.nanmax(d) + self.\_maxSpotWidth*0.7072)
C:\tools\python\Lib\site-packages\pyqtgraph\graphicsItems\ScatterPlotItem.py:888: RuntimeWarning: All-NaN slice encountered
self.bounds[ax] = (np.nanmin(d) - self.\_maxSpotWidth*0.7072, np.nanmax(d) + self.\_maxSpotWidth*0.7072)
EWC loss component: 0.000176, Lambda: 5000.0
2025-05-10 20:59:32,451 [INFO] Epoka 4/1 | Strata: 3.5251, Dokładność: 4687.50% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:32,538 [INFO] Epoka 4/1 | Strata: 3.3682, Dokładność: 4687.50% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:32,610 [INFO] Epoka 4/1 | Strata: 3.4107, Dokładność: 4583.33% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:32,697 [INFO] Epoka 4/1 | Strata: 3.4430, Dokładność: 4609.38% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:32,785 [INFO] Epoka 4/1 | Strata: 3.4345, Dokładność: 4625.00% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:32,892 [INFO] Epoka 4/1 | Strata: 3.3928, Dokładność: 4635.42% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:32,990 [INFO] Epoka 4/1 | Strata: 3.3914, Dokładność: 4508.93% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:33,091 [INFO] Epoka 4/1 | Strata: 3.3239, Dokładność: 4726.56% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:33,190 [INFO] Epoka 4/1 | Strata: 3.3610, Dokładność: 4652.78% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:33,293 [INFO] Epoka 4/1 | Strata: 3.3750, Dokładność: 4562.50% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
EWC loss component: 0.000177, Lambda: 5000.0
2025-05-10 20:59:33,396 [INFO] Epoka 4/1 | Strata: 3.4129, Dokładność: 4488.64% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:33,497 [INFO] Epoka 4/1 | Strata: 3.3732, Dokładność: 4557.29% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:33,599 [INFO] Epoka 4/1 | Strata: 3.3507, Dokładność: 4639.42% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:33,700 [INFO] Epoka 4/1 | Strata: 3.3375, Dokładność: 4620.54% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:33,804 [INFO] Epoka 4/1 | Strata: 3.3583, Dokładność: 4604.17% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:33,895 [INFO] Epoka 4/1 | Strata: 3.3387, Dokładność: 4667.97% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:33,983 [INFO] Epoka 4/1 | Strata: 3.3397, Dokładność: 4632.35% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:34,072 [INFO] Epoka 4/1 | Strata: 3.3134, Dokładność: 4635.42% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:34,157 [INFO] Epoka 4/1 | Strata: 3.2977, Dokładność: 4720.39% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:34,237 [INFO] Epoka 4/1 | Strata: 3.2843, Dokładność: 4812.50% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
EWC loss component: 0.000175, Lambda: 5000.0
2025-05-10 20:59:34,320 [INFO] Epoka 4/1 | Strata: 3.2883, Dokładność: 4776.79% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:34,404 [INFO] Epoka 4/1 | Strata: 3.2861, Dokładność: 4786.93% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
2025-05-10 20:59:34,491 [INFO] Epoka 4/1 | Strata: 3.3356, Dokładność: 4767.93% | Val Strata: 0.0000, Val Acc: 0.00% | Top-3: 0.00%, Top-5: 0.00% | Precision: 0.00%, Recall: 0.00% | F1: 0.00%, AUC: 0.00%
C:\tools\python\Lib\site-packages\sklearn\metrics_ranking.py:379: UndefinedMetricWarning: Only one class is present in y_true. ROC AUC score is not defined in that case.
warnings.warn(
Ostrzeżenie: Nie udało się obliczyć niektórych metryk: Number of classes in 'y_true' (3) not equal to the number of classes in 'y_score' (33).You can provide a list of all known classes by assigning it to the `labels` parameter.
Epoka 4/5 | Train Loss: 3.3356 | Train Acc: 47.68% | Val Loss: 3.1408 | Val Acc: 21.67%

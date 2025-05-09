Zestawienie parametrów dedykowanych dla monitorowania doszkalania
Poniżej przedstawiam zestawienie parametrów, które warto monitorować specyficznie podczas procesu doszkalania modelu:

1. Metryki transferu wiedzy
   ParametrOpisWizualizacjaKnowledge Transfer Index (KTI)Stosunek wydajności modelu doszkolonego do modelu bazowegoWykres liniowy rosnący z epoki na epokęLayer Adaptation RateStopień zmiany wag w poszczególnych warstwachWykres słupkowy pokazujący zmiany wag w każdej warstwieFeature Space AlignmentMierzy jak dobrze cechy ze zbioru źródłowego mapują się na zbiór docelowyWizualizacja t-SNE/UMAP pokazująca zbliżanie się przestrzeni cech
2. Metryki efektywności doszkalania
   ParametrOpisWizualizacjaFine-tuning EfficiencyStosunek przyrostu dokładności do liczby zaktualizowanych parametrówWykres liniowy pokazujący efektywność wykorzystania parametrówLayer-wise Gradient MagnitudePokazuje, które warstwy najbardziej się ucząMapa cieplna intensywności gradientów dla warstwCatastrophic Forgetting MetricMierzy, jak model zachowuje się na oryginalnym zbiorze danychWykres liniowy pokazujący wydajność na zbiorze źródłowym
3. Metryki strategii odmrażania warstw
   ParametrOpisWizualizacjaFrozen vs Unfrozen Performance GapRóżnica w wydajności między zamrożonymi i odmrożonymi wariantamiWykres porównawczyProgressive Unfreezing ImpactWpływ stopniowego odmrażania warstw na wydajnośćWykres schodkowy pokazujący zmiany po odmrożeniu każdej warstwyLayer Contribution IndexWskazuje, które odmrożone warstwy wnoszą największy wkładWykres słupkowy wkładu poszczególnych warstw
4. Metryki adaptacji do nowej domeny
   ParametrOpisWizualizacjaDomain Adaptation ScoreMierzy jak dobrze model adaptuje się do nowej domenyWykres liniowy z krzywą uczenia dla nowej domenyClass-specific Adaptation RatePokazuje, które klasy adaptują się najszybciej/najwolniejMapa cieplna postępu adaptacji dla każdej klasyFeature Activation DriftŚledzi zmiany w aktywacjach warstw między domenamiHistogramy aktywacji przed/po doszkoleniu
5. Metryki diagnostyczne dla doszkalania
   ParametrOpisWizualizacjaLearning Rate SensitivityPokazuje wpływ różnych współczynników uczenia na doszkalanieWykres wpływu LR na dokładnośćWeight Drift VelocityPrędkość zmiany wag w porównaniu do oryginalnego modeluWykres kumulacyjnej zmiany wag w czasieActivation Boundary ShiftMierzy, jak zmieniają się granice decyzyjneWizualizacja granic decyzyjnych przed/po
6. Metryki dla zaawansowanej analizy doszkalania
   ParametrOpisWizualizacjaRepresentational Similarity AnalysisMierzy podobieństwo reprezentacji między warstwamiMacierz podobieństwa reprezentacjiPretraining Retention ScoreJak wiele wiedzy z pretreningu zostało zachowaneWykres % retencji wiedzyModel Plasticity IndexZdolność modelu do adaptacji bez utraty wydajnościWykres pokazujący kompromis stabilność-plastyczność
7. Praktyczne metryki wdrożeniowe dla doszkalania
   ParametrOpisWizualizacjaDomain Generalization GapRóżnica w wydajności między domeną źródłową i docelowąWykres porównawczyFine-tuning Convergence RateJak szybko model osiąga optymalną wydajnośćWykres drugiej pochodnej krzywej uczeniaResource Efficiency MetricsStosunek poprawy wydajności do zużytych zasobówWykres efektywności (GPU-godziny / % poprawa)
   Monitoring tych specjalistycznych parametrów pozwala na głębsze zrozumienie procesu doszkalania i podejmowanie bardziej świadomych decyzji dotyczących strategii treningu, co przekłada się na lepsze wyniki końcowe modelu.

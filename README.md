# **AI for Personalized Medicine: AIDS Clinical Trials Group Study 175**

Questo progetto applica tecniche di **Machine Learning** e **Inferenza Causale (S-Learner)** per trasformare i dati clinici in strumenti decisionali. L'obiettivo è duplice: prevedere la progressione della malattia in pazienti affetti da HIV e identificare il trattamento farmacologico ottimale per ogni singolo individuo (Medicina di Precisione).


##  **Descrizione del Progetto**

Il progetto si articola in tre fasi principali:

1. **Analisi Predittiva:** Confronto tra **Logistic Regression** (approccio lineare e bilanciato) e **Random Forest** (approccio non lineare) per identificare i pazienti ad alto rischio di evento critico (`cid`).  
2. **Analisi Prescrittiva:** Utilizzo di un approccio **S-Learner** per simulare l'effetto di quattro diversi trattamenti sullo stesso paziente, selezionando quello con il rischio stimato minore.  
3. **Valutazione dell'Impatto Clinico:** Quantificazione del beneficio attraverso la riduzione del rischio assoluto e il calcolo del **Number Needed to Treat (NNT)**.

##  **Dataset**

Viene utilizzato il dataset **AIDS Clinical Trials Group Study 175** recuperato dall'UCI Machine Learning Repository (ID: 890).

* **Pazienti:** 2.139.  
* **Target (`y`):** `cid` (indicatore di progressione dell'AIDS o decesso).  
* **Caratteristiche principali (`X`):** Età, peso, conta dei CD4 (`cd40`), giorni di terapia antiretrovirale precedente (`preanti`) e punteggio di Karnofsky (`karnof`).

##  **Librerie Utilizzate**

Il progetto è sviluppato in Python e richiede le seguenti librerie:

* **`ucimlrepo`**: Per l'importazione diretta dei dati.  
* **`pandas` e `numpy`**: Per la manipolazione dei dati e calcoli matematici.  
* **`scikit-learn`**: Per la pipeline di ML (scaling, split, modelli e metriche).  
* **`matplotlib` e `seaborn`**: Per la generazione di grafici clinici e matrici di confusione.

## 

##  **Come Eseguire il Codice**

Per riprodurre i risultati:

1\. **Installazione Dipendenze:**  
	pip install ucimlrepo pandas scikit-learn matplotlib seaborn

2\. **Preparazione Dati:** Il codice scarica automaticamente il dataset, rimuove la colonna `time` per evitare *data leakage* e gestisce lo sbilanciamento delle classi tramite pesi bilanciati.

3\. **Esecuzione:** Eseguire lo script Python o le celle del notebook.

4\. **Output:** Lo script genererà:

* Report di classificazione (Precision, Recall, F1-score).  
* Matrici di confusione e Curve ROC.  
* Analisi dei profili dei pazienti: si scopre che il trattamento 2 è ideale per i pazienti “naive”, mentre il trattamento 3 è per i “veterani” che hanno avuto un’alta esposizione ai farmaci. 



##  **Risultati Chiave**

* **Capacità Predittiva**: La Random Forest cattura interazioni non lineari complesse, mentre la Logistic Regression bilanciata minimizza i falsi negativi (fondamentale in ambito medico).  
* **Efficacia AI**: Il modello suggerisce una terapia ottimale che riduce il rischio medio del 3.93%.  
* **Impatto Clinico**: È stato calcolato un NNT di 25.4, indicando un'elevata efficienza nel prevenire eventi critici tramite la personalizzazione della cura.

# -*- coding: utf-8 -*-
"""Progetto AI2 "AIDS Clinical Trials"


pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

X = aids_clinical_trials_group_study_175.data.features
y = aids_clinical_trials_group_study_175.data.targets

print(aids_clinical_trials_group_study_175.metadata)

print(aids_clinical_trials_group_study_175.variables)

X = X.drop(columns=['time'], errors='ignore')
# 1. Uniamo X e y per vedere tutto insieme un attimo (opzionale ma utile)
df_total = pd.concat([X, y], axis=1)

print("\n--- Valori Mancanti ---")
print(df_total.isnull().sum())

print("Nuove colonne in X:", X.columns)

print("\n--- Colonne in X ---")
print(X.columns)

print("\n--- Colonne in y ---")
print(y.columns)

y_clean = y['cid'] if hasattr(y, 'columns') else y

X_train, X_test, y_train, y_test = train_test_split(X, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=42,class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)

y_pred_log = log_reg.predict(X_test_scaled)

print("--- LOGISTIC REGRESSION REPORT ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

feature_names = X.columns
coefficients = log_reg.coef_[0]

coef_df = pd.DataFrame({'Feature': feature_names,'Coefficient': coefficients})

coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='vlag')
plt.title('Importanza delle Feature (Logistic Regression Coefficients)')
plt.xlabel('Valore del Coefficiente (Peso)')
plt.ylabel('Variabili Cliniche')
plt.axvline(x=0, color='black', linestyle='--')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight="balanced")
model.fit(X_train, y_train.values.ravel())

paziente = X_test.iloc[0].copy()

possibili_trattamenti = [0, 1, 2, 3]

def trova_trattamento_migliore(paziente,modello):
    risultati_simulati = []
    for t in possibili_trattamenti:
        paziente_clone = paziente.copy()
        paziente_clone['trt'] = t
        df_clone = pd.DataFrame([paziente_clone])
        prob_fallimento = model.predict_proba(df_clone)[0][1]
        risultati_simulati.append( prob_fallimento)
        miglior_trattamento = np.argmin(risultati_simulati)
        prob_minima = risultati_simulati[miglior_trattamento]
        return miglior_trattamento, prob_minima, risultati_simulati
print("\n--- Analisi primo paziente ---")
paziente_reale = X_test.iloc[0]
print(f"Paziente reale (trattamento reale:{int(paziente_reale['trt'])})")
best_trt, min_risk, tutti_rischi = trova_trattamento_migliore(paziente_reale,model)
for t,rischio in enumerate(tutti_rischi):
  print(f"Se facesse il Trattamento {t}, Rischio: {rischio:.2%}")
  print(f"L'AI consiglia il trattamento: {best_trt}")
  print(f"   Rischio previsto: {min_risk:.2%}")


plt.figure(figsize=(8, 5))
colors = ['red' if i != best_trt else 'green' for i in range(len(tutti_rischi))]

sns.barplot(x=possibili_trattamenti, y=tutti_rischi, hue=possibili_trattamenti, palette=colors, legend=False)
plt.title(f'Analisi Personalizzata Paziente #{X_test.index[0]}')
plt.xlabel('Tipo di Trattamento (0-3)')
plt.ylabel('Probabilità di Evento Critico (Rischio)')
plt.ylim(0, 1) # Da 0% a 100%

for i, v in enumerate(tutti_rischi):
    plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

plt.show()

print("--- CALCOLO IMPATTO GLOBALE SUL TEST SET ---")
miglioramenti = []
pazienti_ottimizzati = 0
totale_pazienti = len(X_test)

for i in range(totale_pazienti):
    paziente = X_test.iloc[i]
    trattamento_reale = int(paziente['trt'])

    best_trt, min_risk, tutti_rischi = trova_trattamento_migliore(paziente, model)

    rischio_reale = tutti_rischi[trattamento_reale]

    if best_trt != trattamento_reale:
        diff = rischio_reale - min_risk
        if diff > 0:
            miglioramenti.append(diff)
            pazienti_ottimizzati += 1

media_riduzione = np.mean(miglioramenti) if miglioramenti else 0
perc_ottimizzati = pazienti_ottimizzati / totale_pazienti

print(f"Pazienti analizzati: {totale_pazienti}")
print(f"Pazienti a cui l'AI cambierebbe terapia: {pazienti_ottimizzati} ({perc_ottimizzati:.1%})")
print(f"Riduzione media del rischio: {media_riduzione:.2%}")

vittorie_trattamenti = {0: 0, 1: 0, 2: 0, 3: 0}

print("Simulazione su tutto il test set in corso...")

for index, row in X_test.iterrows():
    best_t, _, _ = trova_trattamento_migliore(row, model)
    vittorie_trattamenti[best_t] += 1

print("\n--- CLASSIFICA TRATTAMENTI CONSIGLIATI ---")
print("Su quanti pazienti ogni trattamento è risultato il migliore?")
for t, count in vittorie_trattamenti.items():
    print(f"Trattamento {t}: consigliato a {count} pazienti")

plt.figure(figsize=(8,5))
sns.barplot(x=list(vittorie_trattamenti.keys()), y=list(vittorie_trattamenti.values()))
plt.title("Distribuzione dei Trattamenti Ottimali Suggeriti dall'AI")
plt.xlabel("Trattamento")
plt.ylabel("Numero di Pazienti")
plt.show()


labels = ['Trattamento 0', 'Trattamento 1', 'Trattamento 2', 'Trattamento 3']
sizes = [74, 136, 125, 93]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, pctdistance=0.85)

best_treatments = []
for index, row in X_test.iterrows():
    best_t, _, _ = trova_trattamento_migliore(row, model)
    best_treatments.append(best_t)

X_analysis = X_test.copy()
X_analysis['Recommended_Trt'] = best_treatments
print("--- PROFILO PAZIENTE PER TRATTAMENTO CONSIGLIATO ---")
grouped_analysis = X_analysis.groupby('Recommended_Trt')[['age', 'wtkg', 'cd40', 'karnof', 'preanti']].mean()
print(grouped_analysis)

SOGLIA_RILEVANZA = 0.01  # 1%

arr_miglioramenti = np.array(miglioramenti)

beneficiari_reali = arr_miglioramenti[arr_miglioramenti >= SOGLIA_RILEVANZA]
pazienti_ok = len(arr_miglioramenti) - len(beneficiari_reali)

beneficiari_pct = beneficiari_reali * 100

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

sns.histplot(beneficiari_pct, bins=15, color="green", kde=True)

if len(beneficiari_pct) > 0:
    media_reale = np.mean(beneficiari_pct)

    plt.axvline(media_reale, color='red', linestyle='dashed', linewidth=2,
                label=f'Riduzione Media (sui pazienti modificati): {media_reale:.2f}%')

plt.title(f'Analisi Impatto Clinico (Soglia Rilevanza: {SOGLIA_RILEVANZA*100}%)', fontsize=16)
plt.xlabel('Riduzione del Rischio Assoluto (%)', fontsize=12)
plt.ylabel('Numero di Pazienti', fontsize=12)
plt.legend()

media_globale = np.mean(miglioramenti)

if media_globale > 0:
    nnt_valore = 1 / media_globale
else:
    nnt_valore = 0 # Caso impossibile, ma per sicurezza

testo_info = (f"Totale pazienti: {len(miglioramenti)}\n"
              f"Pazienti da cambiare (Gain > 1%): {len(beneficiari_reali)}\n"
              f"Pazienti ben curati (Gain < 1%): {pazienti_ok}\n"
              f"Riduzione media rischio: {media_globale*100:.2f}%\n"
              f"NNT stimato: {nnt_valore:.1f} pazienti")

plt.gca().text(0.95, 0.95, testo_info, transform=plt.gca().transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log, ax=ax[0], cmap='Blues', colorbar=False)
ax[0].set_title('Confusion Matrix: Logistic Regression')

y_pred = model.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[1], cmap='Greens', colorbar=False)
ax[1].set_title('Confusion Matrix: Random Forest')

plt.tight_layout()
plt.show()

y_probs_log = log_reg.predict_proba(X_test_scaled)[:, 1]
y_probs_rf = model.predict_proba(X_test)[:, 1]

fpr_log, tpr_log, _ = roc_curve(y_test, y_probs_log)
roc_auc_log = auc(fpr_log, tpr_log)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(10, 7))
plt.plot(fpr_log, tpr_log, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificità)')
plt.ylabel('True Positive Rate (Sensibilità)')
plt.title('Confronto Curve ROC: RF vs Logistic Regression')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

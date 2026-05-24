import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)

# 1- Génération d'une base client simulée

np.random.seed(42)
n = 2000

data = pd.DataFrame({
    "age": np.random.randint(18, 75, n),
    "anciennete_mois": np.random.randint(1, 72, n),
    "montant_mensuel": np.random.normal(65, 20, n).clip(20, 130),
    "nombre_services": np.random.randint(1, 6, n),
    "appels_support": np.random.poisson(1.5, n),
    "score_satisfaction": np.random.randint(1, 11, n),
    "type_contrat": np.random.choice(
        ["mensuel", "un_an", "deux_ans"],
        n,
        p=[0.55, 0.30, 0.15]
    )
})

#type de contrat
data["contrat_mensuel"] = (data["type_contrat"] == "mensuel").astype(int)
data["contrat_un_an"] = (data["type_contrat"] == "un_an").astype(int)
data["contrat_deux_ans"] = (data["type_contrat"] == "deux_ans").astype(int)

# Construction d'une probabilité de churn réaliste
score_churn = (
    0.03 * data["montant_mensuel"]
    + 0.45 * data["appels_support"]
    - 0.06 * data["anciennete_mois"]
    - 0.35 * data["score_satisfaction"]
    + 1.2 * data["contrat_mensuel"]
    - 0.8 * data["contrat_deux_ans"]
    + np.random.normal(0, 1, n)
)

prob_churn = 1 / (1 + np.exp(-(-1.5 + 0.35 * score_churn)))
data["churn"] = (prob_churn > np.random.uniform(0, 1, n)).astype(int)

# Sauvegarde de la base
data.to_csv("data/customer_churn_data.csv", index=False)

print("Taille de la base :", data.shape)
print(data.head())

print(" Taux de churn :")
print(data["churn"].value_counts(normalize=True))


# 2- Analyse descriptive

print(" Statistiques descriptives :")
print(data.describe())

plt.figure(figsize=(6, 4))
data["churn"].value_counts().plot(kind="bar")
plt.title("Répartition du churn")
plt.xlabel("Churn")
plt.ylabel("Nombre de clients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/figures/churn_distribution.png")
plt.close()

plt.figure(figsize=(6, 4))
data.groupby("churn")["score_satisfaction"].mean().plot(kind="bar")
plt.title("Score moyen de satisfaction selon le churn")
plt.xlabel("Churn")
plt.ylabel("Score moyen de satisfaction")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/figures/satisfaction_by_churn.png")
plt.close()

plt.figure(figsize=(6, 4))
data.groupby("churn")["appels_support"].mean().plot(kind="bar")
plt.title("Nombre moyen d'appels support selon le churn")
plt.xlabel("Churn")
plt.ylabel("Nombre moyen d'appels support")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/figures/support_calls_by_churn.png")
plt.close()


# 3- Préparation des variables

features = [
    "age",
    "anciennete_mois",
    "montant_mensuel",
    "nombre_services",
    "appels_support",
    "score_satisfaction",
    "contrat_mensuel",
    "contrat_un_an",
    "contrat_deux_ans"
]

X = data[features]
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 4- Modèle 1 : Régression logistique

logit_model = LogisticRegression(max_iter=1000, class_weight="balanced")
logit_model.fit(X_train_scaled, y_train)

y_pred_logit = logit_model.predict(X_test_scaled)
y_proba_logit = logit_model.predict_proba(X_test_scaled)[:, 1]

print(" Performance de la régression logistique :")
print("Accuracy :", round(accuracy_score(y_test, y_pred_logit), 3))
print("Precision :", round(precision_score(y_test, y_pred_logit), 3))
print("Recall :", round(recall_score(y_test, y_pred_logit), 3))
print("F1-score :", round(f1_score(y_test, y_pred_logit), 3))
print("ROC-AUC :", round(roc_auc_score(y_test, y_proba_logit), 3))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_logit))


# 5. Modèle 2 : Random Forest


rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("Performance du Random Forest :")
print("Accuracy :", round(accuracy_score(y_test, y_pred_rf), 3))
print("Precision :", round(precision_score(y_test, y_pred_rf), 3))
print("Recall :", round(recall_score(y_test, y_pred_rf), 3))
print("F1-score :", round(f1_score(y_test, y_pred_rf), 3))
print("ROC-AUC :", round(roc_auc_score(y_test, y_proba_rf), 3))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_rf))


# 6. Visualisations des modèles

RocCurveDisplay.from_predictions(
    y_test,
    y_proba_logit,
    name="Régression logistique"
)
RocCurveDisplay.from_predictions(
    y_test,
    y_proba_rf,
    name="Random Forest"
)

plt.title("Comparaison des courbes ROC")
plt.tight_layout()
plt.savefig("outputs/figures/roc_curve_comparison.png")
plt.close()
importance = pd.DataFrame({
    "variable": features,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(" Importance des variables :")
print(importance)

plt.figure(figsize=(8, 5))
plt.barh(importance["variable"], importance["importance"])
plt.title("Importance des variables - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/figures/feature_importance.png")
plt.close()

print("\nProjet terminé avec succès.")

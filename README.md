## Résultats et interprétation

La base contient 2 000 clients. On observe que 15,1 % des clients quittent l’entreprise ce qui signifie que la variable churn est déséquilibrée. Il y a beaucoup plus de clients qui restent que de clients qui partent.
Pour éviter que les modèles prédisent presque toujours “pas de churn”, j’ai utilisé l’option class_weight="balanced". Ca permet de mieux prendre en compte les clients qui quittent l’entreprise.
La régression logistique obtient un recall de 0,627. Ca veut dire qu’elle repère environ 62,7 % des clients qui partent réellement. Dans un contexte CRM ce résultat est intéressant, car l’objectif est surtout d’identifier les clients à risque pour pouvoir agir avant leur départ.

Le Random Forest obtient une accuracy plus élevée, avec 0,760, mais son recall est plus faible avec 0,400. Il prédit donc mieux l’ensemble des clients, mais il détecte moins bien les clients qui quittent réellement l’entreprise. 

Dans ce projet, la régression logistique peut donc être préférée si l’objectif principal est de repérer les clients à risque. Même si elle fait plus d’erreurs, elle permet de cibler davantage de clients susceptibles de partir.
Les variables les plus importantes sont le montant mensuel payé par le client, son ancienneté, son score de satisfaction, son âge et le nombre d’appels au support. Ces résultats sont assez cohérents un client moins satisfait, qui contacte souvent le service client ou qui paie un montant élevé peut être plus susceptible de quitter l’entreprise.
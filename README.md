## Résultats et interprétation

## Ce projet porte sur une problématique classique en data science, prédire le départ des clients appelé churn.

L’objectif est d’identifier les clients qui risquent de quitter l’entreprise à partir de plusieurs informations comme l’ancienneté, le montant payé chaque mois, le score de satisfaction, le nombre d’appels au support et le type de contrat.

Ce type d’analyse peut être utile dans beaucoup de secteurs par exemple les télécommunications, l’assurance, la banque ou les services en ligne. L’idée est de repérer les clients à risque afin de mieux comprendre leur comportement et de mettre en place des actions de fidélisation.

La base utilisée contient 2 000 clients.  
On observe que 15,1 % des clients quittent l’entreprise. La variable `churn` est ducoup déséquilibrée, car il y a beaucoup plus de clients qui restent que de clients qui partent.
Ce déséquilibre pose un problème pour la modélisation. Un modèle peut obtenir une bonne accuracy en prédisant presque toujours que les clients ne partent pas. Mais dans ce cas il ne serait pas vraiment utile, car il détecterait mal les clients à risque.

Pour limiter ce problème j’ai utilisé l’option `class_weight="balanced"`. Cette option permet de donner plus de poids aux clients qui quittent réellement l’entreprise pendant l’apprentissage du modèle.

Deux modèles ont été comparés : une régression logistique et un Random Forest.

La régression logistique obtient un recall de 0,627. Cela signifie qu’elle repère environ 62,7 % des clients qui partent réellement.

Dans ce projet ce résultat est intéressant car l’objectif principal est surtout d’identifier les clients à risque. Il vaut mieux détecter davantage de clients susceptibles de partir, même si cela entraîne quelques erreurs supplémentaires.

Le Random Forest obtient une accuracy plus élevée, avec 0,760.  Cela signifie qu’il prédit mieux l’ensemble des clients.
Par contre son recall est plus faible, avec 0,400. Il détecte donc moins bien les clients qui quittent réellement l’entreprise.

Dans ce cas la régression logistique peut donc être préférée si l’objectif principal est de repérer les clients à risque.  
Même si elle fait plus d’erreurs, elle permet de cibler davantage de clients susceptibles de partir.

L’analyse des variables montre que les facteurs les plus importants sont :
- le montant mensuel payé par le client 
- l’ancienneté du client 
- le score de satisfaction 
- l’âge 
- le nombre d’appels au support

Ces résultats sont cohérents. Un client moins satisfait, qui contacte souvent le service client ou qui paie un montant mensuel élevé peut être plus susceptible de quitter l’entreprise.

Ce projet montre comment un modèle de machine learning simple peut être utilisé pour mieux comprendre le comportement des clients et aider à la prise de décision.

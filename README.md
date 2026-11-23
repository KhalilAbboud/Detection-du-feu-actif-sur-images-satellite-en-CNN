# SatFire CNN — Détection du feu actif sur images satellites [Active Fire Detection on Satellite Imagery]

Projet de vision par ordinateur consacré à la **segmentation sémantique pixel par pixel** d’images satellites VIIRS. À l’aide d’un réseau de neurones convolutif dilaté (CNN), le modèle classe chaque pixel en quatre catégories : **forêt [Forest]**, **feu actif [Active Fire]**, **zone brûlée [Burned Area]** et **autre [Other]**.

Le projet s’appuie sur le jeu de données **TS-SatFire**. Le **notebook d’analyse et d’entraînement [Analysis and Training Notebook]** constitue le cœur du travail : il documente la préparation des données, la construction des masques, l’entraînement, l’évaluation et l’analyse des erreurs. Une petite application Streamlit est également présente, mais elle sert uniquement de prototype expérimental pour vérifier l’inférence sur une image.

## Objectifs du projet [Project Goals]

L’objectif est d’étudier dans quelle mesure un CNN peut localiser les pixels de feu actif tout en les distinguant des zones déjà brûlées et des autres types de couverture du sol.

Il ne s’agit donc pas d’une simple classification binaire feu / absence de feu. Le modèle réalise une **segmentation multi-classes [Multi-class Segmentation]** : il attribue une étiquette à chaque pixel de l’image.

## Données et préparation [Dataset and Preprocessing]

### Jeu de données [Dataset]

Le projet utilise **TS-SatFire**, un jeu de données d’images satellites VIIRS et d’annotations associées aux événements de feu.

- [Jeu de données TS-SatFire sur Kaggle](https://www.kaggle.com/datasets/z789456sx/ts-satfire)
- [Article scientifique associé](https://doi.org/10.1038/s41597-025-06271-3)

Chaque entrée est une image de **256 × 256 pixels et 6 bandes spectrales**.

Sur 3 556 dates appariées, 1 150 ont été écartées parce que la couche d’annotation du feu ou de la zone brûlée était vide. Le jeu retenu comprend donc **2 346 paires image-masque**, issues de **140 événements**.

### Classes cibles [Target Classes]

Les masques sont construits à partir des informations de couverture du sol et des couches d’annotation du feu et des surfaces brûlées.

| ID | Classe | Règle d’annotation |
|---:|---|---|
| 0 | Forêt [Forest] | Classes de couverture du sol MODIS / IGBP 1 à 5, hors feu et zone brûlée |
| 1 | Feu actif [Active Fire] | Pixels annotés comme feu actif ; cette classe est prioritaire |
| 2 | Zone brûlée [Burned Area] | Périmètre brûlé, en l’absence de feu actif |
| 3 | Autre [Other] | Sol connu non forestier, hors feu et zone brûlée ; classes MODIS / IGBP 6 à 17 |
| 255 | Ignoré [Ignored] | Valeurs manquantes, sol inconnu ou étiquette incertaine ; ces pixels sont exclus de la perte |

En cas de chevauchement, **le feu actif est prioritaire sur la zone brûlée**. Les pixels incertains ne participent pas au calcul de la fonction de perte.

### Prétraitement et découpage [Preprocessing and Data Split]

- **Normalisation :** moyenne et écart-type calculés séparément pour chaque bande, uniquement à partir des données d’entraînement.
- **Augmentation des données [Data Augmentation] :** retournements, rotations de 90°, variations de gain et de décalage par bande, ainsi qu’ajout de bruit gaussien.
- **Découpage par événement [Event-based Split] :** toutes les images d’un même incendie restent dans le même groupe afin d’éviter qu’un événement apparaisse à la fois dans l’entraînement et dans le test.

| Groupe | Événements | Images | Pixels feu | Pixels brûlés | Pixels autre |
|---|---:|---:|---:|---:|---:|
| Entraînement [Train] | 98 | 1 758 | 207 064 | 1 096 731 | 95 247 178 |
| Validation [Validation] | 21 | 259 | 26 935 | 76 909 | 15 116 700 |
| Test [Test] | 21 | 329 | 49 778 | 310 464 | 17 672 792 |

Le déséquilibre des classes est important : les pixels de feu sont très rares par rapport aux pixels de la classe « Autre ». L’entraînement tient compte de ce déséquilibre à l’aide de poids de classes et d’une perte combinée.

## Architecture du modèle [Model Architecture]

Le modèle est un **CNN dilaté [Dilated CNN]** conçu pour conserver la résolution spatiale tout en élargissant progressivement le contexte utilisé pour classifier chaque pixel.

| Couche | Filtres | Dilatation | Champ réceptif cumulatif |
|---|---:|---:|---:|
| Bloc convolutionnel 1 | 32 | 1 | 3 × 3 |
| Bloc convolutionnel 2 | 32 | 2 | 7 × 7 |
| Bloc convolutionnel 3 | 64 | 4 | 15 × 15 |
| Bloc convolutionnel 4 | 64 | 8 | 31 × 31 |
| Bloc convolutionnel 5 | 64 | 16 | 63 × 63 |
| Bloc convolutionnel 6 | 32 | 1 | 65 × 65 |
| Couche de sortie 1 × 1 | 4 | — | 65 × 65 |

Chaque bloc convolutionnel utilise une convolution 3 × 3 suivie d’une **Batch Normalization** et d’une activation **ReLU**. La couche finale 1 × 1 utilise **Softmax** pour produire les probabilités des quatre classes.

Le réseau ne contient ni pooling ni U-Net : les cartes de caractéristiques conservent la résolution **256 × 256**, ce qui permet une prédiction pixel par pixel sans étape de suréchantillonnage.

- **Entrée :** 256 × 256 × 6
- **Sortie :** 256 × 256 × 4
- **Nombre de paramètres :** 122 820, dont 576 non entraînables
- **Taille indicative des paramètres :** environ 480 Ko

## Fonction de perte [Loss Function]

Pour limiter l’effet du déséquilibre des classes, l’apprentissage utilise une combinaison de deux termes :

**Perte totale = entropie croisée pondérée [Weighted Cross-Entropy] + 0,9 × (1 − Dice moyen sur le feu et le brûlé) [Mean Dice Loss]**

Les poids de classes utilisés sont :

| Classe | Poids |
|---|---:|
| Forêt [Forest] | 2,27 |
| Feu actif [Active Fire] | 10,00 |
| Zone brûlée [Burned Area] | 9,32 |
| Autre [Other] | 1,00 |

Les poids suivent la règle `min(√(n_max / n_c), 10)`, où `n_max` est le nombre de pixels de la classe majoritaire et `n_c` celui de la classe considérée.

Le terme Dice se concentre sur les classes **Feu** et **Brûlé**. Les pixels d’ID 255 sont masqués et n’influencent pas la perte.

## Entraînement et sélection [Training and Model Selection]

| Paramètre | Configuration |
|---|---|
| Optimiseur | Adam |
| Taux d’apprentissage initial | 0,001 |
| Taille du lot [Batch Size] | 8 |
| Nombre maximal d’époques | 30 |
| Étapes par époque | 220 |
| Arrêt anticipé [Early Stopping] | Patience de 8 époques |
| Réduction du taux [Learning Rate Reduction] | Facteur 0,5 après 3 époques sans amélioration de l’IoU feu en validation |
| Critère de sélection | IoU de la classe Feu sur la validation |

La perte d’entraînement est passée de **1,55 à 0,71** au cours des 30 époques, pour environ **31 minutes** d’entraînement dans l’exécution présentée. Le meilleur modèle a été obtenu à l’époque 25, avec une **IoU feu de validation de 0,455**.

Le jeu de test n’a pas été utilisé pour choisir l’époque du modèle ni pour régler les facteurs de décision.

## Règle de décision [Decision Rule]

Après la prédiction des probabilités, une règle de décision ajuste la compétition entre certaines classes :

- facteur appliqué au feu : **1,25** face à la classe brûlée ;
- facteur appliqué à la forêt : **0,67** face à la classe « Autre ».

Ces facteurs ont été choisis sur le jeu de validation puis figés avant l’évaluation sur le test. Leur effet sur le test est toutefois **très faible** : l’IoU feu passe de 0,5335 avec `argmax` à 0,5320 après réglage. Le réglage modifie légèrement le compromis entre classes, sans améliorer les résultats globaux.

## Résultats sur le jeu de test [Test Results]

L’évaluation principale porte sur **21 événements et 329 images** répartis dans un jeu de test à quatre classes. Les métriques par classe ci-dessous correspondent à l’évaluation avec les facteurs de décision retenus sur la validation.

### Indicateurs globaux [Overall Metrics]

| Indicateur | Résultat |
|---|---:|
| Accuracy [Exactitude globale] | **90,8 %** |
| Référence « tout Autre » [All-Other Baseline] | 82,2 % |
| IoU moyenne des 4 classes [Mean IoU] | **0,622** |
| IoU moyenne Feu / Brûlé [Mean Fire / Burned IoU] | **0,506** |

L’accuracy est supérieure à la référence qui attribue systématiquement la classe « Autre ». Cependant, compte tenu du fort déséquilibre des classes, l’accuracy ne suffit pas à évaluer le modèle : les métriques par classe sont également nécessaires.

### Résultats par classe [Per-class Metrics]

| Classe | Précision [Precision] | Rappel [Recall] | F1-score [F1-score] | IoU [Intersection over Union] |
|---|---:|---:|---:|---:|
| Forêt [Forest] | 0,760 | 0,706 | 0,732 | **0,577** |
| Feu actif [Active Fire] | 0,654 | 0,741 | 0,694 | **0,532** |
| Zone brûlée [Burned Area] | 0,596 | 0,712 | 0,649 | **0,480** |
| Autre [Other] | 0,942 | 0,951 | 0,946 | **0,898** |

Le modèle retrouve une partie importante des pixels de feu : le rappel de la classe Feu atteint **0,741**. Sa précision est de **0,654**, ce qui indique également la présence de fausses détections. La classe « Autre » est la mieux reconnue, tandis que la séparation entre forêt et autres terrains reste difficile.

### Matrice de confusion normalisée [Normalized Confusion Matrix]

Les lignes correspondent aux classes réelles et les colonnes aux classes prédites.

| Réel ↓ / Prédit → | Forêt | Feu | Brûlé | Autre |
|---|---:|---:|---:|---:|
| Forêt | 0,71 | 0,00 | 0,01 | **0,28** |
| Feu | 0,02 | **0,74** | 0,16 | 0,08 |
| Brûlé | 0,05 | 0,03 | **0,71** | 0,21 |
| Autre | 0,04 | 0,00 | 0,01 | **0,95** |

L’erreur dominante est la confusion entre **Forêt** et **Autre** : 28 % des pixels réellement forestiers sont prédits comme « Autre ». Par ailleurs, 16 % des pixels de feu sont classés comme brûlés.

### Variabilité selon les événements [Event-level Variability]

Les moyennes masquent des différences d’un événement à l’autre :

- IoU feu médiane par événement : **0,494** ; les valeurs observées vont approximativement de 0,25 à 0,72.
- IoU brûlé médiane par événement : **0,407**.
- Pixels urbains prédits comme feu : **0,004 %**.
- Pixels d’eau prédits comme feu : **0,001 %**.

Les scores calculés sur des événements contenant très peu de pixels de feu ou de brûlé doivent être interprétés avec prudence.

## Validation croisée à 5 plis [5-fold Cross-Validation]

Une expérience distincte de validation croisée à cinq plis a également été réalisée, avec **28 événements de test par pli**.

**Attention : cette expérience utilise un modèle à trois classes (Forêt, Feu, Brûlé), sans la classe « Autre ». Ses résultats ne sont donc pas directement comparables au test à quatre classes présenté ci-dessus.** Cette expérience sert principalement à étudier la stabilité des résultats entre les plis.

| Pli | Événements | IoU forêt | IoU feu | IoU brûlé | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 28 | 0,971 | 0,579 | 0,631 | 96,8 % |
| 2 | 28 | 0,983 | 0,595 | 0,704 | 97,9 % |
| 3 | 28 | 0,975 | 0,576 | 0,747 | 97,2 % |
| 4 | 28 | 0,982 | 0,573 | 0,659 | 97,9 % |
| 5 | 28 | 0,981 | 0,535 | 0,612 | 97,9 % |
| **Moyenne ± écart-type** | — | **0,978 ± 0,005** | **0,572 ± 0,022** | **0,671 ± 0,055** | **97,5 ± 0,5 %** |

L’IoU feu reste relativement stable entre les plis (0,535 à 0,595), tandis que l’IoU brûlé varie davantage (0,612 à 0,747).

## Notebook principal [Main Notebook]

Le notebook est le support principal pour comprendre et reproduire la démarche expérimentale. Il regroupe les étapes essentielles du projet :

1. Chargement et exploration des images VIIRS et des annotations TS-SatFire.
2. Préparation des masques et définition des quatre classes.
3. Normalisation des bandes et séparation des données par événement.
4. Augmentation des données et calcul des poids de classes.
5. Construction et entraînement du CNN dilaté.
6. Sélection du modèle et réglage de la règle de décision sur la validation.
7. Évaluation sur le jeu de test, analyse de la matrice de confusion et étude des résultats par événement.
8. Expérience complémentaire de validation croisée à cinq plis.

Pour présenter le travail, privilégier les visualisations produites par le notebook : image satellite, masque de référence, prédiction du modèle, matrice de confusion et graphiques des métriques. **Il n’est pas nécessaire d’ajouter des captures d’écran de l’application Streamlit.**

## Prototype Streamlit [Streamlit Prototype]

Le dépôt contient également une application Streamlit utilisée comme **prototype de test** pour vérifier que le modèle entraîné peut effectuer une prédiction à partir d’une image fournie en entrée.

Cette application n’est pas le livrable principal du projet et ne doit pas être considérée comme une interface finalisée. Le travail central reste le notebook et l’analyse expérimentale du modèle. Les captures d’écran de cette application ne sont donc pas nécessaires dans ce README.

## Organisation du dépôt [Repository Structure]

- **Notebook principal :** préparation des données, entraînement, évaluation et visualisations.
- `model/` : fichiers du modèle entraîné et éléments nécessaires à l’inférence.
- `app.py` : prototype Streamlit de test d’inférence.
- `app_samples/` : exemples utilisés avec le prototype.
- `checktiff.py` : utilitaire d’inspection de fichiers TIFF.
- `docs/` : documentation complémentaire.

## Limites et pistes d’amélioration [Limitations and Future Work]

- **Confusion Forêt / Autre :** 28 % des pixels forestiers sont classés comme « Autre ».
- **Confusion Feu / Brûlé :** 16 % des pixels de feu sont classés comme zones brûlées.
- **Annotations imparfaites :** certaines zones brûlées prédites ne figurent pas dans le masque de référence ; il peut s’agir d’annotations manquantes qui nécessitent une vérification.
- **Variabilité entre événements :** l’IoU feu varie sensiblement d’un incendie à l’autre.
- **Réglage de décision limité :** les facteurs choisis sur la validation ont un effet quasi nul sur l’IoU du test.
- **Validation indépendante à renforcer :** les 21 événements du jeu de test étaient exclus de l’entraînement et des choix de validation de cette version, mais ils avaient déjà été consultés lors d’une version antérieure du projet (V4). Une confirmation sur de nouveaux événements jamais examinés serait donc nécessaire.
- **Scènes sans feu :** l’ajout de scènes sans incendie permettrait de mieux mesurer le taux de fausses alertes en conditions sans feu.

## Auteurs [Contributors]

Projet réalisé en collaboration par :

- **Khalil Abboud** — [GitHub](https://github.com/KhalilAbboud)
- **Zakaria Ghazi** — [GitHub](https://github.com/ZAKARIAGHAZI)

---

*SatFire CNN — Étude de la segmentation des feux actifs et des zones brûlées à partir d’images satellites VIIRS.*

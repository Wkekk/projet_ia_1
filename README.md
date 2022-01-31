# Intelligence artificielle et régression linéaire

## Rappel sur les régressions

La régression linéaire est un modèle d'étude statistique cherchant à mettre en lumière une relation entre une variable, dite expliquée, et une ou plusieurs autres variables, dites explicatives. La régression linéaire simple concerne les variables expliquées par une seule autre variable et la régression multiple celles expliquées par plusieurs, la régression polynomiale est quand à elle un cas particulier de la régression multiple dans lequel on explique une variables à l'aide des puissances d'une autre.

### Mise en place d'une régression

Pour mettre en place une régression linéaire dans laquelle on cherche à expliquer une variable *y* à l'aide d'une variable *x*, on va chercher le couple de réels *(a,b)* de la fonction *f(x) = ax+b* de sorte que la droite de la fonction *f* s'approche le plus possible de celle de *y*.
Pour faire cela nous allons transformer cette équation en équation matricielle : *Y = Xθ*, où *Y* est un vecteur colonne de dimension *n* constitué des valeurs des données à expliquer, *X* une matrice 2x*n* dont la première colonne reprend les données explicatives et la deuxième n'est composée que de 1, *θ* et un vecteur de dimension 2 représentant le couple *(a,b)*.
Une fois cette équation posée nous calculerons son gradient afin de mettre en place la descente du gradient et de pouvoir nous approcher petit à petit de la meilleure solution.

Les régressions multiples et polynomiales reprennent un fonctionnement similaire en augmentant les dimensions de *X* et *θ*. 

### Évaluation des modèles

Les régressions servent donc à mettre en place des modèles d'analyse de données mais nous avons besoin de savoir si ces modèles sont justes ou non. Pour cela nous avons différents outils, dans ce projet pour évaluer la régression simple nous avons utilisé le coefficient de détermination qui repose sur une évaluation de la différence entre les valeurs obtenues et celles déjà existantes et l'écart type. Pour les régressions multiples et polynomiales nous avons utilisé l'erreur quadratique moyenne qui s'évalue à l'aide du biais et de la variance du modèle.


## Le programme :


Le programme consiste en un ensemble de fonctions qui, une fois exécutées, permettent d'effectuer une régression linéaire.
Il utilise seulement les bibliothèque Numpy, pour le calcul, et Pandas pour la lecture des données (ainsi que SciKit Learn mais uniquement pour les calculs d'erreur quadratique des régressions multiples et polynomiales)

### La fonction `model()`

Cette fonction a pour but de calculer le modèle de donnée obtenu grâce à la régression pour multiplication matricielle. 

### La fonction `func_cout()`

Cette fonction calcule le coût du modèle, elle applique en version matricielle la formule *(Xθ-Y)/2n*

### La fonction `gradient()`  

La fonction `gradient()` calcule le gradient évoqué plus tôt en appliquant la formule *(TX(Xθ-Y))/n* où *TX* est la transposée de *X*

### La fonction `descente_gradient()`

Cette fonction effectue comme son nom l'indique la descente du gradient, c'est à dire qu'elle va modifier petit à petit *θ* pour que la fonction *f(x)=ax+b* s'approche le plus possible de *y*. Elle calcule également le coût de chaque itération à l'aide de `func_cout()` et stocke les valeurs obtenues afin que l'on puisse tracer la courbe de l'évolution du coût en fonction du nombre d'itérations.

### La fonction `coefficient_determination()`

Cette fonction a pour but de calculer le coefficient de détermination du modèle obtenu à l'aide de la régression simple afin d'évaluer le modèle en question. Nous utilisons pour ça la formule ` r_sqrd = 1 - s_fx / s_moy_y` avec ` s_fx = (Y[i] - (theta[0] * X[i][0] - theta[1])) ** 2` et `s_moy_y = (Y[i] - moy_y) ** 2` qui représentent respectivement l'écart entre les valeurs du modèle et celle des données et l'écart type des données.

### L'exécution du programme : la régression simple

Pour effectuer le calcul de la régression nous commençons par lire les données du fichier `regression_simple.csv` à l'aide de la bibliothèque Pandas. Une fois les données mises dans un dataframe nous les répartissons entre *Xµ et *Y*, les notes dans *Y*, car ce sont les données que nous voulons expliquer,  et le temps de révision dans *X*, car ce sont les données explicatives. la création de *Y* se fait avec `y = np.asarray(df['note'])` et celle de *Xµ avec `for i in range (len(df)) :
    x.append([df['heure_rev'][i], 1])`
où `x` est une liste vide dans laquelle on insère des tuples `(heure_rev, 1)` pour pouvoir procéder aux calculs matriciels. L'initialisation de *θ* se fait en créant un couple de valeurs aléatoires : `theta = np.random.random(2)`.

Une fois ces initialisations faites nous appelons la fonction de descente de gradient : `grad_desc = descente_gradient(x, y, theta, 0.001, 100)` avec un pas de 0.001 et pour 100 itérations. Nous récupérons le *θ* ayant subi la descente et le tableau du coût par itération dans `grad_desc`. Après cela nous calculons le coefficient de détermination avec `coefficient_determination()` et utilisons la bibliothèque matplotlib pour tracer la courbe d'évolution du coût par itération `pltc = plt.plot(cout)` avec `cout = grad_desc[1]` (la liste des valeurs par itération).


### L'exécution du programme : la régression multiple

Comme pour la régression simple nous commençons par initialiser les valeurs de *X*, *Y* et *θ*. Pour cela nous utilisons la même méthode pour *Y*, mais pour *X* nous sélectionnons d'abord les variables explicatives les plus intéressantes afin d'éviter des overflows liés à un trop grand nombre de valeurs lors des différents calculs du programme. Pour cela nous utilisons le coefficient de Pearson et la bibliothèque Pandas `val = df[i].corr(df['MEDV'], method='pearson')`. Nous avons choisi de sélectionner les variables ayant un taux de corrélation avec `MEDV` (les données à expliquer) supérieur à 60%. 
Une fois cette sélection faite nous reprenons le même processus d'initialisation que pour la régression simple en adaptant la taille des différentes matrices et vecteurs et en normalisant les valeurs des variables sélectionnées pour éviter de futurs overflows et pour avoir de meilleurs résultats . Ensuite nous suivons le même chemin que lors de la régression simple mais en utilisant l'erreur quadratique *via* SciKit-Learn au lieu du coefficient de détermination pour évaluer notre modèle. Enfin nous affichons nos résultats obtenus sous forme de nuages points à l'aide de matplotlib (avec en bleu les données et en orange le modèle. En exécutant le programme par la console le graphique est plus lisible que sur jupyter).

### L'exécution du programme : la régression polynomiale

Pour cette régression nous appliquons une méthode similaire à la régression simple pour l'initialisation du fichier `Position_Salaries.csv` mais en créant une matrice de dimension supérieure avec les puissances de la variable explicative (jusqu'au degré 3) pour *X* et nous adaptons la taille des vecteurs en fonction. Pour le fichier `qualite-vin-rouge.csv` la méthode utilisée est plus proche de celle de la régression multiple, en effet nous sélectionnons la variable explicative par son taux de corrélation de Pearson avec la variable expliquée puis nous créons une matrice *X* avec les puissances de cette variable.
`for i in range (deg+1) :
    x.append(df['alcool']**i)`
Ensuite, la régression polynomiale étant un cas particulier de la régression multiple, nous appliquons la même méthode que pour la régression multiple.


## Analyse des résultats

### régression simple

coefficient de détermination : 0.9994310728769705
Ce coefficient de détermination est très proche 1,  ce qui veut dire que le nuage de points des notes est très resserré autour de la droite obtenue par la régression, et par extension, que le modèle est bon.


[](/image/cout_iteration.png) "Régression simple sans scikit"

Le programme a été lancé pour 100 itérations de descente du gradient avec un pas de 0.001, on constate sur l'évolution du coût que la courbe est similaire à *f(x)=1/x* et que le coût semble tendre vers 0. Une asymptote *y=0* est présente car le coût, même s'il diminue autant que possible, ne peut pas atteindre 0.


### Régression multiple

Erreur quadratique : 0.024078444850824082

L'erreur quadratique obtenue est proche de 0, ce qui indique que le modèle est assez bon.

Le programme à été exécuté pour 10000 itérations de descente du gradient avec avec un pas de 0.001. Grâce à la représentation graphique, on constate que le modèle forme un plan qui intersecte bien les données. De plus si l'on change l'angle de vue on constate que les données calculées reprennent bien la forme des données source, mais sans avoir la même répartition sur l'axe z.

[](/image/reg_multiple_1.png) "régression multiple sans scikit"
[](/image/reg_multiple_2.png) "régression multiple sans scikit"

### régression polynomiale

#### Position_Salaries

erreur quadratique moyenne : 0.008084594527487578

L'erreur quadratique moyenne est proche de 0, ce qui semble indiquer un bon modèle.

Le programme a été exécuté pour 10000 itérations de de la descente de gradient avec un pas de 0.001, on constate sur le graphique que la courbe polynomiale est proche de celle des données réelles mais elle est loin d'être parfaite, notamment concernant les extrema. En effet le minimum a une valeur négative, proche zéro certes, et le maximum a un écart de 20 points avec la valeur réelle.

[](/image/reg_poly_salaire.png) "régression polynomiale Position_Salaries sans scikit"

#### qualité-vin-rouge

erreur quadratique : 0.02252703461002281
L'erreur quadratique étant proche de 0, elle semble indiquer un bon modèle.

Le programme a, comme pour l'autre fichier, été exécuté pour 10000 itérations et avec un pas de 0.001. Cependant comme le montre la représentation graphique, le modèle ne semble pas convenir. Il semblerait que la régression ne soit pas la bonne méthode pour analyser les données du fichier `qualite-vin-rouge.csv`

[](/image/reg_poly_vin.png) "régression polynomiale qualite-vin-rouge sans scikit"

## Régression linéaire avec scikit learn:

### La régression simple:
J'ai obtenu un graphique avec une bonne prédiction.La ligne bleu, représentant mon modèle, est proche du nuage de points, ce qui veut dire que le modèle est bon.
Comme le démontre la MSE (= mean squared error) qui est de 0.05611556400762798, le modèle est très bon car lplus le résultat est proche de 0 meilleur est le modèle.
La MAE (= mean absolute error) est de 0.16289645323167729 vu que cette erreur est proche de zéro cela indique que le modèle est performant.
    
[]( /image/reg_simple_scikit.png) "régression simple avec scikit"
    
### La régression multiple:
Le modèle obtenu forme un plan qui intersecte les données, on peut constater que le nuage de points obtenu se superpose  bien avec celui des données. Comme le dit la MSE en dessous du graphique le modèle n'est pas performent, la valeur étant d'un ordre de grandeur similaire aux données. Résultat = 0.7099743395705382
[](/image/graph_multiple_scikit.png) "régression multiple avec scikit"
   
### La régression polynomiale:
    
#### Qualité vin rouge:
    
Graphiquement on constate que les résultats obtenus ne sont pas modélisables c'est probablement lié au fait  que la régression ne soit pas la bonne méthode pour traiter ces données.   
[](/image/graph_alcool1_scikit.png)"régression polynomiale  qualite-vin-rouge avec scikit"
[](/image/graph_poly_vin_scikit.png) "régression polynomiale qualite-vin-rouge avec scikit"
#### Position salaries:
    
    Graphiquement les prédictions ont l'air bonnes mais la MSE semble incohérente car étant de 
5.11048629132908 ce qui, comparé aux valeurs des données, est énorme.
[](/image/graph_poly_salaire_scikit_1.png) "régression polynomiale Position_Salaries avec scikit"
[](/image/graph_poly_salaire_scikit_2.png)"régression polynomiale Position_Salaries avec scikit"
        
## Comparaison des résultats:
        
        Les résultats obtenus avec et sans scikit sont similaires, mais on peut remarquer une meilleur précision sur les exécutions sans scikit, ces écarts sont probablement dus au nombre d'itérations de la descente du gradient, en effet pour les régressions multiples et polynomiales, le programme exécute 10000 tours de boucle dans la descente, ce qui a forcément un impact bénéfique sur la précision mais est également plus couteux en ressources.
Nous remarquons également le même type de problèmes avec les deux méthodes sur le fichier `qualite-vin-rouge.csv`, à savoir que les rendus graphiques ne sont pas lisibles malgré de bons résultats en terme d'erreur quadratique, nous pensons que ces problèmes sont liés au fait que la méthode de la régression ne soit pas adaptée pour traiter les données de ce fichier.


## Conclussion:

### Loic

    Pendant ce brief j'ai appris à me servir de scikit-learn malgré des lacune en maths car sans Quentin les méthodes à la main auraient été très laborieuses de mon côté. Je me suis aidé des cours de machine learning sur internet pour comprendre mieux ce que je faisait et surtout comprendre quelles données je devais récupérer. J'ai eu beaucoup de difficultés sur le brief c'est surtout au niveau maths et compréhension du travail demandé. Après le projet je me sens relativement plus a l'aise avec scikit mais à la rue avec les méthodes mathématiques. Mais à côté de ça il est vrai que je me sens beaucoup plus l'aise en code pur.



### Quentin 

    Ce projet m'aura permis de mieux comprendre le fonctionnement de la régression linéaire et ses étapes. Il m'aura également permis de me familiariser davantage avec les bibliothèque numpy et scikit-learn de python.

    Certains aspects du projet ont pu poser problème, sur la programmation des régressions sans sci-kit notamment, que ce soit des erreurs d'inattention dans l'écriture de fonctions (des incrémentations oubliées ou des conditions d'arrêt éronnées) ou des erreurs liées au traitement des données (mauvaise sélection de variables explicatives ou non normalisation des données) qui ont entrainé des overflows dans les temps d'exécution.

    Je pense que ce projet m'aura permis d'apprendre à voir venir des erreurs évitables et à mieux gérer des projet morcelés en plusieurs parties.











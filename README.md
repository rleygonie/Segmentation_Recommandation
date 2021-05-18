PROJET : Segmentation de client et recommandation de films

Notre projet utilise trois sources de données : 


Données clients : (https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
Composé des colonnes suivantes: (CustomID, Gender, Age, Annual Income (k$),Spending Score (1-100))



Données des avis sur les films : (https://www.kaggle.com/rounakbanik/the-movies-dataset?select=ratings.csv
Composé des colonnes suivantes: (userId, movieId, rating, timestamp)



Données de description des films: (https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata.csv)
Composé des colonnes suivantes : (adult,belongs_to_collection,budget,genres,homepage,id,imdb_id,original_language,original_title,overview,popularity,poster_path,production_companies,production_countries,release_date,revenue,runtime,spoken_languages,status,tagline,title,video,vote_average,vote_count)



Afin de mieux connaître les différents types de clients du centre commercial nous allons tout d’abord les séparer en plusieurs groupes cohérents en utilisant l’algorithme de regroupement Kmeans. 
Nous appliquons RandomForest afin de classer un nouveau client dans un des groupes trouvés par Kmeans au préalable.
Une fois cette segmentation finalisée nous utilisons les précédents avis des clients pour leur recommander des nouveaux films à voir et à acheter dans la boutique du centre commercial.


Pour la partie segmentation client :

	Algorithme de clustering utilisé pour la segmentation de clients en plusieurs groupes : comparaison entre KMeans et Bisecting k-means 

	Algorithme de classification pour classer un nouveau client dans un des groupes trouvés par l’étape précédente : comparaison entre RandomForest et DecisionTree.

	L'utilisation de ces deux algorithmes constituent notre approche hybride.


Pour la partie recommandation de films :

	Jointure entre les clients et les notes de films (création de notre table pour traiter le sujet de la recommandation).

	Utilisation de l'algorithme de filtrage collaboratif pour recommander des films en fonction des clients.



Notre répertoire est organisé de manière suivante: 

	input_data : les données clients et de films pour l’apprentissage de nos modèle

	src : contenant les sources python 
		main.ipynb : le fichier principal contenant nos modèles

		utils : répertoire des fonctions utilisées dans le main
		
		annexe : répertoire contenant les analyses exploratoires de nos données ainsi que les fonctions utilisées pour comparer les modèles

	présentation : rapport et présentation des résultats de notre projet

	README : informations sur le projet



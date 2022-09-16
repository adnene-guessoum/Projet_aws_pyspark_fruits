# Projet_aws_pyspark_fruits
setting up a big data environment with aws emr and pyspark for image feature extraction and processing.

<h3> Contenu du repo.: </h3>

P8_resnet_loading_S3 : script pour stocker le modèle utilier (ResNet50 sans top layer) sur s3.

P8_script_echantillon_img : script pour tirer au sort et stocker les images de fruits localement et sur S3 en vue du traitement.

P8_script_local : script de réalisation du projet localement pour prise en main de pyspark, boto3, etc...

EMR_setup: document config, bootstrap et script de réalisation des calculs dans le cloud en pyspark. tout le nécessaire pour pouvoir réaliser le projet de manière distribuée sur un cluster emr configuré correctement. dossier également stocker sur s3 pour lancement du cluster et soumission spark.

<h3> Jeu de données : </h3>

Données disponibles sur kaggle contenant images de fruits et labels associées: https://www.kaggle.com/datasets/moltean/fruits

Nombre total d’images: 90483 ;
Set d’entraînement : 67692 images (un fruit/légume par image) ; set de test : 22688 images (un fruit/légume par image).
Nombre de classes: 131 (types de fruits ou légumes).
Taille des mages: 100x100 pixels.

<h3> Outils et procédure : </h3>

- Mise en place d’un environnement Linux (WSL2, Ubuntu) avec AWS CLI configurée (clefs secrètes, etc...) pour interagir avec aws.
- Construction des différents scripts python (pyspark) pour gérer l’ensemble des étapes du projet sans interventions manuels sur les données.
- Chargement (dans un compartiment S3) :
  - des données (images tirées au sort pour rester raisonnable en terme de budget)
  - du script python pour les différentes opérations à réaliser.
  - du fichier de configuration du cluster EMR pour un environnement pyspark (config_EMR.json) 
  - du fichier d’amorçage pour installer au lancement des serveurs sur toutes les machines du clusters les dépendances nécessaires pour le script (bootstrap.sh).
- Lancement d’un cluster EMR configuré : m5.xlarge, emr 6.7.0, Spark Hadoop 3.2.1, Tensorflow...
- Accès à la master node via ssh depuis la ligne de commande. Lancement du script : « spark-submit »
- Vérification de l’écriture des résultats dans s3 (parquet) et résiliation du cluster.

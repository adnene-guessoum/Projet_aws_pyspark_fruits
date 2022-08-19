# général
import pandas as pd
import numpy as np
import random
import os
import shutil

Image_path = "Dev/Projet_8/archive/fruits-360_dataset/fruits-360/Training"

def random_image_selection(num_img, path):
    os.chdir(path)
    thisdir = os.getcwd()

    #récupérer tous les path vers images (.jpg) du folder archive télécharger depuis kaggle (Training dataset uniquement)
    #ajout de underscore.sh dans le dossier Training pour changer les espaces dans les noms de dossiers par underscore en vue de S3 
    data = []
    for r, d, f in os.walk(thisdir):
        for file in f:
            if ".jpg" in file:
                data.append(os.path.join(r,file))

    #récupérer aléatoirement quelques images pour les insérer dans un nouveau training à envoyer dans S3
    reduced_data = random.sample(data, num_img)
    #récupérer labels et nom img
    labels_img_lst = []
    for i in reduced_data :
        split_path = i.split("/")
        labels_img_folder = split_path[-2:]
        labels_img_lst.append(labels_img_folder)
    #placer les images sélectionnées dans un nouveau dossier
    dossier_source = path
    dossier_arrivee = r"/home/adnene/Dev/Projet_8/echantillon-img"

    for i,j in zip(reduced_data, labels_img_lst):
        source = i
        destination = os.path.join(dossier_arrivee, j[0], j[1])
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source, destination)
    print('Done')

def random_selection_images_files(num_img, num_files, pth):
    
    pass

if __name__ == '__main__':
    random_image_selection(300, Image_path)
# général
import random
import os
import sys
import shutil

import boto3


Image_path = "Dev/Projet-8/archive/fruits-360_dataset/fruits-360/Training"

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
    dossier_arrivee = r"/home/adnene/Dev/Projet-8/echantillon-img"

    for i,j in zip(reduced_data, labels_img_lst):
        source = i
        destination = os.path.join(dossier_arrivee, j[0], j[1])
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source, destination)
    print('Done')

def targetted_random_image_selection(num_img, num_files, pth, to_S3):
    os.chdir(pth)
    thisdir = os.getcwd()
    list_subdir = [f for f in os.listdir(thisdir) if os.path.isdir(os.path.join(thisdir, f))]

    chosen_fruits = random.sample(list_subdir, num_files)

    data = []
    for i in chosen_fruits:
        path_to_jpg = os.path.join(thisdir,i)
        onlyfiles = [f for f in os.listdir(path_to_jpg) if os.path.isfile(os.path.join(path_to_jpg, f))]
        chosen_image = random.sample(onlyfiles, num_img)
        data.append(chosen_image)

    print(data)
    print("---------------------------------------------------")
    print(chosen_fruits)

    
    #placer les images sélectionnées dans un nouveau dossier
    dossier_source = r"/home/adnene"
    dossier_arrivee = r"/home/adnene/Dev/Projet-8/echantillon-img"

    for i,j in zip(chosen_fruits, data):
        for z in j:
            source = os.path.join(dossier_source, pth, i, z)
            destination = os.path.join(dossier_arrivee, i, z)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy(source, destination)
    print('Done')

    if to_S3 == True :
        os.chdir(dossier_arrivee)
        thisdir = os.getcwd()

        AWS_REGION = "eu-west-1"
        client = boto3.client("s3", region_name=AWS_REGION)

        #bucket creation :
        bucket = 'echantillon-img'
        location = {'LocationConstraint': AWS_REGION}
        response = client.create_bucket(Bucket=bucket, CreateBucketConfiguration=location)

        print("Amazon S3 bucket has been created")


        destination = 'echantillon-img'

        for r, d, f in os.walk(thisdir):
            for file in f :
                path = os.path.join(r,file)
                r_path = os.path.relpath(path, thisdir)
                s3_path = r_path

                try:
                    client.head_object(Bucket=bucket, Key=s3_path)
                    print("le path existe déja sur s3. abandon %s..." % s3_path)

                except:
                    print('CHARGEMENT : %s ...' % s3_path)
                    client.upload_file(path, bucket, s3_path)

        print("Done for S3")

if __name__ == '__main__':
    #random_image_selection(300, Image_path)
    targetted_random_image_selection(3, 4, Image_path, to_S3=True)
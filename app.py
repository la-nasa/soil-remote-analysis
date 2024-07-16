import streamlit as st
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from skimage import exposure, segmentation
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image  # Importer PIL pour la manipulation des images

# Fonction pour sauvegarder une image en utilisant rasterio
def save_image(data, filepath):
    transform = from_origin(0, 0, 1, 1)  # Modification si nécessaire
    with rasterio.open(
        filepath, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype, transform=transform
    ) as dst:
        dst.write(data, 1)

# Fonction pour charger une bande avec une réduction de résolution
def load_band(i, image_folder, save_folder):
    filename = os.path.join(image_folder, f"LC08_L2SP_187056_20240103_20240113_02_T1_SR_B{i}.tif")
    if not os.path.exists(filename):
        st.error(f"Fichier non trouvé: {filename}")
        return None
    with rasterio.open(filename) as src:
        st.write(f"Chargement de {filename}")
        data = src.read(1)
        # Sauvegarde de l'image originale de la bande
        original_save_path = os.path.join(save_folder, f"band_{i}_original.tif")
        save_image(data, original_save_path)
        data_resized = cv2.resize(data, (data.shape[1] // 8, data.shape[0] // 8))  # Réduction de taille
        # Sauvegarde de l'image redimensionnée de la bande
        resized_save_path = os.path.join(save_folder, f"band_{i}_resized.tif")
        save_image(data_resized, resized_save_path)
        return data_resized

# Fonction pour convertir les images TIFF en PNG
def convert_tiff_to_png(tiff_path, png_path):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        # Convertir les données en uint8
        data_uint8 = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
        img = Image.fromarray(data_uint8)
        img.save(png_path)

# Chargement du modèle
model = tf.keras.models.load_model('models/cnn_model_optimized.h5')

st.title("Application d'Analyse du Sol")
st.sidebar.title("Options")

# Téléchargement de fichier
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier d'image satellite", type=["tif"])

if uploaded_file is not None:
    # Sauvegarde du fichier téléchargé dans un dossier temporaire
    image_folder = "temp_images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    with open(os.path.join(image_folder, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Chargement des bandes
    save_folder = "processed_images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    band_data = [load_band(i, image_folder, save_folder) for i in range(1, 8)]
    
    # Vérification si des bandes sont None
    if any(band is None for band in band_data):
        st.error("Erreur lors du chargement d'une ou plusieurs bandes.")
    else:
        # Conversion de la liste en tableau numpy
        band_data = np.array(band_data, dtype=np.uint8)
        st.write("Dimensions des données des bandes:", band_data.shape)

        # Extraction des canaux Rouge (bande 4) et Proche Infrarouge (bande 5)
        red = band_data[3]
        nir = band_data[4]

        # Calcul de l'NDVI
        denominator = nir + red
        denominator[denominator == 0] = 1  # Pour éviter la division par zéro
        ndvi = (nir - red) / denominator
        st.write("NDVI calculé, dimensions:", ndvi.shape)

        # Sauvegarde de l'image NDVI
        ndvi_save_path = os.path.join(save_folder, "NDVI.tif")
        save_image(ndvi, ndvi_save_path)

        # Redimensionnement de l'image pour réduire la consommation de mémoire
        scale_factor = 0.05  # Réduction de la taille de l'image à 5% de sa taille originale
        gray_image_resized = cv2.resize((red + nir) / 2, (0, 0), fx=scale_factor, fy=scale_factor)
        gray_image_resized = exposure.rescale_intensity(gray_image_resized)

        # Sauvegarde de l'image prétraitée (redimensionnée)
        preprocessed_save_path = os.path.join(save_folder, "preprocessed_image.tif")
        save_image(gray_image_resized, preprocessed_save_path)

        # Segmentation de l'image pour extraire les informations sur la texture du sol
        regions = segmentation.felzenszwalb(gray_image_resized, scale=100, sigma=0.5, min_size=50)
        st.write("Segmentation de l'image complétée")

        # Sauvegarde de l'image segmentée
        segmented_save_path = os.path.join(save_folder, "segmented_image.tif")
        save_image(regions, segmented_save_path)

        # Conversion des images TIFF en PNG
        converted_images = []
        for i in range(1, 8):
            tiff_path = os.path.join(save_folder, f"band_{i}_original.tif")
            png_path = os.path.join(save_folder, f"band_{i}_original.png")
            convert_tiff_to_png(tiff_path, png_path)
            converted_images.append(png_path)
        
        resized_converted_images = []
        for i in range(1, 8):
            tiff_path = os.path.join(save_folder, f"band_{i}_resized.tif")
            png_path = os.path.join(save_folder, f"band_{i}_resized.png")
            convert_tiff_to_png(tiff_path, png_path)
            resized_converted_images.append(png_path)

        # Conversion des autres images
        ndvi_png_path = os.path.join(save_folder, "NDVI.png")
        convert_tiff_to_png(ndvi_save_path, ndvi_png_path)

        preprocessed_png_path = os.path.join(save_folder, "preprocessed_image.png")
        convert_tiff_to_png(preprocessed_save_path, preprocessed_png_path)

        segmented_png_path = os.path.join(save_folder, "segmented_image.png")
        convert_tiff_to_png(segmented_save_path, segmented_png_path)

        # Affichage des images
        st.subheader("Images Originales et Traitées")
        def display_images(image_paths, captions, width=150):
            images = [Image.open(path).convert("RGB") for path in image_paths]
            st.image(images, caption=captions, width=width)
        
        display_images(converted_images, [f"Bande {i} Originale" for i in range(1, 8)])
        display_images(resized_converted_images, [f"Bande {i} Redimensionnée" for i in range(1, 8)])
        display_images([ndvi_png_path], ["Image NDVI"])
        display_images([preprocessed_png_path], ["Image Prétraitée"])
        display_images([segmented_png_path], ["Image Segmentée"])

        # Génération des prédictions
        resized_image = cv2.resize(band_data[3], (32, 32)).reshape(-1, 32, 32, 1)
        predictions = model.predict(resized_image)

        st.subheader("Prédictions")
        st.write(predictions)

        # Tracé des métriques d'évaluation
        st.subheader("Métriques d'Évaluation du Modèle")
        history_df = pd.read_csv('results/training_history_optimized.csv')
        fig, ax = plt.subplots()
        ax.plot(history_df['loss'], label='Perte')
        ax.plot(history_df['val_loss'], label='Perte de Validation')
        ax.set_xlabel('Époque')
        ax.set_ylabel('Perte')
        ax.legend()
        st.pyplot(fig)

        st.write("Entraînement et évaluation du modèle terminés.")

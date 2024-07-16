import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from skimage import exposure, segmentation
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Function to save an image using rasterio
def save_image(data, filepath):
    transform = from_origin(0, 0, 1, 1)  # Modify as needed
    with rasterio.open(
        filepath, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype, transform=transform
    ) as dst:
        dst.write(data, 1)

# Function to load a band with resolution reduction
def load_band(i, image_folder, save_folder):
    filename = os.path.join(image_folder, f"LC08_L2SP_187056_20240103_20240113_02_T1_SR_B{i}.tif")
    with rasterio.open(filename) as src:
        print(f"Loading {filename}")
        data = src.read(1)
        # Save the original band image
        original_save_path = os.path.join(save_folder, f"band_{i}_original.tif")
        save_image(data, original_save_path)
        data_resized = cv2.resize(data, (data.shape[1] // 8, data.shape[0] // 8))  # Reduce size further
        # Save the resized band image
        resized_save_path = os.path.join(save_folder, f"band_{i}_resized.tif")
        save_image(data_resized, resized_save_path)
        return data_resized

# Path to the folder containing images of different bands
image_folder = "C:/Users/tmdev/Documents/SOIL ANALYSIS/LC08_L2SP_187056_20240103_20240113_02_T1/"

# Folder to save the images at different stages
save_folder = "C:/Users/tmdev/Documents/finalapp/save_folder"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Load data from different bands in parallel
band_data = [load_band(i, image_folder, save_folder) for i in range(1, 8)]

# Convert the list to a numpy array
band_data = np.array(band_data, dtype=np.uint8)  # Use efficient data type
print("Band data dimensions:", band_data.shape)

# Extract Red (band 4) and Near-Infrared (band 5) channels
red = band_data[3]
nir = band_data[4]

# Calculate NDVI
denominator = nir + red
denominator[denominator == 0] = 1  # To avoid division by zero
ndvi = (nir - red) / denominator
print("NDVI calculated, dimensions:", ndvi.shape)

# Save NDVI image
ndvi_save_path = os.path.join(save_folder, "NDVI.tif")
save_image(ndvi, ndvi_save_path)

# Resize the image to reduce memory consumption
scale_factor = 0.05  # Reduce image size further to 5% of its original size
gray_image_resized = cv2.resize((red + nir) / 2, (0, 0), fx=scale_factor, fy=scale_factor)
gray_image_resized = exposure.rescale_intensity(gray_image_resized)

# Save preprocessed (resized) image
preprocessed_save_path = os.path.join(save_folder, "preprocessed_image.tif")
save_image(gray_image_resized, preprocessed_save_path)

# Segment the image to extract information on soil texture
regions = segmentation.felzenszwalb(gray_image_resized, scale=100, sigma=0.5, min_size=50)
print("Image segmentation completed")

# Save segmented image
segmented_save_path = os.path.join(save_folder, "segmented_image.tif")
save_image(regions, segmented_save_path)

# Load soil data
soil_data = pd.read_csv("C:/Users/tmdev/Documents/SOIL ANALYSIS/Datasets/projectmodel/soil_data.csv")
print("Soil data dimensions:", soil_data.shape)

# Example: Integrate NDVI or other features with soil data (mean NDVI value for simplicity)
soil_data['Mean_NDVI'] = np.mean(ndvi)

# Encode categorical variables
soil_data_encoded = pd.get_dummies(soil_data, drop_first=True)

# Separate features and target variables
X = soil_data_encoded.drop(columns=['Nitrogen_N_ppm', 'Phosphorus_P_ppm', 'Potassium_K_ppm', 'Depth_cm', 'pH', 'Organic_Matter_%', 'Moisture_Content_%', 'Bulk_Density_g/cm³', 'Electrical_Conductivity_dS/m', 'Porosity_%', 'Water_Holding_Capacity_%'])
y = soil_data_encoded[['Nitrogen_N_ppm', 'Phosphorus_P_ppm', 'Potassium_K_ppm', 'Depth_cm', 'pH', 'Organic_Matter_%', 'Moisture_Content_%', 'Bulk_Density_g/cm³', 'Electrical_Conductivity_dS/m', 'Porosity_%', 'Water_Holding_Capacity_%']]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and test sets")

# Data generator function
def data_generator(X_indices, y_data, band_data, batch_size=32):
    while True:
        for start in range(0, len(X_indices), batch_size):
            end = min(start + batch_size, len(X_indices))
            batch_indices = X_indices[start:end]
            batch_images = []
            for idx in batch_indices:
                resized_image = cv2.resize(band_data[3], (32, 32))  # Reduce size further
                batch_images.append(resized_image)
            batch_images_array = np.array(batch_images).reshape(-1, 32, 32, 1)
            yield batch_images_array, y_data.iloc[batch_indices].values

# Indices for training and test sets
X_train_indices = np.arange(X_train.shape[0])
X_test_indices = np.arange(X_test.shape[0])

# Data generators
train_gen = data_generator(X_train_indices, y_train, band_data, batch_size=32)
test_gen = data_generator(X_test_indices, y_test, band_data, batch_size=32)

# Build the optimized CNN model
model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),  # Reduced filters and size
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),  # Reduced dense layer size
    Dropout(0.5),
    Dense(11, activation='linear')  # Assuming 11 output features
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model with generators
steps_per_epoch = len(X_train) // 32
validation_steps = len(X_test) // 32
history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=25, validation_data=test_gen, validation_steps=validation_steps)

# Save the model
model.save('models/cnn_model_optimized.h5')

# Evaluate the model
loss, mae = model.evaluate(test_gen, steps=validation_steps)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Make predictions
y_pred = model.predict(test_gen, steps=validation_steps)

# Regression metrics
mse = np.mean((y_test.values[:len(y_pred)] - y_pred) ** 2, axis=0)
mae = np.mean(np.abs(y_test.values[:len(y_pred)] - y_pred), axis=0)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Save history
history_df = pd.DataFrame(history.history)
history_df.to_csv('results/training_history_optimized.csv', index=False)

print("Model training and evaluation completed.")

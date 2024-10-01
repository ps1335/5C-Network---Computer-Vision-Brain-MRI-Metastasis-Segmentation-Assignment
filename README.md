# 5C-Network---Computer-Vision-Brain-MRI-Metastasis-Segmentation-Assignment
Welcome to the Brain MRI Metastasis Segmentation project! This project leverages advanced deep learning architectures like Nested U-Net and Attention U-Net to segment metastasis regions in brain MRI images. We use TensorFlow/Keras for model training, FastAPI for serving the model as a web service, and Streamlit for creating a simple and interactive UI.

Project Overview
This project focuses on automatically segmenting metastasis regions in brain MRI images using two powerful deep learning architectures: Nested U-Net (U-Net++) and Attention U-Net. The segmented regions can assist medical professionals in diagnosing and treating cancer metastasis in the brain.

We utilize:

Nested U-Net: A more advanced version of U-Net with better feature fusion.
Attention U-Net: Uses attention mechanisms to focus on important regions of the image.
Features
Preprocessing: Automatically enhances and normalizes MRI images using CLAHE.
Deep Learning Models: Implements both Nested U-Net and Attention U-Net.
FastAPI Backend: Allows real-time prediction of MRI image segmentations.
Streamlit UI: Provides a user-friendly interface to upload images and see the segmentation results.

Project Structure -
brain_mri_segmentation/
│
├── data_preprocessing.py    # Preprocessing of images and masks
├── models.py                # Nested U-Net and Attention U-Net architectures
├── train.py                 # Training and evaluation script
├── app.py                   # FastAPI server to serve model predictions
├── ui.py                    # Streamlit UI for user interaction
├── requirements.txt         # Project dependencies
└── README.md                # This file!



Future Improvements
Here are a few ideas to take this project to the next level:

Improve the UI: Add more interactivity or styling to the Streamlit interface.
Add Model Comparison: Let users choose between different models (Nested U-Net, Attention U-Net) to see how each performs on the same image.
Data Augmentation: Introduce data augmentation techniques to improve model performance and generalization.
Dockerize the Project: Containerize the application using Docker to make it easier to deploy.

import os
import shutil
from data_pipeline import *  # Dataset creation
from train_pipeline import *  # Model training
from predict_pipeline import *  # Model prediction

# Function to clean the results directory
def clean_results_directory(output_directory):
    # Iterate over all the files and folders in the directory
    for filename in os.listdir(output_directory):
        file_path = os.path.join(output_directory, filename)
        
        # Check if it's a file or directory and remove it accordingly
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path) 
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    print("All files and folders have been removed.")

def main():
    # Define the directory to clean
    results_directory = '/Users/adriancerezuelahernandez/Desktop/Q7/TAED2/project1/results' 
    # Step 0: Clean the results directory
    clean_results_directory(results_directory)

    # Step 1: Generate dataset YAML
    yaml_file_path = data()
    print(f"YAML file generated at: {yaml_file_path}")

    # Step 2: Train and evaluate the model
    model, metrics, model_path = train_eval(yaml_file_path)
    print(f"Model training complete. Metrics: {metrics}")

    # Step 3: Run prediction on a sample image
    image_path = '/kaggle/working/runs/detect/predict/00028.jpg' 
    predicted_image = predict(model_path, image_path)
    print(f"Prediction completed. Predicted image saved at: {predicted_image}")

if __name__ == '__main__':
    main()  # Start the pipeline

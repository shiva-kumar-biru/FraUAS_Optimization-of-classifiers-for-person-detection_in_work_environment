# Optimization of classifiers for Person Detection


## Project Overview

Efficiently detecting the presence of individuals in an office environment is crucial for optimizing energy consumption, ensuring safety, and enabling smart workplace applications. This project utilizes ultrasonic data collected via a Red Pitaya device to develop machine learning classifiers capable of reliably detecting occupancy.

To achieve this, the project integrates advanced data processing, robust machine learning models, and seamless deployment strategies. The solution is scalable and automated, making it suitable for real-world implementation in dynamic office settings.

## Project Structure

### 1. Data Processing and Model Training - `Convert_ADC_to_FFT_and_Model_Training.ipynb`

- **Purpose**: 
  - Converts the raw data received from the Red Pitaya in `.txt` format to `.csv` in the `conversion_from_txt_to_csv`
  - Converts Analog-to-Digital Converter (ADC) data into the frequency domain using Fast Fourier Transform (FFT).
  - Trains multiple machine learning classifiers: Random Forest, SVM, Logistic Regression, XGBoost, and Gradient Boosting.
  - Saves the trained models in the `models/` directory.

- **Usage**:
  - Run the `Convert_ADC_to_FFT_and_Model_Training.ipynb` script to process the data and train the models.
  - Ensure that the `.txt` files from the Red Pitaya are available in the appropriate directory before running the script.

### 2. Scenario-Based Model Testing - `Scenarios_test_for_classifers.ipynb`
- **Purpose**: 
  - Loads the pre-trained models from the `models/` directory.
  - Evaluates each model on specific test scenarios.
  - Provides detailed performance metrics, including confusion matrices, accuracy, precision, and recall for each scenario.

- **Usage**:
  - Run the `Scenarios_test_for_classifers.ipynb` script to test the models on scenario-specific data.
  - The script automatically loads the necessary models and applies them to the test data.


- **Purpose**:
  - Provides an interactive interface for users to evaluate model performance.
  - Allows users to select test data from the `data/panel_test/` folder and view the confusion matrix, metrics, and accuracy of each classifier for a chosen scenario.

- **Implementation**:
  - The GUI is built using the `Panel` library, offering a user-friendly way to interact with the models and visualize results.


### 3. Graphical User Interface (GUI)

- **Purpose**:

  - Provides an interactive interface for users to evaluate model performance.
  - Allows users to select test data from the `data/panel_test/` folder and view the confusion matrix, metrics, and accuracy of each classifier for a chosen scenario.
  
- **Implementation**:

  - The GUI is built using the Panel library, offering a user-friendly way to interact with the models and visualize results.
  - The GUI is dockerized and the container image is pushed to AWS Elastic Container Registry (ECR).
  - The application is deployed and hosted on AWS ECS Fargate, ensuring scalable and serverless execution.
  
- **Access**:

  - The GUI is running on AWS ECS and can be accessed through the provided endpoint 

    [URL for the GUI](panelalb-648001900.us-east-1.elb.amazonaws.com)

### 4. Deployment and Automation

- **Dockerfile**:

   - The Dockerfile in the root directory is used to containerize the application, including all necessary dependencies and configurations.

- **CI/CD Pipeline**:

   - A fully automated pipeline is implemented using GitHub Actions to build, test, push the Docker image to AWS ECR, and deploy the container on AWS ECS Fargate.
   - Pipeline configuration is located in the file: `.github/workflows/deploy_to_ECS.yml.`
 
- **Task Definition**:

   - The ECS task definition is stored in task-definition.json, which specifies how the container should run, including resources, networking, and environment variables. (`task-definition.json`)


  
###  5. Source Code:

   - All Python scripts and relevant code are located in the src/ folder.



  
## Installation Instructions to run on Local 

### 1. Ensure You Have Python and pip Installed

Make sure Python is installed on your system. You can check by running:

```bash
python --version
```
```bash 
pip --version
```

### 2. Navigate to Your Project Directory

Use the terminal or command prompt to navigate to the directory where your requirements.txt file is located. For example:
```bash 
cd path/to/your/project
```
### 3. Install the Dependencies

Run the following command to install all the dependencies listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```


### 4. Open the terminal, navigate to the folder where the panel-GUI code is present `gui_of_person_detection_classifiers.py`

### 5. Run the command `panel serve gui_of_person_detection_classifiers.py --show --autoreload `to start the GUI

### - **Usage**:
  - Launch the GUI through the provided script or environment.
  - Use the GUI to select and load test data
  - select any classifier from the dropdown list and click on **run** button to get the confusion matrix and the metrics of the particular data 
  - The GUI will display the results in the form of confusion matrix, accuracy, precision, recall
  - If you want to get the accuracies of all the classifiers as a barchart to compare the accuracies of the classifer , select each classifer and **run** and the click on the **accuracies** button to get the chart 


### - The GUI Example 
![gui](https://github.com/shiva-kumar-biru/FraUAS_Optimization-of-classifiers-for-person-detection_in_work_environment/blob/main/Images/panel_gui.png)

### - GUI output 
![gui_output](https://github.com/shiva-kumar-biru/FraUAS_Optimization-of-classifiers-for-person-detection_in_work_environment/blob/main/Images/panel_gui_output.png)

## Directory Structure

- **`models/`**: Directory where trained models are stored.
- **`dataset`**: Folders contains the Dataset and the Test Data set 
- **`dataset/panel_test/`**: Folder containing test data for GUI-based evaluation.
- **`Convert_ADC_to_FFT_and_Model_Training.ipynb`**: Script for data conversion, FFT processing, and model training.
- **`Scenarios_test_for_classifers.ipynb`**: Script for scenario-wise model testing.
- **`gui_of_person_detection_classifiers.py`**: GUI script for model evaluation and comparison.
-  **`dataprocessing_of_gui`**: This file contains the conversion process and the loaded trained models of the GUI 
-  **`.github/workflows/deploy_to_ECS.yml`**: GitHub Actions configuration file for automating CI/CD pipeline, including deployment to AWS ECS.
-  **`task-definition.json`**: ECS task definition file specifying container runtime details, resources, and environment configurations.
-  **`Dockerfile`**: File for containerizing the application, including dependencies and configurations.
- **`README.md`**: This readme file.














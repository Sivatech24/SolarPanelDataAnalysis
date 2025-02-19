# Predictive Maintenance in Solar Power Systems

# Problem Statement: 
Solar power plants play a crucial role in the transition toward clean energy. However, maintaining large solar farms is challenging, as breakdowns or inefficiencies can significantly reduce power output and cause financial losses. Traditionally, maintenance schedules are set based on time intervals, which may lead to over-maintenance or under-maintenance of equipment. The introduction of AI-driven predictive maintenance offers a solution that can help identify potential failures before they happen.

# Dataset:

https://www.kaggle.com/datasets/anikannal/solar-power-generation-data?datasetId=836676&sortBy=voteCount&select=Plant_1_Generation_Data.csv
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data?datasetId=836676&sortBy=voteCount&select=Plant_1_Weather_Sensor_Data.csv
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data?datasetId=836676&sortBy=voteCount&select=Plant_2_Generation_Data.csv
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data?datasetId=836676&sortBy=voteCount&select=Plant_2_Weather_Sensor_Data.csv

# Analysis
# *Plant 1 Generation Data
![Desktop Screenshot 2024 10 24 - 08 33 28 03](https://github.com/user-attachments/assets/2c6ee341-25a6-4b41-8289-8d71eac0fc31)

# *Plant 1 Weather Sensor Data
![Desktop Screenshot 2024 10 24 - 08 33 34 38](https://github.com/user-attachments/assets/f26672d1-7555-4664-a570-d2ee5b616be4)

*Plotting generation data
![1](https://github.com/user-attachments/assets/61d54bd2-a642-4762-992b-30fe1cd7fca8)

*Plotting weather data
![2](https://github.com/user-attachments/assets/590f8f1b-7ebe-4c23-929b-5d50b6d87e5d)

*Calculating DC Power Converted
![3](https://github.com/user-attachments/assets/1364b9b1-bf53-4cea-8541-8a0a780fda0c)

*Filtering for day time hours
![4](https://github.com/user-attachments/assets/ee83e8ef-2632-4d25-bb64-0d68231f75a4)

*Inverter performance analysis
![5](https://github.com/user-attachments/assets/964de037-8687-4b6c-830c-c0f68e68cfce)

*Inverter specific data
![6](https://github.com/user-attachments/assets/17db1dd1-977a-4ae6-9b3d-e29b5dcbc3c1)

*Daily yield analysis
![7](https://github.com/user-attachments/assets/0a3b33f7-e318-4f11-ad85-3a3fcafbdef9)

*ARIMA model
![8](https://github.com/user-attachments/assets/2786f6c5-6cd1-4666-8e12-5c5814ec2adc)

*Plotting ARIMA results
*SARIMA model
*Plotting SARIMA results
*Comparing ARIMA and SARIMA forecasts

# AI-Driven Sustainable Energy Management

## Overview
This repository contains code, datasets, and notebooks for analyzing and optimizing solar energy production using AI-driven techniques. It focuses on forecasting solar power generation, predicting inverter failures, and optimizing energy efficiency through supervised learning algorithms.

## Features
- **Solar Power Analysis**: Includes Jupyter notebooks for analyzing solar panel efficiency.
- **Failure Prediction**: Uses machine learning techniques like Random Forest and Logistic Regression.
- **Hyperparameter Tuning**: Implements k-Fold Cross-Validation for optimizing models.
- **Visualization**: Provides plots for data analysis and insights.
- **Web Application**: Python-based application for solar panel data analysis.

## Repository Structure
```
├── DataSet/                       # Contains datasets for analysis
├── Documentation/                 # Documentation files
├── ElectricityAnalysis/            # Scripts for analyzing electricity consumption
├── Images/                         # Images and plots for visualization
├── Jupyter Notebook/               # Jupyter Notebooks for data analysis and modeling
├── DataSets.txt                    # List of datasets used
├── LICENSE                         # License file
├── README.md                       # Repository documentation
├── RandomForestToPredectionOfFailure.ipynb  # Random Forest model for failure prediction
├── SolarPanelAnalysisPage.py       # Python script for analyzing solar panel data
├── SolarPanelDataAnalysis1.ipynb   # Solar panel data analysis notebook 1
├── SolarPanelDataAnalysis2.ipynb   # Solar panel data analysis notebook 2
├── SolarPlantAnalysisApp.py        # Application script for solar plant analysis
├── SolarPlantLogesticRegression.ipynb  # Logistic Regression model for analysis
├── SolarPowerPlantAnalysis.ipynb   # Comprehensive solar power plant analysis
├── first_plot.png                  # Sample visualization 1
├── second_plot.png                 # Sample visualization 2
├── hyperparametertuningprocess.ipynb  # Notebook for hyperparameter tuning
├── k-Fold Cross-Validation.ipynb    # Notebook for k-Fold validation
├── requirements.txt                 # List of dependencies
├── solarpanel.ipynb                 # General solar panel analysis notebook
├── spa.py                           # Supporting script for solar panel analysis
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Sivatech24/SolarPanelDataAnalysis.git
   cd SolarPanelDataAnalysis
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook:
   ```sh
   jupyter notebook
   ```

## Usage
- Open and run Jupyter notebooks to analyze solar energy data.
- Use `SolarPlantAnalysisApp.py` to run a Python-based analysis application.
- Modify and experiment with different machine learning models for prediction and optimization.

## Live Demo
- [Solar Plant Analysis App](https://huggingface.co/spaces/CodingMaster24/SolarPlantAnalysisApp)
- [Solar Panel Analysis Page](https://huggingface.co/spaces/CodingMaster24/SolarPanelAnalysisPage)

## Contributions
Contributions are welcome! Please submit a pull request with your changes.

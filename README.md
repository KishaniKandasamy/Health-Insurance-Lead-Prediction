# Wave Health-Insurance-Lead-Prediction

This application allows us to  preprocess raw datasets and train & predict the health Insurance Leads using Decision Tree classifier.

## Running this App Locally

### System Requirements

1. Python 3.6+
2. pip3

### 1. Run the Wave Server

New to H2O Wave? We recommend starting in the documentation to [download and run](https://h2oai.github.io/wave/docs/installation) the Wave Server on your local machine. Once the server is up and running you can easily use any Wave app.

### 2. Setup Your Python Environment

in Windows
```bash
git clone https://github.com/KishaniKandasamy/Health-Insurance-Lead-Prediction.git
cd Health-Insurance-Lead-Prediction
python3 -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 3. Configure the App
You need to set your WAVE SDK installation path in ```config.py```

### 4. Run the App

```bash
 wave run app
```

Note! You need activate your virtual environment this will be:

```bash
venv\Scripts\activate.bat
```

### 5. View the App

Point your favorite web browser to [localhost:10101/decisiontree](http://localhost:10101/decisiontree)

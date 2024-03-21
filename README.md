# Predictive_maintenance

**Web address of dataset:**
https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

**Directory details:**
1. data: contains raw data in csv format
2. Config: contains yaml for configuration
3. Notebook: contains notebooks for processing, analysis, and learning
4. Src: contains differents script py (details in next chapter)

**Notebook details:**
1. '#1_Load_Analyse_data' : Load and analyse original dataset
2. '#2_Preprocess_data' : Transform, prepare data for training notebook
3. '#3a_Modeling_DNN' : Training notebook, save model in 'Model' directory for deep learning network
4. '#3b_regressionLog' : Training notebook, save model in 'Model' directory for logistic regressionmethod
4. '#4_Model_explication' : Notebook to explain the model global and local

**Src details:**
1. 'Config': contains script py for mange configuration file
2. 'utils': contains script for differents modules
3. 'APP' : contains script utils to API

**How to use API**


An image is generated with githubactions.
This image is available with command :
    docker pull yannickbodin/pred_maint_projects
The library used to implement the API is FastAPI.
The different entry points available are :
- '/' : just a hello
- '/list_features/' : returns a list of input variables 
- '/features/{variable}': returns *variable* details (type, min, max,...)
- '/predict': returns probabilities for each class
- '/explain' : returns an explanation of the decision using the LIME library

An online version is available on Render :
https://test-pred-maint-projects.onrender.com/docs

Warning: one minute latency for first startup

Translated with www.DeepL.com/Translator (free version)

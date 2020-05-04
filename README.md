# Disaster Response Pipeline Project

### Requirements :
The code has been written on python 3.6.3.
The following python libraries are needed to run the code.
* Numpy
* Pandas
* sqlalchemy
* joblib
* Scikit-learn
* nltk
* flask
* re

### File Descriptions :

#### app 
It contains the python script run.py containing the flask code used to run the app.

#### data
It contains two dataset files disaster_categories.csv and disaster_messages.csv.  
One database file DisasterResponse.db containing the final dataframe in a table named data.  
A python script for loading,cleaning and saving data into database process_data.py

#### models 
It contains the python script train_classifier.py used to train and save the ML model in the same folder.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/(Your_Database_Name).db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/(Your_Database_Name).db models/(Your_Model_Name).pkl`   
      
2. Update the database name and the trained model name in appropriate locations of the run.py file.

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://127.0.0.1:3001/ 

### Results:

You can see the results and some visualizations of the data using the above link.
You can even enter a custom message and have it categorised with more than 92% accuracy.

### Note :
The saved model .pkl file has not been uploaded due to size restrictions.

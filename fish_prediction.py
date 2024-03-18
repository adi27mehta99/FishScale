"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Fish.csv')

# Save a copy of the original species data
species = df['Species'].copy()

# The error is caused by trying to convert categorical data to numerical data
# We can solve this by encoding the categorical data
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Prepare the data
X = df.drop('Weight', axis=1)
y = df['Weight']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Ensure all predictions are positive
predictions = [max(0, pred) for pred in predictions]

# Create a DataFrame to hold the species and predicted weights
results = pd.DataFrame({
    'Species': species[X_test.index],
    'Predicted Weight (in grams)': predictions
})

# Print the species and predicted weights
print(results)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Fish.csv')

# Save a copy of the original species data
species = df['Species'].copy()

# The error is caused by trying to convert categorical data to numerical data
# We can solve this by encoding the categorical data
#le = LabelEncoder()
#df['Species'] = le.fit_transform(df['Species'])

# Prepare the data
X = df.drop('Weight', axis=1)
y = df['Weight']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict(data):
    # Convert data to appropriate format
    # This will depend on how your form data is structured
    # For example, if data is a dictionary of feature values:
    input_data = pd.DataFrame(data, index=[0])
    input_data['Species'] = le.transform(input_data['Species'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Ensure prediction is positive
    prediction = max(0, prediction[0])
    
    return prediction
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
# Load the dataset
df = pd.read_csv('Fish.csv')

# Save a copy of the original species data
species = df['Species'].copy()

# The error is caused by trying to convert categorical data to numerical data
# We can solve this by encoding the categorical data
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Prepare the data
X = df.drop('Weight', axis=1)
y = df['Weight']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



def predict(data):
    # Convert data to appropriate format
    # This will depend on how your form data is structured
    # For example, if data is a dictionary of feature values:
    input_data = pd.DataFrame(data, index=[0])
    # Check if 'Species' is in the input data
    if 'Species' not in input_data:
        return "Error: No species data provided. Please include 'Species' in the input data."
    input_data['Species'] = le.transform(input_data['Species'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Ensure prediction is positive
    prediction = max(0, prediction[0])
    
    return prediction



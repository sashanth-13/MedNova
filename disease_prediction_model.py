import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re
import warnings
warnings.filterwarnings("ignore")

print("Loading and preprocessing datasets...")

# Load all the datasets
data = pd.read_csv('data/dataset.csv')
desc = pd.read_csv('data/symptom_Description.csv')
precaution = pd.read_csv('data/symptom_precaution.csv')
severity = pd.read_csv('data/Symptom-severity.csv')

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nDisease count:")
print(data['Disease'].value_counts().head(10))
print(f"Total number of unique diseases: {data['Disease'].nunique()}")

# Data Preprocessing
# Replace NaN values with 0
data = data.fillna(0)
data.head()

# Symptom standardization function
def standardize_symptom(symptom):
    """Standardize symptom text to ensure consistent format"""
    if not isinstance(symptom, str) or symptom == '0':
        return '0'
    
    # Trim whitespace and lowercase
    symptom = symptom.strip().lower()
    
    # Remove punctuation
    symptom = re.sub(r'[^\w\s]', '', symptom)
    
    # Replace spaces with underscores
    symptom = re.sub(r'\s+', '_', symptom)
    
    return symptom

# Standardize all symptoms in the dataset
for col in data.columns[1:]:  # Skip the 'Disease' column
    data[col] = data[col].apply(standardize_symptom)

# Create a list of all symptoms from the dataset
# Extract the symptoms columns (all columns except the first one which is 'Disease')
columns = data.columns[1:]
symptoms = []
for col in columns:
    symptoms.extend(data[col].unique())
    
# Remove '0' and duplicates
symptoms = list(set([s for s in symptoms if s != '0']))
print(f"Total number of unique symptoms: {len(symptoms)}")

# Create an improved feature matrix with binary encoding for symptoms
def create_feature_matrix(df, symptoms):
    """
    Creates a binary feature matrix where each row corresponds to a sample
    and each column corresponds to a symptom.
    """
    feature_matrix = np.zeros((len(df), len(symptoms)))
    
    for i, row in df.iterrows():
        # Get all symptoms for this sample (excluding Disease)
        indices = []
        for symptom in row[1:]:
            if symptom != '0' and symptom in symptoms:
                try:
                    indices.append(symptoms.index(symptom))
                except ValueError:
                    # Handle case where symptom might not be in our list
                    continue
        
        feature_matrix[i, indices] = 1
        
    return feature_matrix

# Create a disease-symptom mapping for better matching
disease_to_symptoms = {}
for _, row in data.iterrows():
    disease = row['Disease']
    if disease not in disease_to_symptoms:
        disease_to_symptoms[disease] = set()
    
    for col in data.columns[1:]:
        symptom = row[col]
        if symptom != '0' and isinstance(symptom, str):
            disease_to_symptoms[disease].add(symptom)

# Create feature matrix and labels
print("Creating feature matrix...")
X = create_feature_matrix(data, symptoms)
y = data['Disease']

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
disease_classes = le.classes_

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Try to improve model with a grid search for best parameters
print("\nFinding optimal model parameters...")
param_grid = {
    'n_estimators': [200, 300],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [4, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', 'balanced_subsample']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
rf_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# View classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=disease_classes)
print(report)

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(rf_model, X, y_encoded, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores):.4f}")

# Incorporate symptom severity into the model
# Create a symptom severity dictionary
# First standardize the symptoms in severity dataframe
severity['Symptom'] = severity['Symptom'].apply(standardize_symptom)
symptom_severity_dict = dict(zip(severity['Symptom'], severity['weight']))

# Function to calculate severity score of symptoms
def calculate_severity(symptoms_list):
    severity_score = 0
    for symptom in symptoms_list:
        if symptom in symptom_severity_dict:
            severity_score += symptom_severity_dict[symptom]
        else:
            # Try to find closest match for the symptom
            for known_symptom in symptom_severity_dict:
                if symptom in known_symptom or known_symptom in symptom:
                    severity_score += symptom_severity_dict[known_symptom]
                    break
    return severity_score

# Save the model and related data
model_data = {
    'model': rf_model,
    'label_encoder': le,
    'symptoms_list': symptoms,
    'symptom_severity_dict': symptom_severity_dict,
    'disease_descriptions': dict(zip(desc['Disease'], desc['Description'])),
    'disease_precautions': {disease: list(row[1:]) for disease, row in precaution.iterrows()},
    'disease_classes': disease_classes.tolist(),
    'disease_to_symptoms': disease_to_symptoms  # Add symptom mapping for better matching
}

with open('disease_prediction_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModel and related data saved to 'disease_prediction_model.pkl'")

# Enhanced function to predict multiple diseases based on symptoms
def predict_diseases(symptoms_input, top_n=3, confidence_threshold=5.0):
    """
    Predicts top N diseases based on a list of symptoms.
    Returns predicted diseases, confidence scores, descriptions, and precautions.
    Only includes diseases with confidence above the threshold.
    
    Args:
        symptoms_input: List of symptom names
        top_n: Number of top predictions to return
        confidence_threshold: Minimum confidence (percentage) to include a prediction
        
    Returns:
        List of dictionaries, each containing disease, confidence, severity, description, and precautions
    """
    # Load the model data
    with open('disease_prediction_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    rf_model = model_data['model']
    le = model_data['label_encoder']
    symptoms_list = model_data['symptoms_list']
    severity_dict = model_data['symptom_severity_dict']
    disease_descriptions = model_data['disease_descriptions']
    disease_precautions = model_data['disease_precautions']
    disease_classes = model_data['disease_classes']
    disease_to_symptoms = model_data.get('disease_to_symptoms', {})
    
    # Standardize the input symptoms
    standardized_input = [standardize_symptom(s) for s in symptoms_input]
    
    # Create feature vector for the input symptoms
    input_vector = np.zeros(len(symptoms_list))
    valid_symptoms = []
    
    for symptom in standardized_input:
        # Exact match
        if symptom in symptoms_list:
            input_vector[symptoms_list.index(symptom)] = 1
            valid_symptoms.append(symptom)
            continue
            
        # Try to find closest match
        matched = False
        for i, known_symptom in enumerate(symptoms_list):
            if symptom in known_symptom or known_symptom in symptom:
                input_vector[i] = 1
                valid_symptoms.append(known_symptom)
                matched = True
                break
        
        if not matched:
            print(f"Warning: Could not find match for symptom '{symptom}'")
    
    # Handle case where no valid symptoms were provided
    if not valid_symptoms:
        return [{
            'disease': 'Unknown',
            'confidence': 0,
            'severity_score': 0,
            'description': "No valid symptoms were provided. Please check your symptoms and try again.",
            'precautions': ["Consult with a healthcare professional"],
            'matching_symptoms': []
        }]
    
    # Get prediction probabilities
    probabilities = rf_model.predict_proba([input_vector])[0]
    
    # Get indices of top N predictions
    top_indices = probabilities.argsort()[::-1][:top_n]
    
    predictions = []
    for idx in top_indices:
        confidence = probabilities[idx] * 100
        if confidence >= confidence_threshold:
            disease = disease_classes[idx]
            
            # Find what symptoms typically match this disease using the disease-symptom mapping
            typical_symptoms = disease_to_symptoms.get(disease, set())
            
            # Find matching symptoms
            matching_symptoms = [s for s in valid_symptoms if s in typical_symptoms]
            
            # For better user experience, also check if any input symptom is similar to disease symptoms
            similar_matches = []
            for vs in valid_symptoms:
                if vs not in matching_symptoms:
                    for ts in typical_symptoms:
                        if vs in ts or ts in vs:
                            similar_matches.append(vs)
                            break
            
            # Combine exact and similar matches
            all_matching = list(set(matching_symptoms + similar_matches))
            
            predictions.append({
                'disease': disease,
                'confidence': round(confidence, 2),
                'severity_score': calculate_severity(valid_symptoms),
                'description': disease_descriptions.get(disease, "No description available"),
                'precautions': [p for p in disease_precautions.get(disease, ["No specific precautions available"]) 
                               if p != '0' and p is not None],
                'matching_symptoms': all_matching
            })
    
    # If no predictions meet the threshold, include the top one regardless
    if not predictions:
        top_idx = top_indices[0]
        disease = disease_classes[top_idx]
        
        # Get typical symptoms for this disease
        typical_symptoms = disease_to_symptoms.get(disease, set())
        matching_symptoms = [s for s in valid_symptoms if s in typical_symptoms]
        
        predictions.append({
            'disease': disease,
            'confidence': round(probabilities[top_idx] * 100, 2),
            'severity_score': calculate_severity(valid_symptoms),
            'description': disease_descriptions.get(disease, "No description available"),
            'precautions': [p for p in disease_precautions.get(disease, ["No specific precautions available"]) 
                           if p != '0' and p is not None],
            'matching_symptoms': matching_symptoms
        })
    
    return predictions

# For backward compatibility
def predict_disease(symptoms_input):
    """
    Legacy function that returns only the top prediction.
    """
    predictions = predict_diseases(symptoms_input, top_n=1)
    return predictions[0]

# Display the most important symptoms for prediction according to our model
feature_importance = pd.DataFrame({
    'Symptom': symptoms,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("\nTop 15 most important symptoms for prediction:")
print(feature_importance.head(15))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Symptom', data=feature_importance.head(15))
plt.title('Top 15 Most Important Symptoms for Disease Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")
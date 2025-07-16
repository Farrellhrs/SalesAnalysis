# PREDICTIVE MODEL MODULE
"""
Predictive model for deal probability estimation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class PredictiveModel:
    """Deal probability prediction model"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.feature_importance_df = None
        self.model_metrics = {}
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        df_ml = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['Progress', 'Segmen', 'Status_Customer', 'Level_Sales']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_ml[f'{col}_Encoded'] = self.label_encoders[col].fit_transform(df_ml[col])
            else:
                # For prediction, use existing encoders
                try:
                    df_ml[f'{col}_Encoded'] = self.label_encoders[col].transform(df_ml[col])
                except ValueError:
                    # Handle unseen categories
                    df_ml[f'{col}_Encoded'] = 0
        
        # Create binary target variable
        df_ml['Deal_Binary'] = (df_ml['Status_Kontrak'] == 'Deal').astype(int)
        
        # Feature engineering
        df_ml['Target_Sales_Scaled'] = df_ml['Target_Sales'] / 1e9
        df_ml['Target_Segmen_Scaled'] = df_ml['Target_Segmen'] / 1e9
        
        # Interaction features
        df_ml['Progress_Visit_Interaction'] = df_ml['Progress_Encoded'] * df_ml['Kunjungan_Ke']
        df_ml['Segmen_Visit_Interaction'] = df_ml['Segmen_Encoded'] * df_ml['Kunjungan_Ke']
        
        # Define feature columns
        self.feature_columns = [
            'Progress_Encoded', 'Kunjungan_Ke', 'Segmen_Encoded',
            'Status_Customer_Encoded', 'Level_Sales_Encoded',
            'Target_Sales_Scaled', 'Target_Segmen_Scaled',
            'Progress_Visit_Interaction', 'Segmen_Visit_Interaction'
        ]
        
        return df_ml
    
    def train(self, df):
        """Train the predictive model"""
        # Prepare features
        df_ml = self.prepare_features(df)
        
        # Check if we have enough data
        if len(df_ml) < 3:
            print(f"Warning: Insufficient data for training ({len(df_ml)} samples). Need at least 3 samples.")
            # Create a dummy model that predicts based on simple rules
            self.model = None
            self.model_metrics = {
                'model_name': 'Rule-based (insufficient data)',
                'train_accuracy': 0.0,
                'test_accuracy': 0.0,
                'cv_mean': 0.0,
                'cv_std': 0.0
            }
            return
        
        # Select features and target
        X = df_ml[self.feature_columns]
        y = df_ml['Deal_Binary']
        
        # Check if we have both classes
        unique_classes = y.unique()
        if len(unique_classes) < 2:
            print(f"Warning: Only one class present in data ({unique_classes}). Cannot train classifier.")
            # Create a simple rule-based model
            self.model = None
            self.model_metrics = {
                'model_name': 'Rule-based (single class)',
                'train_accuracy': 1.0 if unique_classes[0] == 1 else 0.0,
                'test_accuracy': 1.0 if unique_classes[0] == 1 else 0.0,
                'cv_mean': 1.0 if unique_classes[0] == 1 else 0.0,
                'cv_std': 0.0
            }
            return
        
        # Train-test split with fallback for small datasets
        test_size = max(0.1, min(0.3, 1/len(df_ml)))  # Adaptive test size
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails (too few samples), split without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Try different models and select the best one
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_model = None
        best_score = 0
        model_results = {}
        
        # Determine appropriate CV folds based on sample size
        n_samples = len(X_train)
        cv_folds = min(5, max(2, n_samples // 2))  # Use 2-5 folds, at least 2 samples per fold
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Cross-validation with adaptive folds
            if cv_folds >= 2:
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except ValueError:
                    # Fallback: use test accuracy if CV fails
                    cv_mean = test_accuracy
                    cv_std = 0.0
            else:
                # Not enough samples for CV, use test accuracy
                cv_mean = test_accuracy
                cv_std = 0.0
            
            model_results[name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
            
            # Select best model based on cross-validation score
            if cv_mean > best_score:
                best_score = cv_mean
                best_model = model
                self.model_metrics = model_results[name].copy()
                self.model_metrics['model_name'] = name
        
        # Set the best model
        self.model = best_model
        
        # Calculate feature importance
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            self.feature_importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # Create dummy feature importance for rule-based models
            self.feature_importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': [1.0/len(self.feature_columns)] * len(self.feature_columns)
            })
        
        self.is_trained = True
        
        return model_results
    
    def predict_probability(self, progress, visit_number, segment='Private',
                          customer_status='Baru', level_sales='AM',
                          target_sales=5e9, target_segmen=50e9):
        """Predict deal probability for given parameters"""
        if not self.is_trained or self.model is None:
            # Use simple rule-based prediction if no model available
            if progress in ['Paska Deal']:
                return 1.0
            elif progress in ['Inisiasi']:
                return 0.2
            elif progress in ['Presentasi']:
                return 0.4
            elif progress in ['Penawaran Harga']:
                return 0.6
            elif progress in ['Negosiasi']:
                return 0.8
            else:
                return 0.5
        
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Progress': [progress],
                'Kunjungan_Ke': [visit_number],
                'Segmen': [segment],
                'Status_Customer': [customer_status],
                'Level_Sales': [level_sales],
                'Target_Sales': [target_sales],
                'Target_Segmen': [target_segmen],
                'Status_Kontrak': ['Tidak Deal']  # Dummy value
            })
            
            # Prepare features
            input_ml = self.prepare_features(input_data)
            
            # Select features
            X_input = input_ml[self.feature_columns]
            
            # Predict probability
            probability = self.model.predict_proba(X_input)[0, 1]
            
            return probability
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance dataframe"""
        return self.feature_importance_df
    
    def get_model_metrics(self):
        """Get model performance metrics"""
        return self.model_metrics
    
    def batch_predict(self, df):
        """Predict probabilities for entire dataframe"""
        if not self.is_trained or self.model is None:
            # Use rule-based predictions
            def rule_based_probability(row):
                if row['Progress'] == 'Paska Deal':
                    return 1.0
                elif row['Progress'] == 'Inisiasi':
                    return 0.2
                elif row['Progress'] == 'Presentasi':
                    return 0.4
                elif row['Progress'] == 'Penawaran Harga':
                    return 0.6
                elif row['Progress'] == 'Negosiasi':
                    return 0.8
                else:
                    return 0.5
            
            return df.apply(rule_based_probability, axis=1).values
        
        try:
            # Prepare features
            df_ml = self.prepare_features(df)
            
            # Select features
            X = df_ml[self.feature_columns]
            
            # Predict probabilities
            probabilities = self.model.predict_proba(X)[:, 1]
            
            return probabilities
            
        except Exception as e:
            print(f"Batch prediction error: {e}")
            return None
    
    def get_prediction_insights(self, df):
        """Get insights from predictions"""
        if not self.is_trained:
            return {}
        
        # Get predictions for entire dataset
        probabilities = self.batch_predict(df)
        
        if probabilities is None:
            return {}
        
        # Add probabilities to dataframe
        df_with_pred = df.copy()
        df_with_pred['Predicted_Probability'] = probabilities
        
        # Calculate insights
        insights = {
            'avg_probability': probabilities.mean(),
            'high_probability_count': (probabilities > 0.7).sum(),
            'medium_probability_count': ((probabilities > 0.3) & (probabilities <= 0.7)).sum(),
            'low_probability_count': (probabilities <= 0.3).sum(),
            'best_segment_probability': df_with_pred.groupby('Segmen')['Predicted_Probability'].mean().idxmax(),
            'best_progress_probability': df_with_pred.groupby('Progress')['Predicted_Probability'].mean().idxmax(),
            'optimal_visit_number': df_with_pred.groupby('Kunjungan_Ke')['Predicted_Probability'].mean().idxmax()
        }
        
        return insights

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField, TextAreaField, DateField
from wtforms.validators import DataRequired, Email, ValidationError, Length
from flask_wtf.file import FileField, FileRequired, FileAllowed
from datetime import datetime

class DiagnosisFormValidator:
    """
    Validator class for diagnosis forms with disease-specific validation rules.
    """
    
    @classmethod
    def validate_symptoms_for_disease(cls, symptoms_list, disease_name):
        """
        Validate symptoms based on the disease type.
        
        Args:
            symptoms_list: List of symptoms
            disease_name: Name of the disease
            
        Returns:
            tuple: (is_valid, error_message or None)
        """
        if not symptoms_list:
            return False, "At least one symptom is required"
        
        # Remove empty symptoms
        valid_symptoms = [s.strip() for s in symptoms_list if s.strip()]
        
        if len(valid_symptoms) == 0:
            return False, "At least one valid symptom is required"
        
        # Disease-specific validations
        if disease_name == "Breast Cancer":
            return cls._validate_breast_cancer_symptoms(symptoms_list)
        elif disease_name == "Lung Cancer":
            return cls._validate_lung_cancer_symptoms(symptoms_list)
        
        # Default validation for unknown diseases
        return True, None
    
    @classmethod
    def _validate_breast_cancer_symptoms(cls, symptoms_list):
        """Validate breast cancer specific symptoms."""
        common_breast_symptoms = [
            'lump', 'pain', 'discharge', 'swelling', 'redness', 'dimpling',
            'breast pain', 'nipple discharge', 'breast lump', 'skin changes',
            'breast swelling', 'breast tenderness', 'nipple pain', 'breast mass'
        ]
        
        symptoms_text = ' '.join(symptoms_list).lower()
        
        # Check if at least one common breast cancer symptom is mentioned
        has_relevant_symptom = any(symptom.lower() in symptoms_text for symptom in common_breast_symptoms)
        
        if not has_relevant_symptom:
            return False, "Please include symptoms relevant to breast cancer (e.g., breast lump, pain, discharge)"
        
        return True, None
    
    @classmethod
    def _validate_lung_cancer_symptoms(cls, symptoms_list):
        """Validate lung cancer specific symptoms."""
        common_lung_symptoms = [
            'cough', 'shortness of breath', 'chest pain', 'blood in sputum', 'fatigue',
            'persistent cough', 'breathing difficulty', 'wheezing', 'hoarseness',
            'weight loss', 'chest tightness', 'hemoptysis', 'dyspnea'
        ]
        
        symptoms_text = ' '.join(symptoms_list).lower()
        
        # Check if at least one common lung cancer symptom is mentioned
        has_relevant_symptom = any(symptom.lower() in symptoms_text for symptom in common_lung_symptoms)
        
        if not has_relevant_symptom:
            return False, "Please include symptoms relevant to lung cancer (e.g., persistent cough, chest pain, shortness of breath)"
        
        return True, None
    
    @classmethod
    def get_disease_specific_hints(cls, disease_name):
        """
        Get disease-specific hints for form filling.
        
        Args:
            disease_name: Name of the disease
            
        Returns:
            dict: Dictionary containing hints for different fields
        """
        if disease_name == "Breast Cancer":
            return {
                'symptoms_hint': 'Common symptoms: breast lump, breast pain, nipple discharge, skin changes, swelling',
                'clinical_history_hint': 'Include family history, previous breast examinations, hormonal factors, age at menarche',
                'image_hint': 'Upload mammography or breast ultrasound images'
            }
        elif disease_name == "Lung Cancer":
            return {
                'symptoms_hint': 'Common symptoms: persistent cough, shortness of breath, chest pain, blood in sputum, fatigue',
                'clinical_history_hint': 'Include smoking history, occupational exposure, family history, previous lung conditions',
                'image_hint': 'Upload chest X-ray or CT scan images'
            }
        else:
            return {
                'symptoms_hint': 'Please describe your symptoms in detail',
                'clinical_history_hint': 'Include relevant medical history and family history',
                'image_hint': 'Upload relevant medical images if available'
            }

class EnhancedDiagnosisForm(FlaskForm):
    # Patient Information
    patient_name = StringField('Patient Name', validators=[DataRequired(), Length(min=2, max=100)])
    patient_age = StringField('Patient Age', validators=[DataRequired()])
    patient_email = StringField('Email', validators=[DataRequired(), Email()])
    
    # Disease Selection
    disease_type = SelectField('Disease Type', 
                              choices=[
                                  ('', 'Select Disease Type'),
                                  ('Breast Cancer', 'Breast Cancer'),
                                  ('Lung Cancer', 'Lung Cancer')
                              ],
                              validators=[DataRequired()],
                              default='')
    
    # Medical Information
    symptoms = TextAreaField('Symptoms', 
                           validators=[DataRequired(), Length(min=10, max=1000)],
                           render_kw={"placeholder": "Describe symptoms in detail..."})
    
    clinical_history = TextAreaField('Clinical History',
                                   validators=[DataRequired(), Length(min=10, max=2000)],
                                   render_kw={"placeholder": "Include medical history, family history, medications..."})
    
    # Image Upload
    medical_image = FileField('Medical Image', 
                            validators=[FileRequired(), 
                                      FileAllowed(['jpg', 'jpeg', 'png'], 'Only JPG, JPEG, and PNG files are allowed!')])
    
    submit = SubmitField('Submit for Diagnosis')
    
    def validate_symptoms(self, field):
        """Custom validator for symptoms field with disease-specific validation."""
        if self.disease_type.data:
            symptoms_list = [s.strip() for s in field.data.split(',') if s.strip()]
            is_valid, error_message = DiagnosisFormValidator.validate_symptoms_for_disease(
                symptoms_list, self.disease_type.data
            )
            if not is_valid:
                raise ValidationError(error_message)
    
    def validate_patient_age(self, field):
        """Custom validator for patient age."""
        try:
            age = int(field.data)
            if age < 0 or age > 150:
                raise ValidationError('Please enter a valid age between 0 and 150.')
        except ValueError:
            raise ValidationError('Please enter a valid numeric age.')
    
    def validate_disease_type(self, field):
        """Custom validator for disease type selection."""
        if not field.data or field.data == '':
            raise ValidationError('Please select a disease type.')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Sign In')
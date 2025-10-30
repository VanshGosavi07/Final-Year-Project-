import requests
import json
import os
import cv2
import ast
import numpy as np
import logging
from typing import List, Tuple
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_cors import CORS
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField, TextAreaField, FileField, DateField
from wtforms.validators import DataRequired, Email, ValidationError, Length
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from functools import lru_cache
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Load environment variables
load_dotenv()

# Import our custom predictors and validators
from breast_cancer_predictor import BreastCancerPredictor
from lung_cancer_predictor import LungCancerPredictor
from form_validators import EnhancedDiagnosisForm, DiagnosisFormValidator

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Configuration from environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS', 'False').lower() == 'true'
app.config['PDF_FOLDER'] = os.getenv('PDF_FOLDER', 'RAG Data')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))  # 16MB default
app.config['WTF_CSRF_ENABLED'] = os.getenv('WTF_CSRF_ENABLED', 'True').lower() == 'true'

# Cloudinary Configuration
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

# API Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = os.getenv('GROQ_API_URL', 'https://api.groq.com/openai/v1/chat/completions')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')

# Model paths
BREAST_CANCER_MODEL_PATH = os.getenv('BREAST_CANCER_MODEL_PATH', 'Modal/Breast Cancer/breast_cancer.keras')
LUNG_CANCER_MODEL_PATH = os.getenv('LUNG_CANCER_MODEL_PATH', 'Modal/Lung Cancer/Lung Cancer.keras')

# RAG Configuration
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))

# TensorFlow configuration
tf_log_level = os.getenv('TENSORFLOW_CPP_MIN_LOG_LEVEL', '2')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level

# Validate critical environment variables
required_env_vars = {
    'SECRET_KEY': app.config['SECRET_KEY'],
    'SQLALCHEMY_DATABASE_URI': app.config['SQLALCHEMY_DATABASE_URI'],
    'CLOUDINARY_CLOUD_NAME': os.getenv('CLOUDINARY_CLOUD_NAME'),
    'CLOUDINARY_API_KEY': os.getenv('CLOUDINARY_API_KEY'),
    'CLOUDINARY_API_SECRET': os.getenv('CLOUDINARY_API_SECRET'),
    'GROQ_API_KEY': GROQ_API_KEY
}

missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    error_msg = f"CRITICAL: Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    logger.error("Please set all required variables in your .env file")
    raise EnvironmentError(error_msg)

db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)  # Increased to 255 for bcrypt hashes
    contact_number = db.Column(db.String(15), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    city = db.Column(db.String(50), nullable=False)
    user_type = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)

# Forms
class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8)])
    contact_number = StringField("Contact Number", validators=[DataRequired(), Length(min=10, max=15)])
    date_of_birth = DateField("Date of Birth", validators=[DataRequired()])
    city = StringField("City", validators=[DataRequired(), Length(min=2, max=50)])
    user_type = SelectField("User Type", choices=[('doctor', 'Doctor'), ('patient', 'Patient')], validators=[DataRequired()])
    gender = SelectField("Gender", choices=[('male', 'Male'), ('female', 'Female')], validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self, field):
        if User.query.filter_by(email=field.data).first():
            raise ValidationError('Email already taken')

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

class DiagnosisForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=50)])
    dob = DateField('Date of Birth', validators=[DataRequired()])
    disease_name = SelectField('Disease Name', choices=[('Breast Cancer', 'Breast Cancer'), ('Lung Cancer', 'Lung Cancer')], validators=[DataRequired()])
    ct_images = FileField('Upload Medical Images', validators=[FileRequired(), FileAllowed(['png', 'jpg', 'jpeg'], 'Images only!')])
    clinical_history = TextAreaField('Clinical History', validators=[DataRequired(), Length(min=10)])
    symptoms = StringField('Symptoms', validators=[DataRequired(), Length(min=5)])
    submit = SubmitField('Generate Report')

    def validate_name(self, field):
        if not field.data.replace(' ', '').isalpha():
            raise ValidationError('Name must contain only letters')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_FOLDER'], exist_ok=True)

# Initialize predictors
breast_cancer_predictor = None
lung_cancer_predictor = None

def get_breast_cancer_predictor():
    global breast_cancer_predictor
    if breast_cancer_predictor is None:
        breast_cancer_predictor = BreastCancerPredictor(model_path=BREAST_CANCER_MODEL_PATH)
    return breast_cancer_predictor

def get_lung_cancer_predictor():
    global lung_cancer_predictor
    if lung_cancer_predictor is None:
        lung_cancer_predictor = LungCancerPredictor(model_path=LUNG_CANCER_MODEL_PATH)
    return lung_cancer_predictor

# RAG Document Processor
PERSONAL_REPORTS = []

# Pre-loaded processors for fast access
breast_cancer_doc_processor = None
lung_cancer_doc_processor = None
general_doc_processor = None

def get_general_doc_processor():
    """Lazy load general document processor"""
    global general_doc_processor
    if general_doc_processor is None:
        logger.info("Initializing general document processor...")
        general_doc_processor = DocumentProcessor()
        logger.info("General document processor initialized ✓")
    return general_doc_processor

def get_breast_cancer_doc_processor():
    """Lazy load breast cancer document processor"""
    global breast_cancer_doc_processor
    if breast_cancer_doc_processor is None:
        logger.info("Initializing breast cancer document processor...")
        breast_cancer_doc_processor = DocumentProcessor(disease_type='breast_cancer', preload_data=True)
        logger.info("Breast cancer document processor initialized ✓")
    return breast_cancer_doc_processor

def get_lung_cancer_doc_processor():
    """Lazy load lung cancer document processor"""
    global lung_cancer_doc_processor
    if lung_cancer_doc_processor is None:
        logger.info("Initializing lung cancer document processor...")
        lung_cancer_doc_processor = DocumentProcessor(disease_type='lung_cancer', preload_data=True)
        logger.info("Lung cancer document processor initialized ✓")
    return lung_cancer_doc_processor

class DocumentProcessor:
    def __init__(self, disease_type=None, preload_data=False):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = None
        self.disease_type = disease_type
        
        if preload_data and disease_type:
            # Pre-load disease-specific data
            texts = self.extract_text_from_pdfs(app.config['PDF_FOLDER'], disease_type)
            self.update_vector_store(texts)
            logger.info(f"Pre-loaded {disease_type} processor with {len(texts)} text chunks")
        else:
            self.update_vector_store([])

    def extract_text_from_pdfs(self, folder_path: str, specific_disease: str = None) -> List[str]:
        try:
            extracted_texts = []
            pdf_files = []
            
            # If specific disease is mentioned, look for disease-specific PDFs
            if specific_disease:
                disease_mapping = {
                    'breast_cancer': ['Breast_Cancer_Rag_Data.pdf', 'breast_cancer.pdf', 'breast-cancer.pdf'],
                    'lung_cancer': ['Lung_Cancer_Rag_Data.pdf', 'lung_cancer.pdf', 'lung-cancer.pdf']
                }
                
                if specific_disease.lower() in disease_mapping:
                    for filename in disease_mapping[specific_disease.lower()]:
                        if os.path.exists(os.path.join(folder_path, filename)):
                            pdf_files.append(filename)
                            break  # Use the first found file
            
            # If no specific files found or no disease specified, return empty for general processor
            if not pdf_files and not specific_disease:
                logger.info("General processor initialized without loading PDFs (for performance)")
                return []
            
            # If no specific files found, use all PDFs (fallback)
            if not pdf_files:
                pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
            
            for pdf_file in pdf_files:
                file_path = os.path.join(folder_path, pdf_file)
                try:
                    with fitz.open(file_path) as doc:
                        text = "".join(page.get_text("text") + "\n" for page in doc)
                        extracted_texts.append(text)
                        logger.info(f"Loaded RAG data from: {pdf_file}")
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {str(e)}")
            return extracted_texts
        except Exception as e:
            logger.error(f"Error in PDF extraction: {str(e)}")
            return []

    def update_vector_store(self, current_report: List[str], specific_disease: str = None):
        try:
            pdf_texts = self.extract_text_from_pdfs(app.config['PDF_FOLDER'], specific_disease)
            all_documents = current_report + pdf_texts
            if all_documents:
                docs = self.text_splitter.create_documents(all_documents)
                self.vector_store = FAISS.from_documents(docs, self.embedding_model)
            else:
                self.vector_store = None
            logger.info("Vector store updated successfully with current report")
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            raise


    def retrieve_context(self, query: str, k: int = 3) -> str:
        if not self.vector_store:
            return "No context available yet. Please generate a report first."
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return "\n".join(doc.page_content for doc in results)
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""

# Groq Client
class GroqClient:
    @staticmethod
    @lru_cache(maxsize=50)
    def call_groq_api(prompt: str, model: str = None, max_tokens: int = 2000, temperature: float = 0.5) -> str:
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not configured")
            return "API key not configured. Please set GROQ_API_KEY in your .env file."
        
        if model is None:
            model = GROQ_MODEL
            
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens}
        try:
            response = requests.post(GROQ_API_URL, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return "Sorry, I couldn't process that request."

# Global Document Processors
doc_processor = None
breast_cancer_doc_processor = None
lung_cancer_doc_processor = None

@lru_cache(maxsize=500)  # Increased cache for better performance
def model_predict(image_paths: tuple, disease_name: str) -> List[Tuple[str, str]]:
    """
    Predict using the appropriate model based on disease type.
    
    Args:
        image_paths: Tuple of image paths
        disease_name: Name of the disease
        
    Returns:
        List of tuples containing (label, output_image_path)
    """
    results = []
    
    # Get the appropriate predictor
    if disease_name == "Breast Cancer":
        predictor = get_breast_cancer_predictor()
    elif disease_name == "Lung Cancer":
        predictor = get_lung_cancer_predictor()
    else:
        logger.error(f"Unknown disease type: {disease_name}")
        return results
    
    for img_path in image_paths:
        img_full_path = os.path.join('static', img_path)
        if not os.path.exists(img_full_path):
            logger.error(f"File not found: {img_full_path}")
            continue
            
        # Create output path
        new_img_path = img_path.rsplit('.', 1)[0] + '_output.png'
        new_img_full_path = os.path.join('static', new_img_path)
        
        # Get prediction with visualization for diseases with models
        label, confidence, output_path = predictor.predict_with_visualization(img_full_path, new_img_full_path)
        
        if output_path:
            results.append((label, new_img_path))
        else:
            logger.error(f"Failed to create visualization for {img_path}")
            results.append((label, img_path))  # Use original image if visualization fails
    
    return results

@lru_cache(maxsize=200)  # Increased cache for faster report generation
def generate_data(name: str, age: int, disease_name: str, clinical_history: str, 
                 symptoms: tuple, image_paths: tuple, diseases_level: tuple) -> dict:
    if not image_paths:
        logger.warning("No image paths provided")
        return {}
    img_url = url_for('static', filename=image_paths[0], _external=True)
    
    # Disease-specific prompt customization
    disease_specific_info = ""
    if disease_name == "Breast Cancer":
        disease_specific_info = "This is a breast cancer analysis based on mammography/breast imaging. "
    elif disease_name == "Lung Cancer":
        disease_specific_info = "This is a lung cancer analysis based on chest X-ray/CT scan imaging. "
    
    # Determine imaging type
    imaging_type = "Mammography" if disease_name == "Breast Cancer" else "Chest X-Ray/CT Scan"
    
    prompt = (
        f"{disease_specific_info}"
        f"Patient: {name}, Age: {age}, Disease: {disease_name}\n"
        f"Clinical History: {clinical_history}\n"
        f"Symptoms: {list(symptoms)}\n"
        f"AI Prediction: {list(diseases_level)}\n\n"
        "Generate a complete medical report in STRICT Python dictionary format. "
        "IMPORTANT: For imaging studies, each item must be in 'Type: Finding' format.\n\n"
        "Required format (return ONLY this dictionary, no extra text):\n"
        "{\n"
        f"    'detailed findings': 'Detailed analysis of {disease_name} based on prediction: {list(diseases_level)}',\n"
        "    'clinical examination': 'Clinical assessment based on symptoms and history',\n"
        f"    'imaging studies': ['{imaging_type}: Specific radiological findings', 'Assessment: Clinical interpretation of {list(diseases_level)}'],\n"
        "    'pathological staging': 'Staging information based on findings',\n"
        "    'Recommended diet': ['Diet recommendation 1', 'Diet recommendation 2', 'Diet recommendation 3', 'Diet recommendation 4'],\n"
        "    'Recommended exercise': ['Exercise 1', 'Exercise 2', 'Exercise 3', 'Exercise 4'],\n"
        "    'precautions': ['Precaution 1', 'Precaution 2', 'Precaution 3', 'Precaution 4']\n"
        "}\n"
    )
    messages = [
        {"role": "system", "content": "You are an AI assistant that strictly provides responses in dictionary format only, without any explanations or additional text."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = GroqClient.call_groq_api(prompt)
        logger.info(f"GROQ API response: {response[:200]}...")  # Log first 200 chars for debugging
        
        # Clean the response to ensure it's valid Python dict format
        response = response.strip()
        if response.startswith('```python'):
            response = response.replace('```python', '').replace('```', '').strip()
        elif response.startswith('```'):
            response = response.replace('```', '').strip()
        
        return ast.literal_eval(response)
    except (ValueError, SyntaxError) as e:
        logger.error(f"Error parsing GROQ response: {str(e)}")
        logger.error(f"Raw response: {response}")
        
        # Try JSON parsing as fallback
        try:
            import json
            # Replace single quotes with double quotes for JSON compatibility
            json_response = response.replace("'", '"')
            return json.loads(json_response)
        except json.JSONDecodeError:
            logger.error("JSON parsing also failed, using fallback dictionary")
        
        # Return a smart fallback dictionary based on actual predictions
        malignant_detected = any('Malignant' in level or 'Yes' in level for level in diseases_level)
        
        # Determine imaging type based on disease
        if disease_name == "Breast Cancer":
            imaging_type = "Mammography"
        elif disease_name == "Lung Cancer":
            imaging_type = "Chest X-Ray/CT Scan"
        else:
            imaging_type = "Medical Imaging"
        
        if malignant_detected:
            findings = f'{disease_name} detected with malignant characteristics. Abnormal cellular patterns identified.'
            staging = f'Pathological staging required. Based on imaging: Suspicious lesions detected. Further biopsy recommended for definitive staging.'
            imaging_findings = [
                f'{imaging_type}: Abnormal density/mass detected with irregular margins',
                f'Assessment: {", ".join(diseases_level)}. Immediate oncology consultation recommended'
            ]
        else:
            findings = f'{disease_name} screening shows non-malignant findings. No significant abnormalities detected.'
            staging = 'No pathological staging required. Benign findings with no evidence of malignancy.'
            imaging_findings = [
                f'{imaging_type}: Normal tissue architecture with no suspicious lesions',
                f'Assessment: {", ".join(diseases_level)}. Continue routine screening as per guidelines'
            ]
        
        return {
            'detailed findings': findings,
            'clinical examination': f'Patient presents with {", ".join(symptoms) if symptoms else "reported clinical symptoms"}. Physical examination correlates with {disease_name} risk assessment.',
            'imaging studies': imaging_findings,
            'pathological staging': staging,
            'Recommended diet': ['High-protein diet to support immune system', 'Fresh fruits and vegetables rich in antioxidants', 'Adequate hydration (8-10 glasses water daily)', 'Limit processed and high-sugar foods'],
            'Recommended exercise': ['Light aerobic exercise 30 minutes daily as tolerated', 'Breathing exercises and yoga', 'Adequate rest and sleep (7-8 hours)', 'Follow medical guidance for activity level'],
            'precautions': ['Follow all medical instructions strictly', 'Attend regular follow-up appointments', 'Monitor symptoms and report changes immediately', 'Maintain healthy lifestyle and stress management']
        }

def detect_disease_context(user_input: str) -> str:
    """Detect which disease the user is asking about based on keywords."""
    user_input_lower = user_input.lower()
    
    # Breast cancer keywords
    breast_keywords = [
        'breast cancer', 'breast', 'mammogram', 'mammography', 'breast density',
        'benign', 'malignant', 'breast tumor', 'breast mass', 'lump in breast',
        'breast screening', 'breast biopsy', 'mastectomy', 'chemotherapy breast',
        'ductal carcinoma', 'lobular carcinoma', 'her2', 'estrogen receptor',
        'progesterone receptor', 'triple negative', 'breast imaging'
    ]
    
    # Lung cancer keywords  
    lung_keywords = [
        'lung cancer', 'lung', 'pulmonary', 'chest x-ray', 'ct scan chest',
        'lung tumor', 'lung mass', 'lung nodule', 'bronchoscopy', 'thoracic',
        'respiratory', 'cough', 'shortness of breath', 'lung biopsy', 'pneumonia',
        'tissue diagnosis', 'transthoracic needle', 'fine-needle aspiration',
        'computed tomography', 'bronchoscopy brushings', 'lung lesions',
        'central lung', 'peripheral lesions', 'metastasis', 'lymph node',
        'intrathoracic', 'pleural effusion', 'pericardial effusion', 'dysphagia',
        'phrenic nerve', 'superior vena cava', 'horner syndrome', 'dyspnea'
    ]
    
    # Count matches to determine the dominant disease context
    breast_matches = sum(1 for keyword in breast_keywords if keyword in user_input_lower)
    lung_matches = sum(1 for keyword in lung_keywords if keyword in user_input_lower)
    
    # If there are matches, return the disease with more matches
    if lung_matches > breast_matches and lung_matches > 0:
        return 'lung_cancer'
    elif breast_matches > 0:
        return 'breast_cancer'
    
    # Default to general (use all PDFs)
    return None

def get_disease_specific_processor(disease_type: str):
    """Get disease-specific document processor with lazy loading."""
    if disease_type == 'breast_cancer':
        return get_breast_cancer_doc_processor()
    elif disease_type == 'lung_cancer':
        return get_lung_cancer_doc_processor()
    else:
        return get_general_doc_processor()

def chat_with_bot(user_input: str, disease_context: str = None) -> str:
    if not user_input.strip():
        return "Please enter a valid query."
    
    # Use disease context passed from session (from report generation)
    if disease_context:
        # Use disease-specific processor (cached)
        disease_processor = get_disease_specific_processor(disease_context)
        if disease_processor:
            context = disease_processor.retrieve_context(user_input)
            logger.info(f"Using {disease_context.replace('_', ' ').title()} specific RAG data from session")
        else:
            # Fallback to general processor
            context = get_general_doc_processor().retrieve_context(user_input)
            logger.info("Using general RAG data (disease processor not found)")
    else:
        # Fallback: try to detect disease context for standalone chat usage
        detected_disease = detect_disease_context(user_input)
        if detected_disease:
            disease_processor = get_disease_specific_processor(detected_disease)
            context = disease_processor.retrieve_context(user_input)
            logger.info(f"Using {detected_disease.replace('_', ' ').title()} specific RAG data (detected)")
        else:
            # Use general processor for general queries
            context = get_general_doc_processor().retrieve_context(user_input)
            logger.info("Using general RAG data")
    
    # Create a concise, well-formatted prompt
    disease_info = f" regarding {disease_context.replace('_', ' ')}" if disease_context else ""
    
    # Check if asking about patient info from report
    if any(word in user_input.lower() for word in ['my name', 'who am i', 'my problem', 'what is wrong', 'my diagnosis', 'my condition']):
        prompt = f"""Based on the patient report context{disease_info}, answer this question CONCISELY:

Context:
{context}

Question: {user_input}

IMPORTANT: 
- Give a SHORT, DIRECT answer (2-4 sentences max)
- Use bullet points if listing multiple items
- Include only the most relevant information
- Format with line breaks for readability
- Be conversational and clear"""
    else:
        prompt = f"""Based on medical knowledge{disease_info}, answer this question CONCISELY:

Context:
{context}

Question: {user_input}

IMPORTANT:
- Keep answer SHORT and FOCUSED (3-5 sentences)
- Use bullet points for lists
- Format with proper line breaks
- Be clear and direct
- Only include essential information"""
    
    # Use shorter tokens for concise chat responses
    response = GroqClient.call_groq_api(prompt, model=GROQ_MODEL, max_tokens=500, temperature=0.3)
    
    # Ensure response has proper formatting
    return response.strip()

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data.encode('utf-8')
        contact_number = form.contact_number.data
        date_of_birth = form.date_of_birth.data
        city = form.city.data
        user_type = form.user_type.data
        gender = form.gender.data
        age = datetime.now().year - date_of_birth.year - ((datetime.now().month, datetime.now().day) < (date_of_birth.month, date_of_birth.day))
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt()).decode('utf-8')  # Decode to string for storage
        new_user = User(name=name, email=email, password=hashed_password, contact_number=contact_number,
                       date_of_birth=date_of_birth, city=city, user_type=user_type, gender=gender, age=age)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful! Please login.")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Database error during registration: {str(e)}")
            flash("Registration failed. Please try again.")
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data.encode('utf-8')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password, user.password.encode('utf-8')):  # Encode stored password back to bytes
            session['user_id'] = user.id
            session['name'] = user.name
            session['user_type'] = user.user_type
            return redirect(url_for('form'))
        flash("Login failed. Please check your email and password.")
    return render_template('login.html', form=form)

@app.route('/form', methods=['GET', 'POST'])
def form():
    form = DiagnosisForm()
    
    # Get disease-specific hints
    disease_hints = {}
    if request.method == 'POST' and form.disease_name.data:
        disease_hints = DiagnosisFormValidator.get_disease_specific_hints(form.disease_name.data)
    
    return render_template('form.html', form=form, disease_hints=disease_hints)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('name', None)
    session.pop('user_type', None)
    flash("You have been logged out successfully.")
    return redirect(url_for('login'))

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'user_id' not in session:
        flash("Please login first.")
        return redirect(url_for('login'))
    try:
        name = request.form['name']
        dob = datetime.strptime(request.form['dob'], '%Y-%m-%d').date()
        disease_name = request.form['disease_name']
        clinical_history = request.form['clinical_history']
        symptoms_json = request.form.get('symptoms_json', '[]')
        symptoms = tuple(json.loads(symptoms_json)) if symptoms_json else tuple()
        
        # Additional validation for symptoms based on disease type
        validator = DiagnosisFormValidator()
        is_valid, error_message = validator.validate_symptoms_for_disease(list(symptoms), disease_name)
        if not is_valid:
            flash(f"Symptoms validation error: {error_message}")
            return redirect(url_for('form'))
        prepared_by = session.get('name')
        age = datetime.now().year - dob.year - ((datetime.now().month, datetime.now().day) < (dob.month, dob.day))
        ct_images = request.files.getlist('ct_images')
        image_paths = []
        cloudinary_urls = []
        
        # Upload images to Cloudinary
        for image in ct_images:
            if image and image.filename:
                try:
                    # Save locally first (for model processing)
                    filename = image.filename
                    local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image.save(local_path)
                    
                    # Upload to Cloudinary
                    upload_result = cloudinary.uploader.upload(
                        local_path,
                        folder=f"medical_ai/{disease_name.replace(' ', '_').lower()}",
                        public_id=f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        resource_type="image",
                        overwrite=True
                    )
                    
                    # Store both local path (for processing) and Cloudinary URL (for display)
                    relative_path = f'uploads/{filename}'
                    image_paths.append(relative_path)
                    cloudinary_urls.append(upload_result['secure_url'])
                    
                    logger.info(f"Image uploaded to Cloudinary: {upload_result['secure_url']}")
                except Exception as e:
                    logger.error(f"Error uploading to Cloudinary: {str(e)}")
                    # Fallback to local path
                    relative_path = f'uploads/{filename}'
                    image_paths.append(relative_path)
                    cloudinary_urls.append(None)
        image_paths_tuple = tuple(image_paths)
        results = model_predict(image_paths_tuple, disease_name)
        diseases_level = tuple(result[0] for result in results)
        new_image_paths = tuple(result[1] for result in results)
        data = generate_data(name, age, disease_name, clinical_history, symptoms, new_image_paths, diseases_level)
        data = ast.literal_eval(data) if isinstance(data, str) else data
        current_date = datetime.now().strftime('%Y-%m-%d')
        user_type = session.get('user_type')
        
        
        # Append new report details to PERSONAL_REPORTS
        PERSONAL_REPORTS.clear()
        new_report = (
            f"Patient Name: {name}\n"
            f"Date of Birth: {dob}\n"
            f"Age: {age}\n"
            f"Date of Report: {current_date}\n"
            f"Disease Name: {disease_name}\n"
            f"Disease Level: {', '.join(diseases_level)}\n"
            f"Clinical History: {clinical_history}\n"
            f"Symptoms: {', '.join(symptoms)}\n"
            f"Clinical Examination: {data.get('clinical examination', 'N/A')}\n"
            f"Imaging Studies: {', '.join(data.get('imaging studies', ['N/A']))}\n"
            f"Pathological Staging: {data.get('pathological staging', 'N/A')}\n"
            f"Precautions: {', '.join(data.get('precautions', ['N/A']))}\n"
            f"Recommended Diet: {', '.join(data.get('Recommended diet', ['N/A']))}\n"
            f"Recommended Exercise: {', '.join(data.get('Recommended exercise', ['N/A']))}\n"
            f"Prepared By: {prepared_by}\n"
        )
        PERSONAL_REPORTS.append(new_report)
        
        # Store current disease context in session for chat
        disease_mapping = {
            'breast cancer': 'breast_cancer',
            'lung cancer': 'lung_cancer'
        }
        session['current_disease'] = disease_mapping.get(disease_name.lower(), disease_name.lower().replace(' ', '_'))
        
        # Update vector store with disease-specific context
        disease_processor = get_disease_specific_processor(session['current_disease'])
        if disease_processor:
            disease_processor.update_vector_store(PERSONAL_REPORTS, specific_disease=session['current_disease'])
            logger.info(f"Updated {disease_name} specific RAG with current report")
        else:
            get_general_doc_processor().update_vector_store(PERSONAL_REPORTS)

        # Use Cloudinary URLs if available, otherwise fall back to local paths
        display_image_paths = []
        for i, path in enumerate(new_image_paths):
            if i < len(cloudinary_urls) and cloudinary_urls[i]:
                display_image_paths.append(cloudinary_urls[i])
            else:
                display_image_paths.append(path)

        return render_template('report.html', user_type=user_type, name=name, dob=dob, age=age, 
                             disease_name=disease_name, clinical_history=clinical_history, 
                             symptoms=list(symptoms), prepared_by=prepared_by, image_paths=display_image_paths, 
                             diseases_level=list(diseases_level), data=data, current_date=current_date)
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        flash("An error occurred while generating the report. Please try again.")
        return redirect(url_for('form'))

@app.route('/chat-page')
def chat_page():
    current_disease = session.get('current_disease', None)
    disease_display = current_disease.replace('_', ' ').title() if current_disease else 'General Medical'
    return render_template('chat.html', current_disease_context=disease_display)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_query = data.get("query", "").strip()
        if not user_query:
            return jsonify({"response": "Please enter a valid query."}), 400
        
        # Get disease context from session (set during report generation)
        current_disease = session.get('current_disease', None)
        response = chat_with_bot(user_query, disease_context=current_disease)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"response": "An error occurred while processing your request."}), 500

@app.route('/api/disease-hints/<disease_name>')
def get_disease_hints(disease_name):
    """API endpoint to get disease-specific hints for the form."""
    try:
        hints = DiagnosisFormValidator.get_disease_specific_hints(disease_name)
        return jsonify(hints)
    except Exception as e:
        logger.error(f"Error getting disease hints: {str(e)}")
        return jsonify({"error": "Failed to get disease hints"}), 500

def init_app():
    with app.app_context():
        global doc_processor, breast_cancer_predictor, lung_cancer_predictor
        global breast_cancer_doc_processor, lung_cancer_doc_processor
        
        # Create database tables
        db.create_all()
        logger.info("Database tables created/verified")
        
        # LAZY LOADING: Don't pre-load heavy processors/models at startup
        # They will be initialized on first use to avoid memory issues
        logger.info("Application configured for lazy loading (models load on first use)")
        
        global breast_cancer_doc_processor, lung_cancer_doc_processor, general_doc_processor
        
        # Set to None - will be initialized on demand
        general_doc_processor = None
        breast_cancer_doc_processor = None
        lung_cancer_doc_processor = None
        breast_cancer_predictor = None
        lung_cancer_predictor = None
        
        logger.info("✓ Lazy loading enabled - models will load when needed")
        
        # # Uncomment below for eager loading (requires more memory)
        # try:
        #     general_doc_processor = DocumentProcessor()
        #     logger.info("General processor initialized ✓")
        #     
        #     breast_cancer_doc_processor = DocumentProcessor(disease_type='breast_cancer', preload_data=True)
        #     lung_cancer_doc_processor = DocumentProcessor(disease_type='lung_cancer', preload_data=True)
        #     logger.info("All RAG processors pre-loaded successfully ✓")
        #     
        #     logger.info("Initializing AI predictors...")
        #     breast_cancer_predictor = BreastCancerPredictor(model_path=BREAST_CANCER_MODEL_PATH)
        #     lung_cancer_predictor = LungCancerPredictor(model_path=LUNG_CANCER_MODEL_PATH)
        #     logger.info("All AI predictors initialized successfully")
        # except Exception as e:
        #     logger.error(f"Failed to initialize models: {str(e)}")
        
        # Validate environment configuration
        logger.info("Environment configuration:")
        logger.info(f"  Flask Debug: {app.config.get('DEBUG', False)}")
        logger.info(f"  Upload folder: {app.config['UPLOAD_FOLDER']}")
        logger.info(f"  PDF folder: {app.config['PDF_FOLDER']}")
        logger.info(f"  Max upload size: {app.config['MAX_CONTENT_LENGTH']} bytes")
        logger.info(f"  GROQ API configured: {'✓' if GROQ_API_KEY else '✗'}")
    
    logger.info("Application initialized successfully")

if __name__ == '__main__':
    init_app()
    app.run(debug=False)
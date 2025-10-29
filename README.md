# 🏥 Medical AI Diagnosis System: Multi-Disease Intelligent Healthcare Platform

> **Empowering Healthcare through Advanced AI** — Automated Multi-Disease Diagnosis, Intelligent Reports & Real-time Medical Assistance

---

## 📌 Overview

**Medical AI Diagnosis System** is a **production-ready, full-stack AI-powered medical platform** that provides comprehensive healthcare solutions for multiple diseases. The system combines advanced machine learning models, natural language processing, and retrieval-augmented generation (RAG) to deliver accurate diagnoses, detailed medical reports, and intelligent chat assistance.

🎯 **Built for Healthcare Professionals & Patients** using **Flask**, **TensorFlow**, **LangChain**, **FAISS**, and **GROQ API** with support for **2 disease categories** (Breast Cancer & Lung Cancer) and real-time AI assistance.

---

## 🌟 Key Features

### 🔬 **Multi-Disease AI Diagnosis**
- **Breast Cancer Detection** - Mammography & ultrasound analysis
- **Lung Cancer Detection** - Chest X-ray & CT scan analysis

### 🧠 **Advanced AI Capabilities**
- **Deep Learning Models** - CNN-based disease-specific predictions
- **RAG-Powered Chat** - Disease-specific knowledge retrieval
- **Dynamic Report Generation** - AI-generated medical reports
- **Performance Optimized** - Pre-loaded models for instant responses

### 🤖 **Intelligent Medical Assistant**
- **Disease-Specific Chat** - Context-aware medical conversations
- **Real-time Responses** - Instant AI-powered medical guidance
- **Multi-Disease Context** - Automatic disease detection from queries
- **Knowledge Base Integration** - Medical literature and guidelines

### 📊 **Comprehensive Medical Reports**
- **Pathological Staging** - Detailed disease progression analysis
- **Clinical Examination** - Structured medical findings
- **Treatment Recommendations** - Personalized care plans
- **Diet & Exercise Plans** - Disease-specific lifestyle guidance

### 🔐 **Enterprise-Grade Security**
- **Secure Authentication** - Encrypted user management
- **Environment Configuration** - Secure API key management
- **Data Protection** - Patient privacy compliance
- **Session Management** - Secure user sessions

---

## 🏗️ System Architecture

```
Medical AI Diagnosis System/
├── 🧠 AI Models & Predictors
│   ├── Modal/
│   │   ├── Breast Cancer/
│   │   │   ├── breast_cancer.keras
│   │   │   ├── Dataset/
│   │   │   └── Performance Metrics/
│   │   └── Lung Cancer/
│   │       ├── Lung Cancer.keras
│   │       ├── Lung Cancer Dataset/
│   │       └── Performance Metrics/
│   ├── breast_cancer_predictor.py
│   └── lung_cancer_predictor.py
├── 📚 Knowledge Base & RAG
│   └── RAG Data/
│       ├── Breast_Cancer_Rag_Data.pdf
│       └── Lung_Cancer_Rag_Data.pdf
├── 🌐 Web Application
│   ├── main.py                    # Core Flask application
│   ├── form_validators.py         # Multi-disease form validation
│   ├── templates/                 # HTML templates
│   └── static/                    # CSS, JS, assets
├── 🔧 Configuration               
│   ├── .env.example              # Environment template
│   ├── requirements.txt          # Dependencies
│   └── .gitignore               # Git ignore rules
└── 📁 Database & Storage
    ├── instance/                 # Flask instance folder
    └── static/uploads/          # Medical image uploads
```

---

## 🛠️ Technology Stack

| **Category** | **Technologies** |
|--------------|------------------|
| **Backend** | Flask, SQLAlchemy, bcrypt |
| **Frontend** | HTML5, CSS3, Bootstrap 5, JavaScript |
| **AI/ML** | TensorFlow 2.16+, Keras, OpenCV |
| **NLP & RAG** | LangChain, FAISS, HuggingFace Embeddings |
| **APIs** | GROQ API, sentence-transformers |
| **Database** | PostgreSQL (Neon Cloud) with SQLAlchemy ORM |
| **Security** | python-dotenv, werkzeug security |
| **Performance** | functools.lru_cache, optimized loading |

---

## ⚡ Performance Optimizations

### 🚀 **Startup Optimizations**
- **Pre-loaded RAG Processors** - All disease-specific knowledge loaded at startup
- **Model Caching** - AI models loaded once and cached in memory
- **Embedding Pre-computation** - Sentence transformers ready for instant use
- **Vector Store Optimization** - FAISS indices built and cached

### 📈 **Runtime Performance**
- **LRU Caching** - Intelligent caching for frequent operations (500+ cache size)
- **Lazy Loading** - Resources loaded on-demand where appropriate
- **Memory Management** - Efficient resource utilization
- **Response Time** - Sub-second response times for most operations

### 🔧 **Code Optimizations**
- **Error Handling** - Robust error recovery and fallback mechanisms
- **Production Configuration** - Debug mode disabled for performance
- **Resource Pooling** - Efficient database and API connections

---

## 🛠️ Installation & Setup

### 📋 **Prerequisites**
- Python 3.10+ 🐍
- 4GB+ RAM (recommended for AI models)
- 10GB+ disk space (for models and data)
- Internet connection (for API services)

### 🔧 **Quick Installation**

1. **Clone Repository**
```bash
git clone https://github.com/VanshGosavi07/Medical-AI-Diagnosis-System.git
cd Medical-AI-Diagnosis-System
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Setup**
```bash
copy .env.example .env
# Edit .env file with your configuration
# Required: GROQ_API_KEY=your-groq-api-key
# Optional: SECRET_KEY=your-secret-key
```

4. **Get GROQ API Key**
- Visit [GROQ Console](https://console.groq.com/)
- Create account and generate API key
- Add to `.env` file

5. **Launch Application**
```bash
python main.py
```

6. **Access System**
- Open: [http://localhost:5000](http://localhost:5000)
- Create account and start using the system

---

## 🧪 Usage Guide

### 🎯 **Complete Diagnosis Workflow**
1. **👤 User Registration/Login**
2. **🏥 Disease Selection & Form**
3. **📤 Medical Data Upload**
4. **🧠 AI Analysis & Processing**
5. **📄 Report Generation**
6. **💬 Interactive Chat Assistant**

---

## 🔬 Supported Diseases & Capabilities

### 🩺 **Breast Cancer**
- **🖼️ Image Types**: Mammography, Breast Ultrasound
- **🤖 AI Model**: CNN with 95%+ accuracy
- **📋 Symptoms**: Breast lump, nipple discharge, pain, skin changes

### 🫁 **Lung Cancer**
- **🖼️ Image Types**: Chest X-ray, CT scans
- **🤖 AI Model**: Advanced CNN with multi-class classification
- **� Symptoms**: Persistent cough, chest pain, shortness of breath

---

## 🛡️ Security & Compliance

- HIPAA-ready architecture
- Encrypted data storage
- Secure file handling

---

## 🤝 Contributing

We welcome contributions from the healthcare and AI community.

---

## 📧 Support & Contact
- **Email**: vanshgosavi7@gmail.com
- **GitHub**: https://github.com/VanshGosavi07
- **LinkedIn**: https://linkedin.com/in/vanshgosavi07
"# Final-Year-Project-" 

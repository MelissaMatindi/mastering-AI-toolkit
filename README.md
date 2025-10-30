# Mastering the AI Toolkit 🛠️🧠

**AI Tools and Applications Assignment** 

This repository contains our comprehensive submission for the **"Mastering the AI Toolkit"** assignment, demonstrating proficiency in AI frameworks (TensorFlow, PyTorch, Scikit-learn, spaCy) through theoretical analysis, practical implementation, and ethical considerations.

## 📋 Project Overview

**Objective**: Demonstrate understanding of AI tools/frameworks and their real-world applications through theoretical questions, hands-on implementation, and ethical analysis.

**Technologies Used**:
- **Frameworks**: Scikit-learn, TensorFlow/Keras, spaCy
- **Environment**: Python 3.8+, Jupyter Notebook, Google Colab
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, spacy

---


---

## 📚 Assignment Parts & Deliverables

### **Part 1: Theoretical Understanding (40%)**
**Files**: `docs/01_Theoretical_Analysis.pdf`

**Contents**:
- **Short Answer Questions**:
  - Q1: TensorFlow vs PyTorch - Computational graphs, use cases, and selection criteria
  - Q2: Jupyter Notebook use cases in AI development
  - Q3: spaCy advantages over basic Python string operations for NLP
- **Comparative Analysis**: Scikit-learn vs TensorFlow (target applications, ease of use, community support)

**Key Insights**:
- TensorFlow preferred for production deployment; PyTorch for research prototyping
- Jupyter Notebooks excel in iterative model development and data visualization
- spaCy provides pre-trained models and linguistic features for efficient NLP

---

### **Part 2: Practical Implementation (50%)**

#### **Task 1: Classical ML with Scikit-learn**
**File**: `notebooks/01_Iris_DecisionTree_Classifier.ipynb`

**Dataset**: Iris Species Dataset (150 samples, 4 features)
**Model**: Decision Tree Classifier
**Metrics**: Accuracy: 95.3%, Precision: 0.95, Recall: 0.95
**Key Features**:
- Data preprocessing (label encoding, train-test split)
- Model training and evaluation with classification report
- Feature importance visualization

#### **Task 2: Deep Learning with TensorFlow**
**File**: `notebooks/02_MNIST_CNN_Classifier.ipynb`

**Dataset**: MNIST Handwritten Digits (60K train, 10K test)
**Model**: Convolutional Neural Network (CNN)
**Performance**: Test Accuracy: **96.8%** (exceeds 95% target)
**Architecture**:
- 2 Convolutional layers with ReLU activation
- MaxPooling2D layers for dimensionality reduction
- Dense layers with Dropout for regularization
- Softmax output for 10-class classification

**Visualizations**: Sample predictions on 5 test images with confidence scores

#### **Task 3: NLP with spaCy**
**File**: `notebooks/03_Amazon_Reviews_NER_Sentiment.ipynb`

**Dataset**: Amazon Product Reviews (100 sample reviews)
**Tasks**:
- **Named Entity Recognition (NER)**: Extracted product names (e.g., "iPhone 12") and brands (e.g., "Apple", "Samsung")
- **Rule-based Sentiment Analysis**: Positive/Negative classification using keyword matching
**Results**:
- Successfully identified 87% of product entities
- Sentiment accuracy: 82% on validation set

---

### **Part 3: Ethics & Optimization (10%)**
**Files**: `docs/03_Ethics_Optimization.pdf`, `notebooks/04_Model_Debugging_Fix.ipynb`

**Ethical Analysis**:
- **MNIST Bias**: Model may underperform on low-quality or rotated digits; mitigated using TensorFlow Fairness Indicators
- **Amazon Reviews Bias**: Cultural/language bias in sentiment analysis; addressed with diverse training data and rule adjustments
- **Mitigation Strategies**: Data augmentation, fairness metrics, and transparent model documentation

**Debugging Challenge**:
- Fixed dimension mismatch errors in TensorFlow model
- Corrected loss function (binary_crossentropy → categorical_crossentropy)
- Resolved shape broadcasting issues in prediction layer

---

### **Bonus Task: Model Deployment (Extra 10%)**
**Files**: `bonus/05_MNIST_Streamlit_Deployment.py`, `bonus/deployment_screenshot.png`

**Deployment**: Streamlit web application for MNIST digit classification
**Features**:
- Upload handwritten digit images
- Real-time predictions with confidence scores
- Model performance metrics display
**Hosting**: [Live Demo Link](https://my-streamlit-app-url.streamlit.app)

---

## 🎬 Presentation
**File**: `presentation/AI_Tools_Presentation_Video.mp4`

**Duration**: 3 minutes
**Format**: 
1. Project overview and approach 
2. Key technical implementations 
3. Results and insights 
4. Ethical considerations
5. Future improvements 

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab
- Git

### Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/MelissaMatindi/mastering-AI-toolkit.git
   
   cd mastering-AI-toolkit

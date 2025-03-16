import streamlit as st
import re
import os
import time
import torch
import string
import joblib
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import dateparser, datefinder
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.datavalidation import DataValidation

from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from keybert import KeyBERT

# Import conformal classifier (assuming this module exists)
#import conformal_spcr_classifier

# Configure page
st.set_page_config(page_title="SPCR Code Predictor", layout="wide")

# App title and description
st.title("SPCR Code Prediction Tool")
st.markdown("""
This application predicts S (Symptom), P (Problem), C (Cause), and R (Resolution) codes 
for hazardous complaints. Upload your Excel file to generate predictions.
""")

# Define regex pattern for dates
date_pattern = r"""
\b(
    (?:\d{1,2}[-/thstndrd\s]*)?      # Optional day with suffix (1st, 2nd, etc.)
    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
       Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)? # Month names
    [-/\s]*
    (?:\d{1,2})?                     # Optional day number
    [-/\s]*
    \d{2,4}                          # Year (2 or 4 digits)
)\b
"""

# Define ComplaintDataset class for inference
class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Function to extract dates
def extract_dates(*texts):
    """Extract dates from multiple text fields"""
    all_dates = []    
    
    texts = [str(text) if isinstance(text, str) else "" for text in texts]
    
    for text in texts:
        matches = list(datefinder.find_dates(text, strict=True))
        for match in matches:
            if match.tzinfo is not None:  # If datetime has timezone info, remove it
                match = match.replace(tzinfo=None)
            all_dates.append(match)
    
    if all_dates:
        all_dates.sort()
        earliest = all_dates[0].date()
        latest = all_dates[-1].date()
        date_count = len(all_dates)
        days_diff = (latest - earliest).days
        return earliest, latest, date_count, days_diff
    else:
        return None, None, 0, None

# Function to clean narratives
def clean_narratives_of(df, columns, new_column_name):
    """Clean and combine narratives from specified columns into a new column"""
    def clean_text(text):
        if pd.isna(text):  
            return ""        
        text = re.sub(r'\^\^\^.*?\^\^\^', '', text, flags=re.DOTALL)     # Remove text enclosed in ^^^ ... ^^^        
        text = re.sub(r'\*+', ' ', text)                                 # Remove series of * symbols
        text = re.sub(r'\bCHU\b|\bCHUDATA\b|\bPICHUDATA\b', '', text, flags=re.IGNORECASE)
        text = re.sub(date_pattern, '', text)
        return text.strip()    
    
    df[new_column_name] = df[columns].apply(lambda row: ' '.join(clean_text(str(row[col])) for col in columns), axis=1)    
    return df

# Function to extract keywords using KeyBERT
@st.cache_resource
def load_keybert_model():
    return KeyBERT()

def extract_keywords_keybert(texts, stopwords=None, top_n=5):
    """Extract keywords from texts using KeyBERT"""
    kw_model = load_keybert_model()
    keywords_list = []
    
    for text in texts:
        if pd.isna(text) or text == "":
            keywords_list.append("")
            continue
        
        # Extract keywords
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2), 
            stop_words=stopwords, 
            top_n=top_n
        )
        
        # Format keywords as string
        if keywords:
            keywords_str = ", ".join([kw[0] for kw in keywords])
        else:
            keywords_str = ""
        
        keywords_list.append(keywords_str)
    
    return keywords_list

# Function to mark top categories
def mark_top_categories(df, column):
    """Mark top categories covering 80% of the data"""
    category_counts = df[column].value_counts()
    cumulative_percentage = category_counts.cumsum() / category_counts.sum()
    top_categories = cumulative_percentage[cumulative_percentage <= 0.8].index
    df['M_' + column] = df[column].apply(lambda x: x if x in top_categories else 'Other')
    return df

# Function to prepare data for modeling
def prepare_data_for_modelling(input_file):
    with st.spinner("Preparing data for modeling..."):
        progress_bar = st.progress(0)
        
        df = pd.read_excel(input_file)
        progress_bar.progress(10)
        
        # 1. Extract Relevant Columns
        relevant_cols = [
            'ID',                                   
            'Status',                               
            'Customers Issue Description(Full)',    
            'FE\'s Issue Description(Full)',        
            'Actions Taken / Repairs(Full)',        
            'Repair Test / Inspection Data(Full)',  
            'Additional Information(Full)',         
            'Hazardous',                           
            'SPCR Symptom Description',            
            'SPCR Problem Description',            
            'SPCR Root Cause Code',                
            'SPCR  Resolution',                    
            'Symptom Code',
            'Investigation Code',
            'Problem Code',
            'Parts Used'                           
        ]
        
        # Filter columns that exist in the dataframe
        existing_cols = [col for col in relevant_cols if col in df.columns]
        if len(existing_cols) < len(relevant_cols):
            missing_cols = set(relevant_cols) - set(existing_cols)
            st.warning(f"Some columns are missing from the input file: {', '.join(missing_cols)}")
            # Create missing columns with empty values
            for col in missing_cols:
                df[col] = None
        
        df = df[relevant_cols]
        progress_bar.progress(20)
        
        s_col = 'SPCR Symptom Description'
        p_col = 'SPCR Problem Description'
        c_col = 'SPCR Root Cause Code'
        r_col = 'SPCR  Resolution'
        
        # 2. Extract Rows which satisfy Status == Resolved Closed, Hazardous == True
        st.write("Filtering data for Resolved-Closed Status and Hazardous Complaints...")
        df = df[df['Status'] == 'Resolved-Closed']
        df = df[df['Hazardous'] == 'Yes']
        
        if df.empty:
            st.error("No data matches the criteria (Resolved-Closed Status and Hazardous Complaints).")
            return None
        
        progress_bar.progress(30)
        
        # 3. Extract Dates
        st.write("Extracting dates from narratives...")
        date_data = []
        
        for _, row in df.iterrows():
            dates = extract_dates(
                row['Customers Issue Description(Full)'],
                row['FE\'s Issue Description(Full)'],
                row['Actions Taken / Repairs(Full)'],
                row['Repair Test / Inspection Data(Full)'],
                row['Additional Information(Full)']
            )
            date_data.append(dates)
        
        df['Start Date'] = [data[0] for data in date_data]
        df['Closure Date'] = [data[1] for data in date_data]
        df['Number of Iterations'] = [data[2] for data in date_data]
        df['Elapsed Days'] = [data[3] for data in date_data]
        
        progress_bar.progress(40)
        
        # Get unique values for keywords
        symptom_list = df['Symptom Code'].dropna().unique().tolist()
        problem_list = df['Problem Code'].dropna().unique().tolist()
        cause_list = df['Investigation Code'].dropna().unique().tolist()
        
        # 4. Create Clean Narratives
        st.write("Cleaning narratives...")
        # Clean all narratives
        df = clean_narratives_of(df, 
            ['Customers Issue Description(Full)', 
             'FE\'s Issue Description(Full)',
             'Actions Taken / Repairs(Full)',
             'Repair Test / Inspection Data(Full)',
             'Additional Information(Full)'],
            'CleanAllNarratives')
        
        progress_bar.progress(50)
        
        # Clean symptom narratives
        df = clean_narratives_of(df, 
            ['Customers Issue Description(Full)', 
             'FE\'s Issue Description(Full)'],
            'CleanSymptomNarratives')
        
        progress_bar.progress(60)
        
        # Clean cause narratives
        df = clean_narratives_of(df, 
            ['Actions Taken / Repairs(Full)', 
             'Repair Test / Inspection Data(Full)',
             'Additional Information(Full)'],
            'CleanCauseNarratives')
        
        progress_bar.progress(70)
        
        # 5. Extract Keywords for Symptoms
        st.write("Extracting keywords for symptoms...")
        df['KeywordsSymptoms'] = extract_keywords_keybert(
            df['CleanSymptomNarratives'].tolist(), 
            stopwords=symptom_list if symptom_list else None
        )
        
        progress_bar.progress(80)
        
        # 6. Extract Keywords for Causes
        st.write("Extracting keywords for causes...")
        df['KeywordsCause'] = extract_keywords_keybert(
            df['CleanCauseNarratives'].tolist(), 
            stopwords=cause_list if cause_list else None
        )
        
        progress_bar.progress(90)
        
        # 7. Mark Top Categories
        st.write("Marking top categories...")
        # Only mark categories if they exist in the dataframe and have values
        if df[s_col].notna().any():
            df = mark_top_categories(df, s_col)
        if df[p_col].notna().any():
            df = mark_top_categories(df, p_col)
        if df[c_col].notna().any():
            df = mark_top_categories(df, c_col)
        if df[r_col].notna().any():
            df = mark_top_categories(df, r_col)
        
        progress_bar.progress(100)
        
        return df

# Load models function
@st.cache_resource
def load_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Using device: {device}")
    
    try:
        # Load S model
        s_tokenizer = BertTokenizer.from_pretrained("./trained_models/s_model")
        s_model = BertForSequenceClassification.from_pretrained("./trained_models/s_model").to(device)
        s_conformal = conformal_spcr_classifier.load("./trained_conformal_models/s_model", s_tokenizer)
        s_encoder = joblib.load("./trained_models/s_model/label_encoder.pkl")
        
        # Load P model
        p_tokenizer = BertTokenizer.from_pretrained("./trained_models/p_model")
        p_model = BertForSequenceClassification.from_pretrained("./trained_models/p_model").to(device)
        p_conformal = conformal_spcr_classifier.load("./trained_conformal_models/p_model", p_tokenizer)
        p_encoder = joblib.load("./trained_models/p_model/label_encoder.pkl")
        
        # Load C model
        c_tokenizer = BertTokenizer.from_pretrained("./trained_models/c_model")
        c_model = BertForSequenceClassification.from_pretrained("./trained_models/c_model").to(device)
        c_conformal = conformal_spcr_classifier.load("./trained_conformal_models/c_model", c_tokenizer)
        c_encoder = joblib.load("./trained_models/c_model/label_encoder.pkl")
        
        # Load R model
        r_tokenizer = BertTokenizer.from_pretrained("./trained_models/r_model")
        r_model = BertForSequenceClassification.from_pretrained("./trained_models/r_model").to(device)
        r_conformal = conformal_spcr_classifier.load("./trained_conformal_models/r_model", r_tokenizer)
        r_encoder = joblib.load("./trained_models/r_model/label_encoder.pkl")
        
        return {
            's': {'tokenizer': s_tokenizer, 'model': s_model, 'conformal': s_conformal, 'encoder': s_encoder},
            'p': {'tokenizer': p_tokenizer, 'model': p_model, 'conformal': p_conformal, 'encoder': p_encoder},
            'c': {'tokenizer': c_tokenizer, 'model': c_model, 'conformal': c_conformal, 'encoder': c_encoder},
            'r': {'tokenizer': r_tokenizer, 'model': r_model, 'conformal': r_conformal, 'encoder': r_encoder}
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

    #     def predict_and_create_excel(df, models, output_path):
    # with st.spinner('Processing data and making predictions...'):
    #     progress_bar = st.progress(0)
        
    #     # Define column names
    #     s_col = 'SPCR Symptom Description'
    #     p_col = 'SPCR Problem Description'
    #     c_col = 'SPCR Root Cause Code'
    #     r_col = 'SPCR Resolution'
        
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models for S, P, C, R
@st.cache_resource
def load_model(model_path):
    return BertForSequenceClassification.from_pretrained(model_path)

# models   = {
#         "Symptom": "model.safe.tensors",
#         "Problem": "model (1).safetensors",
#         "Cause": "model (2).safetensors",
#         "Resolution": "model (3).safetensors"
#     }

st.success("Models loaded and ready for predictions!")

# Define function to predict codes and create Excel output

def predict_and_create_excel():
    st.write("Function to predict and export to Excel is under construction!")

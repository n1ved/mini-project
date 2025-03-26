import numpy as np
import pandas as pd
from urllib.parse import urlparse
from cuml.ensemble import RandomForestClassifier
import tldextract
import joblib

model_rf = joblib.load('phishing_rf_model_gpu.pkl')
model_svm = joblib.load('phishing_svm_model_gpu.pkl')
def extract_features(url):

    parsed = urlparse(url)
    ext = tldextract.extract(url)

    features = {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'has_ip': 1 if any(segment.isdigit() for segment in parsed.netloc.split('.')) else 0,
        'num_subdomains': len(ext.subdomain.split('.')) if ext.subdomain else 0,
        'path_length': len(parsed.path),
        'https': 1 if parsed.scheme == 'https' else 0,
        'special_chars': sum(c in "@-_%?=" for c in url),
        'tld_risk': 1 if ext.suffix in ['xyz', 'top'] else 0,
        'dot_path_ratio': url.count('.') / (len(parsed.path) + 1e-6),
        'subdomain_depth': np.log1p(len(ext.subdomain.split('.'))) if ext.subdomain else 0
    }

    return pd.DataFrame([features])

def predict_phishing_rf(url):

    features = extract_features(url)

    prediction = model_rf.predict(features)[0]
    print(prediction)
    return "Phishing" if prediction == 0 else "Legitimate"


def predict_phishing_with_confidence_rf(url):
    features = extract_features(url)
    probabilities = model_rf.predict_proba(features)  # Ensure it's a NumPy array
    confidence_phishing = probabilities[0][0]
    result = "Phishing" if confidence_phishing > 0.5 else "Legitimate"
    return result, confidence_phishing

def predict_phishing_svm(url):

    features = extract_features(url)

    prediction = model_svm.predict(features)[0]
    print(prediction)
    return "Phishing" if prediction == 0 else "Legitimate"


def predict_phishing_with_confidence_svm(url):
    features = extract_features(url)
    probabilities = model_svm.predict_proba(features)  # Ensure it's a NumPy array
    confidence_phishing = probabilities[0][0]
    result = "Phishing" if confidence_phishing > 0.5 else "Legitimate"
    return result, confidence_phishing

def predict_phishing_combiner(url):
    features = extract_features(url)
    isPhishing = False
    probabilities_rf = model_rf.predict_proba(features)
    confidence_phishing_rf = probabilities_rf[0][0]
    if(confidence_phishing_rf > 0.45 and confidence_phishing_rf < 0.55):
        probabilities_svm = model_svm.predict_proba(features)
        confidence_phishing_svm = probabilities_svm[0][0]
        p_svm = str(confidence_phishing_svm)
        if(confidence_phishing_svm > 0.5):
            isPhishing = True
    else:
        isPhishing = True if  confidence_phishing_rf>0.5 else False

    result = "Phishing" if isPhishing else "Legitimate"
    p_rf = str(confidence_phishing_rf)
    if(confidence_phishing_rf > 0.45 and confidence_phishing_rf < 0.55):
        return result,p_rf,p_svm
    else:
        return result,p_rf
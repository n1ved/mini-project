from flask import request, jsonify
from app import app
import joblib
import pandas as pd
import tldextract
from urllib.parse import urlparse
from nltk.tokenize import RegexpTokenizer
import validators
import sys
from sklearn.base import BaseEstimator, TransformerMixin

# Add the Converter class definition
class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()

# Make the Converter class available in __main__ namespace
sys.modules['__main__'].Converter = Converter

class PhishingURLDetector:
    def __init__(self, model_path='app/models/phishing_url_svc_model.joblib'):
        self.model = joblib.load(model_path)
        self.svm = joblib.load('app/models/phishing_url_svc_model.joblib')
        self.rf = joblib.load('app/models/phishing_url_rf_model.joblib')
        self.tokenizer = RegexpTokenizer(r'[A-Za-z]+')

    # Rest of your code remains the same
    def parse_url(self, url: str) -> pd.DataFrame:
        try:
            no_scheme = not url.startswith('https://') and not url.startswith('http://')
            if no_scheme:
                parsed_url = urlparse(f"http://{url}")
            else:
                parsed_url = urlparse(url)

            url_features = pd.DataFrame({
                'length': [len(url)],
                'tld': [tldextract.extract(parsed_url.netloc).suffix or 'None'],
                'is_ip': [bool(parsed_url.netloc.replace('.','').isnumeric())],
                'domain_hyphens': [parsed_url.netloc.count('-')],
                'domain_underscores': [parsed_url.netloc.count('_')],
                'path_hyphens': [parsed_url.path.count('-')],
                'path_underscores': [parsed_url.path.count('_')],
                'slashes': [parsed_url.path.count('/')],
                'full_stops': [parsed_url.path.count('.')],
                'num_subdomains': [self._get_num_subdomains(parsed_url.netloc)],
                'domain_tokens': [self._tokenize_domain(parsed_url.netloc)],
                'path_tokens': [self._tokenize_path(parsed_url.path)]
            })

            return url_features

        except Exception as e:
            print(f"Error parsing URL: {e}")
            return None

    def _get_num_subdomains(self, netloc: str) -> int:
        subdomain = tldextract.extract(netloc).subdomain
        return subdomain.count('.') + 1 if subdomain else 0

    def _tokenize_domain(self, netloc: str) -> str:
        split_domain = tldextract.extract(netloc)
        no_tld = str(split_domain.subdomain + '.' + split_domain.domain)
        return " ".join(map(str, self.tokenizer.tokenize(no_tld)))

    def _tokenize_path(self, path: str) -> str:
        return " ".join(map(str, self.tokenizer.tokenize(path)))

    def predict(self, url: str) -> dict:
        url_features = self.parse_url(url)

        if url_features is None:
            return {
                'error': 'Could not parse URL',
                'is_phishing': None,
                'confidence': None
            }

        try:
            prediction = self.model.predict(url_features)
            try:
                proba = self.model.predict_proba(url_features)[0]
            except AttributeError:
                proba = None
            try:
                confidence = self.model.decision_function(url_features)[0]
            except AttributeError:
                confidence = None

            return {
                'url': url,
                'prediction': prediction,
                'probability': proba,
                'confidence': confidence
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'error': 'Prediction failed',
                'is_phishing': None,
                'confidence': None
            }

    def predict_weighted(self, url: str) -> dict:
        url_features = self.parse_url(url)
        rf_percent = None
        rf_prediction = None
        svm_percent = None
        svm_prediction = None

        if url_features is None:
            return {
                'error': 'Cannot Parse URL',
                'is_phishing': None,
                'confidence': None,
            }

        try:
            rf_percent = self.rf.predict_proba(url_features)[0][0]
            rf_prediction = self.rf.predict(url_features)[0]
        except AttributeError:
            rf_percent = None

        if rf_percent is not None and rf_percent >= 0.4 and rf_percent <= 0.6:
            try:
                svm_percent = self.svm.decision_function(url_features)[0]
                svm_prediction = self.svm.predict(url_features)[0]
            except:
                svm_percent = None

        return {
            'error': None,
            'is_phishing': None,
            'confidence': rf_percent if svm_percent is None else svm_percent,
            'prediction': ('good' if rf_percent < 0.5 else 'bad') if svm_prediction is None else ('good' if svm_prediction == 0 else 'bad')
        }

# Initialize the detector
detector = PhishingURLDetector("app/models/phishing_url_svc_model.joblib")

@app.route('/predict', methods=['POST','OPTIONS'])
def predict():
    """API endpoint that receives a URL, extracts features, and predicts phishing status."""
    if request.method == 'OPTIONS':
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
            return '', 200, headers
    data = request.get_json()
    url = data.get("url")

    if not url or not validators.url(url):
        return jsonify({"error": "Invalid URL"}), 400

    # Use the detector to make predictions
    result = detector.predict(url)

    if 'error' in result and result['error']:
        return jsonify({"error": result['error']}), 500

    # Extract prediction from the result
    print(result)
    prediction = result['prediction']
    is_phishing = bool(prediction[0] if isinstance(prediction, (list, tuple)) else prediction)
    confidence = result['confidence']
    # Return the result to the browser extension
    return jsonify({
        "url": url,
        "is_phishing": str(prediction[0]),
        "confidence":confidence
    })

@app.route('/predict/weighted', methods=['POST'])
def predict_weighted():
    """API endpoint that uses weighted prediction approach for more accurate phishing detection."""
    data = request.get_json()
    url = data.get("url")

    if not url or not validators.url(url):
        return jsonify({"error": "Invalid URL"}), 400

    # Use the detector's weighted prediction method
    result = detector.predict_weighted(url)

    if 'error' in result and result['error']:
        return jsonify({"error": result['error']}), 500

    # Determine if it's phishing based on the prediction
    is_phishing = result['prediction'] == 'bad' or result['prediction'] == 1

    return jsonify({
        "url": url,
        "is_phishing": is_phishing,
        "confidence": result['confidence'],
        "prediction": "phishing" if is_phishing else "legitimate"
    })

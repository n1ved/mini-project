from phishion import predict_phishing_with_confidence_svm
test_urls = [
    "https://www.paypal.com",          # Legit
    "http://paypall-security.xyz",     # Phishing
    "https://github.com/n1ved/auth",   # Legit
    "https://glthub-verification.co"    # Phishing
]

for url in test_urls:
    print(f"{url} → {predict_phishing_with_confidence_svm(url)}")
from phishion import predict_phishing_combiner
test_urls = [
    "https://www.paypal.com",         
    "http://paypall-security.xyz",
    "https://github.com/n1ved/auth",
    "https://glthub-verification.co"
]

for url in test_urls:
    print(f"{url} → {predict_phishing_combiner(url)}")
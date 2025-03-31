import pandas as pd
from urllib.parse import urlparse
import tldextract

class URLFeatureExtractor:
    """Extracts relevant phishing detection features from a given URL."""
    
    def _get_num_subdomains(self, netloc: str) -> int:
        """Returns the number of subdomains in the URL."""
        return netloc.count('.') - 1  

    def _tokenize_domain(self, netloc: str) -> int:
        """Returns the number of unique tokens in the domain."""
        return len(set(netloc.replace('-', '.').split('.')))

    def _tokenize_path(self, path: str) -> int:
        """Returns the number of unique tokens in the URL path."""
        return len(set(path.replace('-', '/').split('/')))

    def parse_url(self, url: str) -> pd.DataFrame:
        """Extracts features from the given URL and returns them as a Pandas DataFrame."""
        try:
            # Ensure the URL has a scheme
            no_scheme = not url.startswith('https://') and not url.startswith('http://')
            parsed_url = urlparse(f"http://{url}") if no_scheme else urlparse(url)

            # Extract features
            url_features = pd.DataFrame({
                'length': [len(url)],
                'tld': [tldextract.extract(parsed_url.netloc).suffix or 'None'],
                'is_ip': [bool(parsed_url.netloc.replace('.', '').isnumeric())],
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

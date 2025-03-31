# First define the Converter class
from sklearn.base import BaseEstimator, TransformerMixin

class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()

# Then import your app
from app import app

if __name__ == '__main__':
    app.run(debug=True)

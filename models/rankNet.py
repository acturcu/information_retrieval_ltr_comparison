import pyterrier as pt
import numpy as np
from LambdaRankNN import LambdaRankNN
from sklearn.base import BaseEstimator

class RankNetWrapper(BaseEstimator):
    def __init__(self, input_size):
        self.model = LambdaRankNN(
            input_size=input_size,
            hidden_layer_sizes=(32, 16),
            activation=('relu', 'relu'),
            solver='adam'
        )

    def fit(self, X, y, qid):
        self.model.fit(X, y, qid, epochs=10)
        return self

    def predict(self, X):
        return self.model.predict(X)

def get_ranknet_model(base_model: pt.Transformer):
    class RankNetLTR(pt.Transformer):
        def __init__(self):
            self.model = None
            self.input_size = None

        def fit(self, topics, qrels, valid_topics=None, valid_qrels=None):
            df = base_model.transform(topics)
            df = df.merge(qrels, on=["qid", "docno"])
            X = np.vstack(df["features"].values)
            y = df["label"].values
            qid = df["qid"].astype("category").cat.codes.values

            self.input_size = X.shape[1]
            self.model = RankNetWrapper(self.input_size)
            self.model.fit(X, y, qid)
            return self

        def transform(self, topics):
            df = base_model.transform(topics)
            X = np.vstack(df["features"].values)
            df["score"] = self.model.predict(X)
            return df.sort_values(["qid", "score"], ascending=[True, False])

    return RankNetLTR()
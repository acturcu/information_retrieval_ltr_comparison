import pyterrier as pt
from sklearn.svm import SVR

def get_rankSVM_model(base_model: pt.terrier.FeaturesRetriever):
    return base_model >> pt.ltr.apply_learned_model(SVR())

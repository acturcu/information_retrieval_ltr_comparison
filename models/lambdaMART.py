import pyterrier as pt
from lightgbm import LGBMRanker

def get_lambdaMART_model(base_model: pt.terrier.FeaturesRetriever):
    return base_model >> pt.ltr.apply_learned_model(
    LGBMRanker(
        metric="ndcg",
        importance_type="gain",
    ),
    form="ltr",
)
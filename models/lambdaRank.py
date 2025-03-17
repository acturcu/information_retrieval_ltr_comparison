import pyterrier as pt
from lightgbm import LGBMRanker

def get_lambdaRank_model(base_model: pt.terrier.FeaturesRetriever):
    return base_model >> pt.ltr.apply_learned_model(
    LGBMRanker(
        metric="ndcg",
        importance_type="gain",
        rank="pairwise",
    ),
    form="ltr",
)
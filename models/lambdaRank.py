import pyterrier as pt
from lightgbm import LGBMRanker

def get_lambdaRank_model(base_model: pt.terrier.FeaturesRetriever):
    return base_model >> pt.ltr.apply_learned_model(
        LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            learning_rate=0.05,
            num_leaves=16,
            boosting_type="gbdt",
            index="features",
        ),
        form="ltr",
    )
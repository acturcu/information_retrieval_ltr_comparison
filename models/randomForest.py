import pyterrier as pt
import fastrank

def get_random_forest_model(base_model: pt.terrier.FeaturesRetriever):
    train_request = fastrank.TrainRequest.random_forest()
    train_request.params.num_trees = 100  # You can tweak this
    return base_model >> pt.ltr.apply_learned_model(train_request, form="fastrank")
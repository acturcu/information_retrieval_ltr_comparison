import pyterrier as pt
import fastrank

def get_coord_ascent_model(base_model: pt.terrier.FeaturesRetriever):
    # Setup FastRank's coordinate ascent learner
    train_request = fastrank.TrainRequest.coordinate_ascent()
    params = train_request.params
    params.init_random = True
    params.normalize = True
    params.seed = 1234567

    # Build PyTerrier pipeline using FastRank model
    return base_model >> pt.ltr.apply_learned_model(train_request, form="fastrank")
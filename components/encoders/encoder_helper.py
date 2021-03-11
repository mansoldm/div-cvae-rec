from components.encoders.embedding_encoders import get_slate_conditioning_encoder_from_matrix
from components.encoders.feature_encoders import get_feature_encoder


def get_slate_conditioning_encoders(history_encoder, item_item_scores, use_diverse_model: bool,
                                    diversity_encoder_name: str, slate_size: int):
    if use_diverse_model:
        slate_conditioning_encoder = get_slate_conditioning_encoder_from_matrix(slate_size, item_item_scores)
        diversity_encoder = get_feature_encoder(diversity_encoder_name, slate_size)
        cvae_conditioning_size = history_encoder.output_size + diversity_encoder.output_size
    else:
        slate_conditioning_encoder = None
        diversity_encoder = None
        cvae_conditioning_size = history_encoder.output_size

    return cvae_conditioning_size, diversity_encoder, slate_conditioning_encoder
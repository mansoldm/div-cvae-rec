import torch
from torch import nn


class FeatureEncoder(nn.Module):
    def __init__(self, binary_input_size, *args):
        super(FeatureEncoder, self).__init__()
        self.binary_input_size = binary_input_size

    def forward(self, binary_feature):
        raise NotImplementedError


class SumFeatureEncoder(FeatureEncoder):
    def __init__(self, binary_input_size):
        super(SumFeatureEncoder, self).__init__(binary_input_size)
        self.output_size = 1

    def forward(self, binary_feature):
        return torch.sum(binary_feature, dim=1).unsqueeze(1).type(torch.FloatTensor)


class MeanBinaryEncoder(SumFeatureEncoder):
    def forward(self, binary_feature):
        return torch.mean(binary_feature, dim=1).unsqueeze(1).type(torch.FloatTensor)


class IdentityFeatureEncoder(FeatureEncoder):
    def __init__(self, binary_input_size):
        super(IdentityFeatureEncoder, self).__init__(binary_input_size)
        self.output_size = self.binary_input_size

    def forward(self, binary_feature):
        return binary_feature.type(torch.FloatTensor)


def get_feature_encoder(encoding_type, feature_size) -> FeatureEncoder:

    if encoding_type == 'sum':
        return SumFeatureEncoder(feature_size)

    elif encoding_type == 'mean':
        return MeanBinaryEncoder(feature_size)

    elif encoding_type == 'identity':
        return IdentityFeatureEncoder(feature_size)

    raise NotImplementedError(f'Binary encoding type \'{encoding_type}\' not implemented!')

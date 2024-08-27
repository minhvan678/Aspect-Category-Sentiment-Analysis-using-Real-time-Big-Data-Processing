import torch
from transformers import AutoTokenizer,AutoModel

ASPECT = ['FACILITIES#CLEANLINESS', 'FACILITIES#COMFORT',
       'FACILITIES#DESIGN&FEATURES', 'FACILITIES#GENERAL',
       'FACILITIES#MISCELLANEOUS', 'FACILITIES#PRICES',
       'FACILITIES#QUALITY', 'FOOD&DRINKS#MISCELLANEOUS',
       'FOOD&DRINKS#PRICES', 'FOOD&DRINKS#QUALITY',
       'FOOD&DRINKS#STYLE&OPTIONS', 'HOTEL#CLEANLINESS', 'HOTEL#COMFORT',
       'HOTEL#DESIGN&FEATURES', 'HOTEL#GENERAL', 'HOTEL#MISCELLANEOUS',
       'HOTEL#PRICES', 'HOTEL#QUALITY', 'LOCATION#GENERAL',
       'ROOMS#CLEANLINESS', 'ROOMS#COMFORT', 'ROOMS#DESIGN&FEATURES',
       'ROOMS#GENERAL', 'ROOMS#MISCELLANEOUS', 'ROOMS#PRICES',
       'ROOMS#QUALITY', 'ROOM_AMENITIES#CLEANLINESS',
       'ROOM_AMENITIES#COMFORT', 'ROOM_AMENITIES#DESIGN&FEATURES',
       'ROOM_AMENITIES#GENERAL', 'ROOM_AMENITIES#MISCELLANEOUS',
       'ROOM_AMENITIES#PRICES', 'ROOM_AMENITIES#QUALITY',
       'SERVICE#GENERAL']

class FeatureExtractor(torch.nn.Module):
  def __init__(self, pretrained_path):
    super(FeatureExtractor,self).__init__()
    self.cell = AutoModel.from_pretrained(pretrained_path)

  def forward(self, input):
    seq_out, pooled_out = self.cell(**input)[:2]

    return pooled_out
  
class MyModel(torch.nn.Module):
  def __init__(self,pretrained_path,num_classes,num_polarity):
    super(MyModel, self).__init__()
    self.feature_extractor = FeatureExtractor(pretrained_path)
    self.linear_each_class = torch.nn.ModuleList([torch.nn.Linear(768, num_polarity) for i in range(num_classes)])

  def forward(self, input):
    cls_token = self.feature_extractor(input) # (batch_size,1)

    batch_size = cls_token.size()[0]

    out = []
    for i in range(len(self.linear_each_class)):
      out.append(self.linear_each_class[i](cls_token))

    out = torch.cat(out, dim = 1).view(batch_size, len(ASPECT),4)

    return out
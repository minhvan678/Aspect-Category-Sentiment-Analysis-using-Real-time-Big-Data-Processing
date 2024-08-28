from transformers import AutoTokenizer,AutoModel
import torch

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
    self.num_classes = num_classes

  def forward(self, input):
    cls_token = self.feature_extractor(input) # (batch_size,1)

    batch_size = cls_token.size()[0]

    out = []
    for i in range(len(self.linear_each_class)):
      out.append(self.linear_each_class[i](cls_token))

    out = torch.cat(out, dim = 1).view(batch_size, self.num_classes,4)

    return out
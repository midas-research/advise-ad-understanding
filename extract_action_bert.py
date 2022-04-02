import torch
from transformers import BertTokenizer, BertModel
import json
import numpy as np

input_dir = "data/train/QA_Combined_Action_Reason_train.json"
output_feature_path = "output/stmt_features_bert_train.npy"
batch_size = 64
max_length = 100

device = torch.device('cuda:1')

with open(input_dir, 'r') as fp:
    data = json.loads(fp.read())
    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model = model.to(device)

def extract(sentence):
    enc = torch.zeros([1, max_length], dtype=torch.int32).to(device)
    a = tokenizer(sentence, return_tensors='pt')['input_ids'].to(device)
    s = min(max_length, list(a.shape)[-1])
    enc[:,:s] = a[:,:s]
    enc = model(enc).pooler_output.squeeze(0)
    return enc.detach().cpu().numpy()
  
result = {}      
for n, (image_id, examples) in enumerate(data.items()):
    l1 = []
    for i in examples:
      l2 = []
      for j in i:
        l2.append(extract(j))
      l1.append(l2)
    result[image_id] = l1
    if n%1000 == 0:
      print(n)
    
# Write results.
print(len(result), len(data))
assert len(result) == len(data)
print('Saving')
with open(output_feature_path, 'wb') as wp:
    np.save(wp, result)

print('Saving done.')
    
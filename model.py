import torch.nn as nn
from pytorch_transformers import BertForSequenceClassification, modeling_bert
from pytorch_transformers.modeling_bert import BertConfig

bert_model = BertForSequenceClassification(BertConfig()).from_pretrained('bert-base-uncased')

print(bert_model)

class BertForClassification(modeling_bert.BertPreTrainedModel):
    def __init__(self):
        super(BertForClassification, self).__init__(BertConfig())

        self.embeddings = bert_model.bert.embeddings
        self.encoder = bert_model.bert.encoder
        self.pooler = bert_model.bert.pooler
        self.dropout = bert_model.dropout
        self.classifier = bert_model.classifier
        self.prediction = nn.Sequential( 
                nn.Linear(768, 64), 
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, 1),
        )

        self.head_mask = [None] * 12

        self.apply(self._init_weights)

    def forward(self, text, position_ids, attention_mask):
                      
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0   # ( 64, 1, 1, 71)

        embeddings = self.embeddings(text)
        hidden_states = self.encoder(embeddings, attention_mask, head_mask=self.head_mask)[0] # (64, 71, 768)
        output = self.pooler(hidden_states) # (64, 768)
        output = self.dropout(output) # (64, 768)
        output = self.prediction(output) # (64, 2)
        return output


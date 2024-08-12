import torch
import torch.nn as nn
from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel


class BaseModel(nn.Module):
    def __init__(self, transformers_model_name, num_classes, fc_dropout=0.5, attention_probs_dropout_prob=0.3,
                 hidden_dropout_prob=0.3,
                 output_hidden_states=True, output_attentions=False, emb_dim=256):
        self.config = RobertaConfig.from_pretrained(transformers_model_name, num_labels=num_classes)
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.return_dict = True
        self.config.output_hidden_states = True
        self.config.output_attentions = True
        self.config.classifier_dropout = fc_dropout
        super(BaseModel, self).__init__()

        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

        self.encoder = RobertaForClassification.from_pretrained(transformers_model_name, config=self.config,
                                                                output_hidden_states=self.output_hidden_states,
                                                                output_attentions=self.output_attentions)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        return self.encoder(input_ids, token_type_ids, attention_mask)


class RobertaForClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, output_hidden_states, output_attentions):
        super(RobertaForClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        outputs = self.roberta(**kwargs)
        sequence_output = outputs[0]
        logits, embedding = self.classifier(sequence_output)

        output = {'logits': logits}
        if self.output_attentions:
            output['attention'] = outputs['attentions']

        if self.output_hidden_states:
            output['hidden_state'] = embedding

        return output


class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = torch.tanh(x)
        embedding = x
        x = self.dropout(x)
        x = self.out_proj(x)
        return x, embedding


if __name__ == '__main__':
    from transformers import RobertaTokenizerFast

    transformers_model_name = 'codebert'
    num_classes = 10
    # config = AutoConfig.from_pretrained(transformers_model_name)
    # auto_model = AutoModel.from_config(config)
    tokenizer = RobertaTokenizerFast.from_pretrained(transformers_model_name)
    model = BaseModel(transformers_model_name, num_classes)

    inputs = tokenizer('Hello World!', return_tensors='pt')
    print('Inputs:', inputs)

    outputs = model(**inputs)
    print('Output:', outputs)
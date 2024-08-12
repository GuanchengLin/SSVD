import torch
import torch.nn as nn
from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
import numpy as np
from torch.autograd import Function


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

    def step_grl(self):
        self.encoder.classifier.grl_layer.step()


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
        logits, embedding, pseudo_out, worst_out = self.classifier(sequence_output)

        output = {'logits': logits, 'pseudo_out': pseudo_out, 'worst_out': worst_out}
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

        self.pseudo_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size * 2, config.num_labels)
        )

        self.worst_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.num_labels)
        )

        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                       auto_step=False)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = torch.tanh(x)

        main_out = self.dropout(x)
        main_out = self.out_proj(main_out)

        pseudo_out = self.pseudo_head(x)

        f_adv = self.grl_layer(x)

        worst_out = self.worst_head(f_adv)

        return main_out, x, pseudo_out, worst_out


class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha=1.0, lo=0.0, hi=1., max_iters=1000., auto_step=False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float32(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff=1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


if __name__ == '__main__':
    from transformers import RobertaTokenizerFast

    transformers_model_name = 'codebert'
    num_classes = 2
    # config = AutoConfig.from_pretrained(transformers_model_name)
    # auto_model = AutoModel.from_config(config)
    tokenizer = RobertaTokenizerFast.from_pretrained(transformers_model_name)
    model = BaseModel(transformers_model_name, num_classes)
    model_dict = model.state_dict()
    pretrain_dict = torch.load('./checkpoint/best_teacher.pth', map_location=torch.device('cpu'))
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    inputs = tokenizer('Hello World!', return_tensors='pt')
    print('Inputs:', inputs)

    outputs = model(**inputs)
    print('Output:', outputs)
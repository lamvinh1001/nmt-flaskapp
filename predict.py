from models import Transformer, translate_sentence
from load_vocab import get_fields
import torch
import torch.nn as nn


def predict_sentence(sentence):
    opt = {
        'max_strlen': 160,
        'device': 'cuda',
        'd_model': 512,
        'n_layers': 6,
        'heads': 8,
        'dropout': 0.1,
        'k': 5,
    }
    SRC, TRG = get_fields()
    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_tran = Transformer(len(SRC.vocab), len(
        TRG.vocab), opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout'])
    for p in model_tran.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    state = torch.load('transformer.pth')
    model_tran.load_state_dict(state)

    model_tran.to(device)
    trans_sent = translate_sentence(
        sentence, model_tran, SRC, TRG, opt['device'], opt['k'], opt['max_strlen'])
    return trans_sent


# sentence = 'Sorry Sir! Can you tell me where you come from?'
# out_put = predict_sentence(sentence)
# print(out_put)

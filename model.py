import torch
from torch import nn

embedding_dim_col = {
    'bert-base-cased': 'hidden_size',
    'bert-base-uncased': 'hidden_size',
    'albert-base-v2': 'hidden_size',
    'albert-large-v2': 'hidden_size',
    'distilbert-base-cased': 'dim',
    'distilbert-base-uncased': 'dim',
}


class GRUBERT(nn.Module):
    def __init__(self, bert, config):
        super().__init__()
        self.config = config
        self.bert = bert
        self.embedding_dim = self.bert.config.to_dict()[embedding_dim_col[config['pretrained_model']]]
        self.gru11 = nn.GRU(self.embedding_dim, 512, num_layers=1, batch_first=True)
        self.gru12 = nn.GRU(512, 256, num_layers=1, batch_first=True)
        self.gru13 = nn.GRU(256, 128, num_layers=1, batch_first=True)
        self.gru21 = nn.GRU(self.embedding_dim, 512, num_layers=1, batch_first=True)
        self.gru22 = nn.GRU(512, 256, num_layers=1, batch_first=True)
        self.gru23 = nn.GRU(256, 128, num_layers=1, batch_first=True)

        self.fc = nn.Linear(256, config['out_dim'])
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, text):
        # text = [batch size, sent len]

        if self.config['gru_only']:
            with torch.no_grad():
                embedding = self.bert(text)[0]  # embedding = [batch size, sent len, emb dim]
#             print(embedding.shape)
        else:
            embedding = self.bert(text)[0]  # embedding = [batch size, sent len, emb dim]

        output1, _ = self.gru11(embedding)
        output1 = self.dropout1(output1)  # output1 = [batch size, sent len, 512]
        
        output1, _ = self.gru12(output1)
        output1 = self.dropout2(output1)  # output1 = [batch size, sent len, 256]
        
        _, hidden1 = self.gru13(output1)  # hidden1 = [1, batch size, 128]

        reversed_embedding = torch.from_numpy(embedding.detach().cpu().numpy()[:, ::-1, :].copy()).to(self.DEVICE)
        
        output2, _ = self.gru21(reversed_embedding)
        output2 = self.dropout1(output2)  # output2 = [batch size, sent len, 512]
        
        output2, _ = self.gru22(output2)
        output2 = self.dropout2(output2)  # output1 = [batch size, sent len, 256]
        
        _, hidden2 = self.gru23(output2)  # hidden2 = [1, batch size, 128]
        
        hidden = self.dropout2(torch.cat((hidden1[-1, :, :], hidden2[-1, :, :]), dim=1))  # hidden = [batch size, 256]

        output = self.fc(hidden)  # output = [batch size, out dim]

        return output

class GRUBERT2(nn.Module):
    def __init__(self, bert, embedding_dim, output_dim):
        super().__init__()
        self.bert = bert
        self.embedding_dim = embedding_dim
        self.gru11 = nn.GRU(self.embedding_dim, 512, num_layers=1, batch_first=True)
        # self.gru12 = nn.GRU(512, 256, num_layers=1, batch_first=True)
        self.gru13 = nn.GRU(512, 256, num_layers=1, batch_first=True)
        self.gru21 = nn.GRU(self.embedding_dim, 512, num_layers=1, batch_first=True)
        # self.gru22 = nn.GRU(512, 256, num_layers=1, batch_first=True)
        self.gru23 = nn.GRU(512, 256, num_layers=1, batch_first=True)

        self.fc = nn.Linear(512, output_dim)
        
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, text):
        # text = [batch size, sent len]

        # with torch.no_grad():
        embedding = self.bert(text)[0]  # embedding = [batch size, sent len, emb dim]
#             print(embedding.shape)

        output1, _ = self.gru11(embedding)
        output1 = self.dropout1(output1)  # output1 = [batch size, sent len, 512]
        
        # output1, _ = self.gru12(output1)
        # output1 = self.dropout2(output1)  # output1 = [batch size, sent len, 256]
        
        _, hidden1 = self.gru13(output1)  # hidden1 = [1, batch size, 128]

        reversed_embedding = torch.from_numpy(embedding.detach().cpu().numpy()[:, ::-1, :].copy()).to(self.DEVICE)
        
        output2, _ = self.gru21(reversed_embedding)
        output2 = self.dropout1(output2)  # output2 = [batch size, sent len, 512]
        
        # output2, _ = self.gru22(output2)
        # output2 = self.dropout2(output2)  # output1 = [batch size, sent len, 256]
        
        _, hidden2 = self.gru23(output2)  # hidden2 = [1, batch size, 128]
        
        hidden = self.dropout2(torch.cat((hidden1[-1, :, :], hidden2[-1, :, :]), dim=1))  # hidden = [batch size, 256]

        output = self.fc(hidden)  # output = [batch size, out dim]

        return output

class LSTMBERT(nn.Module):
    def __init__(self, bert, embedding_dim, output_dim):
        super().__init__()
        self.bert = bert
        self.embedding_dim = embedding_dim
        self.gru11 = nn.LSTM(self.embedding_dim, 512, num_layers=1, batch_first=True)
        self.gru12 = nn.LSTM(512, 256, num_layers=1, batch_first=True)
        self.gru13 = nn.LSTM(256, 128, num_layers=1, batch_first=True)
        self.gru21 = nn.LSTM(self.embedding_dim, 512, num_layers=1, batch_first=True)
        self.gru22 = nn.LSTM(512, 256, num_layers=1, batch_first=True)
        self.gru23 = nn.LSTM(256, 128, num_layers=1, batch_first=True)

        self.fc = nn.Linear(256, output_dim)
        
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, text):
        # text = [batch size, sent len]

        # with torch.no_grad():
        embedding = self.bert(text)[0]  # embedding = [batch size, sent len, emb dim]
#             print(embedding.shape)

        output1, _ = self.gru11(embedding)
        output1 = self.dropout1(output1)  # output1 = [batch size, sent len, 512]
        
        output1, _ = self.gru12(output1)
        output1 = self.dropout2(output1)  # output1 = [batch size, sent len, 256]
        
        _, hidden1 = self.gru13(output1)  # hidden1 = [1, batch size, 128]

        reversed_embedding = torch.from_numpy(embedding.detach().cpu().numpy()[:, ::-1, :].copy()).to(self.DEVICE)
        
        output2, _ = self.gru21(reversed_embedding)
        output2 = self.dropout1(output2)  # output2 = [batch size, sent len, 512]
        
        output2, _ = self.gru22(output2)
        output2 = self.dropout2(output2)  # output1 = [batch size, sent len, 256]
        
        _, hidden2 = self.gru23(output2)  # hidden2 = [1, batch size, 128]
        
        hidden = self.dropout2(torch.cat((hidden1[-1, :, :], hidden2[-1, :, :]), dim=1))  # hidden = [batch size, 256]

        output = self.fc(hidden)  # output = [batch size, out dim]

        return output
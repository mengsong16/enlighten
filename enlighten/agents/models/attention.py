import torch
import torch.nn as nn

from torchinfo import summary

class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim):
        super(Attention, self).__init__()
        # hidden states --> embedding
        self.U = nn.Linear(hidden_dim, 512)
        # visual features --> embedding
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim

    def forward(self, img_features, hidden_state):
        #print("====================")
        #print(hidden_state.size())
        #print(img_features.size())
        # print(self.hidden_dim)
        # print(self.encoder_dim)
        
        

        U_h = self.U(hidden_state) #.unsqueeze(1)
        W_s = self.W(img_features)
        # print(U_h.size())
        # print(W_s.size())
        
        att = self.tanh(W_s + U_h)
        # print(att.size())
        # print("====================")
        #exit()
        e = self.v(att).squeeze(2)
        patch_weights = self.softmax(e)
        selected_visual_features = (img_features * patch_weights.unsqueeze(2)).sum(1)
        return selected_visual_features, patch_weights

if __name__ == "__main__":
    attention_model = Attention(encoder_dim=128, hidden_dim=512)
    summary(attention_model, input_size=((1,196,128),(1,1,512)))
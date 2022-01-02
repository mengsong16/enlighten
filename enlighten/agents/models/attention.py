import torch
import torch.nn as nn

from torchinfo import summary

# img_features: [batch_n,49,visual_input_size]
# hidden_states: [batch_n,1,hidden_size]
# selected_visual_features: [batch_n, visual_input_size]
# patch_weights: [batch_n, 49]
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

    def forward(self, img_features, hidden_states):
        #print("====================")
        #print(hidden_state.size())
        #print(img_features.size())
        # print(self.hidden_dim)
        # print(self.encoder_dim)
        
        U_h = self.U(hidden_states) #.unsqueeze(1)
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
    #attention_model = Attention(encoder_dim=128, hidden_dim=512)
    #summary(attention_model, input_size=((1,196,128),(1,1,512)))

    attention_model = Attention(encoder_dim=256, hidden_dim=512)
    batch_n = 5
    img_features = torch.rand((batch_n,49,256))
    hidden_states = torch.rand((batch_n,1,512))
    selected_visual_features, patch_weights = attention_model(img_features=img_features, hidden_states=hidden_states)

    print("img_features: %s"%str(img_features.size()))
    print("hidden_states: %s"%str(hidden_states.size()))
    print("selected_visual_features: %s"%str(selected_visual_features.size()))
    print("patch_weights: %s"%(str(patch_weights.size())))
import torch
import torch.nn as nn

from torchinfo import summary
import numpy as np

# img_features: [batch_n,49,visual_input_size]
# hidden_states: [batch_n,1,hidden_size]
# selected_visual_features: [batch_n, visual_input_size]
# patch_weights: [batch_n, 49]
class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, output_dim, attention_type):
        super(Attention, self).__init__()

        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.fc_output_dim = output_dim

        self.attention_type = attention_type
        if self.attention_type == "fc":
            self.output_size = self.fc_output_dim
        elif self.attention_type == "caption":    
            self.output_size = self.encoder_dim
        else:
            print("Error (attention.py): undefined attention type: %s"%(self.attention_type))  
            exit()  
        # caption
        # hidden states --> embedding
        self.U = nn.Linear(hidden_dim, 512)
        # visual features --> embedding
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)
           
        # fc   
        self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(encoder_dim*49), self.fc_output_dim
                    ),
                    nn.ReLU(True),
            )

    # caption
    def forward(self, img_features, hidden_states):
        if self.attention_type == "caption":
            U_h = self.U(hidden_states) 
            W_s = self.W(img_features)
            
            att = self.tanh(W_s + U_h)
        
            e = self.v(att).squeeze(2)
            patch_weights = self.softmax(e)
            selected_visual_features = (img_features * patch_weights.unsqueeze(2)).sum(1)
            #selected_visual_features = img_features * patch_weights.unsqueeze(2)
            #selected_visual_features = self.fc(selected_visual_features)
            #print("---- caption -----")
        elif self.attention_type == "fc":
            batch_size, patch_number, _ = img_features.size() 
            patch_weights = torch.zeros(batch_size, patch_number, device=img_features.device)  
            selected_visual_features = self.fc(img_features)
            #print("---- fc -----")
        else:    
            print("Error (attention.py): undefined attention model: %s"%(self.attention_type))
            exit()
        
        return selected_visual_features, patch_weights

    
    # average all patch features
    # def forward(self, img_features, hidden_states):
    #     batch_size, patch_number, _ = img_features.size()
    #     patch_weights = torch.ones(batch_size, patch_number) * (1.0/patch_number)
    #     patch_weights = patch_weights.to(img_features.device)
    #     selected_visual_features = (img_features * patch_weights.unsqueeze(2)).sum(1)
    #     return selected_visual_features, patch_weights 

    # average randomly selected patch features
    # def forward(self, img_features, hidden_states):
    #     batch_size, patch_number, _ = img_features.size()
    #     patch_weights = torch.randint(low=0, high=2, size=(batch_size, patch_number))
    #     #print(patch_weights)
    #     #print(patch_weights.size())
    #     total_non_zero = torch.sum(patch_weights, dim=1)
    #     total_non_zero = torch.reshape(total_non_zero, (batch_size, -1))
    #     #print(total_non_zero)
    #     #print(total_non_zero.size())
    #     total_non_zero = total_non_zero.repeat(1, patch_number)
    #     #print(total_non_zero)
    #     #print(total_non_zero.size())
    #     patch_weights = torch.div(patch_weights, total_non_zero)
        
    #     patch_weights = patch_weights.to(img_features.device)
    #     selected_visual_features = (img_features * patch_weights.unsqueeze(2)).sum(1)
    #     return selected_visual_features, patch_weights 
        


if __name__ == "__main__":
    #attention_model = Attention(encoder_dim=128, hidden_dim=512)
    #summary(attention_model, input_size=((1,196,128),(1,1,512)))

    attention_model = Attention(encoder_dim=256, hidden_dim=512, output_dim=512, attention_type="caption")
    batch_n = 5
    img_features = torch.rand((batch_n,49,256))
    hidden_states = torch.rand((batch_n,1,512))
    selected_visual_features, patch_weights = attention_model(img_features=img_features, hidden_states=hidden_states)

    print("img_features: %s"%str(img_features.size()))
    print("hidden_states: %s"%str(hidden_states.size()))
    print("selected_visual_features: %s"%str(selected_visual_features.size()))
    print("patch_weights: %s"%(str(patch_weights.size())))
from statistics import mode
from tkinter import N
from matplotlib.pyplot import axis
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU

import joblib

from src.models.get_model import get_model

def drop_path(x,drop_prob:float = 0.,training:bool = False ):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape  = (x.shape[0],) +(1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape,dtype=x.dtype,device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob)
    return output

class DropPath(nn.Module):
    def __init__(self,drop_prob=None):
        super(DropPath,self).__init__()
        self.drop_prob = drop_prob
    
    def forward (self,x):
        return drop_path(x,self.drop_prob,self.training)
        

class MLP_mu(nn.Module):
    def __init__(self):
        super(MLP_mu,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,256)
        )
    
    def forward(self,x):
        x = self.layers(x)
        return x

class MLP_sigma(nn.Module):
    def __init__(self):
        super(MLP_sigma,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(12,128),
            nn.ReLU(),
            nn.Linear(128,256)
        )
    
    def forward(self,x):
        x = self.layers(x)
        return x

class ExpressionEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256, n_vertices: int = 6890, mean: torch.Tensor = None, stddev: torch.Tensor = None,
                 model_name: str = 'expression_encoder'):
        """
        :param latent_dim: size of the latent expression embedding before quantization through Gumbel softmax
        :param n_vertices: number of face mesh vertices
        :param mean: mean position of each vertex
        :param stddev: standard deviation of each vertex position
        :param model_name: name of the model, used to load and save the model
        """
        super(ExpressionEncoder,self).__init__()
        self.n_vertices = n_vertices
        shape = (1, 1, n_vertices, 3)
        self.register_buffer("mean", torch.zeros(shape) if mean is None else mean.view(shape))
        self.register_buffer("stddev", torch.ones(shape) if stddev is None else stddev.view(shape))

        self.layers = nn.ModuleList([
            nn.Linear(self.n_vertices * 3, 256),
            nn.Linear(256, 128),
        ])
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.code = nn.Linear(128, latent_dim)

    def forward(self, geom):
        """
        :param geom: B x T x n_vertices x 3 Tensor containing face geometries
        :return: code: B x T x heads x classes Tensor containing a latent expression code/embedding
        """
        x = (geom- self.mean) / self.stddev
        x = x.contiguous().view(x.shape[0], x.shape[1], self.n_vertices*3).to(torch.float32)

        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
        x, _ = self.lstm(x)
        x = self.code(x)

        return {"code": x}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000,n_layers=1,hidden_size=256,bidirectional=False):
        super(PositionalEncoding, self).__init__()
        self.gru = nn.GRU(
            input_size = 256,
            hidden_size = hidden_size,
            bidirectional = bidirectional,
            num_layers = n_layers
        )

        self.dropout = nn.Dropout(p=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 256)
        )
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        y,_ = self.gru(x)
        y = y + x
        y = self.mlp(y)
        return self.dropout(y)


# only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)
    

class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,drop_path=0.,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()
        
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        
        self.input_feats = self.njoints*self.nfeats

        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.5)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.expression_encoder = ExpressionEncoder()
       

        if self.ablation == "average_encoder":
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.sigma_layer = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
            self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        # self.expression_code = self.getMeshList()
       
             
    def getMeshList(self):
        meshList = joblib.load('/data1/zxy/ACTOR/data/HumanAct12Poses/mesh.pkl')
        expression_code =mesh = self.expression_encoder(torch.from_numpy(meshList[1]))["code"][:,:90,:]
        for i in range(1,12):
            mesh = self.expression_encoder(torch.from_numpy(meshList[i]))["code"][:,:90,:]
            expression_code  = torch.concat((expression_code,mesh),dim=0)
            # expression_code.append(self.expression_encoder(torch.from_numpy(meshList[i]))["code"][0].detach().numpy().tolist())
        
        # expression_code = np.array(expression_code).astype(np.float64)
        return expression_code.cuda()                                            
    # def get_model_1(slef):
    #     torch.manual_seed(771)
    #     return torch.nn.Sequential(
    #         torch.nn.Linear(256,50),torch.nn.Tanh(),torch.nn.Linear(50,20)
    # )
    

    def forward(self, batch):
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        
        # embedding of the skeleton
        x = self.skelEmbedding(x)

        # only for ablation / not used in the final model
        if self.ablation == "average_encoder":
            # add positional encoding
            x = self.sequence_pos_encoder(x)
            
            # transformer layers
            final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
            # get the average of the output
            z = final.mean(axis=0)
            
            # extract mu and logvar
            mu = self.mu_layer(z)
            logvar = self.sigma_layer(z)
        else:
        #     # adding the mu and sigma queries

        #     # label_embed = nn.Embedding(12,512)
            # m_s = y.cpu()
        #     # mu_lable = label_embed(m_s)[:,:256][None]
        #     # sigma_label   = label_embed(m_s)[:,256:][None]
        #     # mu_lable = mu_lable.cuda()
        #     # sigma_label = sigma_label.cuda()
        #     # # mu_label,sigma_label: [1, 20, 256]   x:[60, 20, 256]
        #     # xseq = torch.cat((mu_lable,sigma_label , x), axis=0)
            
        #     # # xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)
        #     # # add positional encoding

        #     # ######################################3
        #     #
        #     # MLP _ zxy_mu_label
        #     # #########
        #     mlp_mu = MLP_mu()
        #     m_s = y.cpu()
        #     mlp_lable = F.one_hot(m_s,num_classes=12)
        #     mlp_lable = mlp_lable.float()
        #     mlp_lable = mlp_lable + torch.rand(mlp_lable.size())
        #     mu_lable = mlp_mu(mlp_lable)[None].cuda()
        #     # mu_lable = self.sigmoid(mu_lable)[None].cuda()
        #     # mu_lable = self.drop_path(mu_lable)
        #     # # # ########
        #     # # #
        #     # # # MLP _ zxy_sigma_label
        #     # # # #########
        #     # mlp_sigma = MLP_sigma()
        #     # mlp_lable = mlp_lable + torch.rand(mlp_lable.size())
        #     # sigma_label = mlp_sigma(mlp_lable)[None].cuda()
        #     # sigma_label = self.sigmoid(sigma_label)[None].cuda()
        #     # sigma_label = self.drop_path(sigma_label)
            expression_code = self.getMeshList()
            m_s = y.cpu()
            geom = []
            for label in m_s:
                expression =expression_code[label][None]
                geom.append(expression)
            
            a0 = geom[0]
            
            if len(m_s) == 20:
                a0 = geom[0]
                a1 = geom[1]
                a2 = geom[2]
                a3 = geom[3]
                a4 = geom[4]
                a5 = geom[5]
                a6 = geom[6]
                a7 = geom[7]
                a8 = geom[8]
                a9 = geom[9]
                a10 = geom[10]
                a11 = geom[11]
                a12 = geom[12]
                a13 = geom[13]
                a14 = geom[14]
                a15 = geom[15]
                a16 = geom[16]
                a17 = geom[17]
                a18 = geom[18]
                a19 = geom[19]
                mesh_lable = torch.cat((a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19),axis=0)
            else:
                mesh_lable = geom[0]
                for i in range(1,len(m_s)):
                    mesh_lable = torch.cat((mesh_lable,geom[i]),axis=0)
            mesh_lable = mesh_lable.permute([1,0,2])


            xseq = torch.cat((mesh_lable , x), axis=0)
            contrastive_mesh  = xseq
            contrastive_x = x
            xseq = self.sequence_pos_encoder(xseq)

            # create a bigger mask, to allow attend to mu and sigma
            muandsigmaMask = torch.ones((bs, 90), dtype=bool, device=x.device)
            maskseq = torch.cat((muandsigmaMask, mask), axis=1)

            final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
            mu = final[0]
            logvar = final[1]
        #     # from laplace import Laplace
        #     # model  = self.get_model_1()
        #     # la = Laplace(model, 'regression', subset_of_weights='all', hessian_structure='full')
        #     # mu,logvar = la(xseq, src_key_padding_mask=~maskseq)
        #     # print("11")

            
        return {"mu": mu, "logvar": logvar,"contrastive_mesh":contrastive_mesh,"contrastive_x":contrastive_x}
        # else:
        #     # adding the mu and sigma queries
        #     xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)

        #     # add positional encoding
        #     xseq = self.sequence_pos_encoder(xseq)

        #     # create a bigger mask, to allow attend to mu and sigma
        #     muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
        #     maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        #     final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        #     mu = final[0]
        #     logvar = final[1]
            
        # return {"mu": mu, "logvar": logvar}

# class Decoder_TRANSFORMER_gt(nn.Module):
#     def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
#                  latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
#                  ablation=None, **kargs):
#         super().__init__()

#         self.modeltype = modeltype
#         self.njoints = njoints
#         self.nfeats = nfeats
#         self.num_frames = num_frames
#         self.num_classes = num_classes
        
#         self.pose_rep = pose_rep
#         self.glob = glob
#         self.glob_rot = glob_rot
#         self.translation = translation
        
#         self.latent_dim = latent_dim
        
#         self.ff_size = ff_size
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.dropout = dropout

#         self.ablation = ablation

#         self.activation = activation
                
#         self.input_feats = self.njoints*self.nfeats

#         # self.mlp = nn.Sequential(
#         #     nn.Linear(256, 512),
#         #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         #     nn.Linear(512, 512),
#         #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         #     nn.Linear(512, 256)
#         # )

#         # only for ablation / not used in the final model
#         if self.ablation == "zandtime":
#             self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
#         else:
#             self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

#         # only for ablation / not used in the final model
#         if self.ablation == "time_encoding":
#             self.sequence_pos_encoder = TimeEncoding(self.dropout)
#         else:
#             self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        

#         # 编码 
#         seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
#                                                           nhead=self.num_heads,
#                                                           dim_feedforward=self.ff_size,
#                                                           dropout=self.dropout,
#                                                           activation=activation)
#         # 解码
#         self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
#                                                      num_layers=self.num_layers)
        
#         self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
    
#     def tensor_fill(self, tensor_size, val=0):
#         return torch.zeros(tensor_size).fill_(val).requires_grad_(False)

#     def forward(self, batch):
#         z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]

#         latent_dim = z.shape[1]
#         bs, nframes = mask.shape
#         njoints, nfeats = self.njoints, self.nfeats


        
#         prior_vec = self.tensor_fill((1,z.shape[0], z.shape[1]), 0).to(z.device)
       



#         # only for ablation / not used in the final model
#         if self.ablation == "zandtime":
#             yoh = F.one_hot(y, self.num_classes)
#             z = torch.cat((z, yoh), axis=1)
#             z = self.ztimelinear(z)
#             z = z[None]  # sequence of size 1
#         else:
#             # only for ablation / not used in the final model
#             if self.ablation == "concat_bias":
#                 # sequence of size 2
#                 z = torch.stack((z, self.actionBiases[y]), axis=0)
#             else:
#                 # shift the latent noise vector to be the action noise   [20, 256]
#                 z = z + self.actionBiases[y]
#                 z = z[None]  # sequence of size 1

                

#         # 生成60帧
#         mask_60 = mask[:,0:60]
#         timequerie_60 =  torch.zeros(60, bs, latent_dim, device=z.device)
#         timequerie_60 = self.sequence_pos_encoder(timequerie_60)
#         output_60 = self.seqTransDecoder(tgt=timequerie_60, memory=z,
#                                             tgt_key_padding_mask=~mask_60)
#         prior_vec = output_60
#         output = output_60

#         z_prior =  torch.concat((prior_vec,z),dim=0)
#         timequerie_last60 =  torch.zeros(60, bs, latent_dim, device=z.device)
#         timequerie_last60 = self.sequence_pos_encoder(timequerie_last60)

#         z_prior = self.mlp(z_prior)
#         output_last60 = self.seqTransDecoder(tgt=timequerie_last60, memory=z_prior,
#                                             tgt_key_padding_mask=~mask_60)
#         output = torch.concat((output_60,output_last60),dim=0)
#         # z_prior  = z + prior_vec


#         # for i in range(15,nframes):
#         #     z_prior  = z + prior_vec
#         #     #  单帧生成 test
#         #     timequerie_one =  torch.zeros(1, bs, latent_dim, device=z.device)
#         #     mask_one = mask[:,i:i+1]

#         #     timequerie_one = self.sequence_pos_encoder(timequerie_one)
#         #     output_one = self.seqTransDecoder(tgt=timequerie_one, memory=z_prior,
#         #                                     tgt_key_padding_mask=~mask_one)
            
#         #     output = torch.concat((output,output_one),dim=0)
#         #     prior_vec = output[i-14:i+1]

#         output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)

#         output[~mask.T] = 0
#         output = output.permute(1, 2, 3, 0)
        
#         batch["output"] = output
#         return batch

class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation
                
        self.input_feats = self.njoints*self.nfeats
        
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 256)
        )
        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        else:
            self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        

        # 编码 
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        # 解码
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
    
    

    def tensor_fill(self, tensor_size, val=0):
        return torch.zeros(tensor_size).fill_(val).requires_grad_(False)

    def forward(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            yoh = F.one_hot(y, self.num_classes)
            z = torch.cat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]  # sequence of size 1
        else:
            # only for ablation / not used in the final model
            if self.ablation == "concat_bias":
                # sequence of size 2
                z = torch.stack((z, self.actionBiases[y]), axis=0)
            else:
                # shift the latent noise vector to be the action noise
                z = z + self.actionBiases[y]
                z = z[None]  # sequence of size 1
            
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        
        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries)
        
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)
        
        batch["output"] = output
        return batch
    # def forward(self, batch):
    #     z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]
    #     z = z + self.actionBiases[y]

    #     joblib.dump(z,"/data1/zxy/feature/{}.pkl".format(y[0].item()))

    #     latent_dim = z.shape[1]
    #     bs, nframes = mask.shape
    #     njoints, nfeats = self.njoints, self.nfeats


        
    #     prior_vec = self.tensor_fill((1,z.shape[0], z.shape[1]), 0).to(z.device)
                
            
    #     timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)

    #     num_stops = nframes
    #     end_factor = np.linspace(0., 1., num=num_stops)


    #     # create a bigger mask, to allow attend to mu and sigma
    #     muandsigmaMask = torch.ones((bs, 90), dtype=bool, device=x.device)
    #     maskseq = torch.cat((muandsigmaMask, mask), axis=1)

    #     final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)


    #     start_factor = 1. - end_factor
    #     interp_features = [(start_factor[i]*self.actionBiases[1]) + (end_factor[i]*self.actionBiases[6]) for i in range(num_stops)]
    #     interp_features = torch.stack(interp_features)


    #     y= torch.tensor(8, dtype=int)[None]
        
    #     video = []
    #     for i in range(0,nframes):
    #             y = interp_features[i][None]
    #             z = z + y
    #             timequerie_one =  torch.zeros(1, bs, latent_dim, device=z.device)
    #             # if i == 0:
    #             #         mask_one = mask[0,0:1][None]
    #             # else:
    #             #         mask_one =  mask[:,i-1: i]
    #             mask_one = mask[0,i:i+1][None]
    #             timequerie_one = self.sequence_pos_encoder(timequerie_one)
    #             output_one = self.seqTransDecoder(tgt=timequerie_one, memory=z[None],
    #                                             tgt_key_padding_mask=~mask_one)
    #             video.append(output_one[0])
    #     output = torch.stack(video,axis=0 )



    #     # timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
    #     # timequeries = self.sequence_pos_encoder(timequeries)
    #     # # timequerie_60 =  torch.zeros(60, bs, latent_dim, device=z.device)
    #     # # timequerie_60 = self.sequence_pos_encoder(timequerie_60)
    #     # output = self.seqTransDecoder(tgt=timequeries, memory=interp_features[None].permute(1,0,2),
    #     #                               tgt_key_padding_mask=~mask)
        
    #     output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
    #     # # MIX
    #     # y_1= torch.tensor(3, dtype=int)[None]
    #     # y_2= torch.tensor(5, dtype=int)[None]
    #     # z = z + self.actionBiases[y_1]*1.05 + self.actionBiases[y_2]*1

    #     # # 
    #     # # z = z + self.actionBiases[y_1]
    #     # timequerie_60 =  torch.zeros(60, bs, latent_dim, device=z.device)
    #     # timequerie_60 = self.sequence_pos_encoder(timequerie_60)
    #     # output = self.seqTransDecoder(tgt=timequerie_60, memory=z[None],
    #     #                               tgt_key_padding_mask=~mask)
    
    #     # output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
    #     # a = 1

    #     # # only for ablation / not used in the final model
    #     # if self.ablation == "time_encoding":
    #     #     timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
    #     # else:
    #     #     timequeries = self.sequence_pos_encoder(timequeries)
        
    #     # output = self.seqTransDecoder(tgt=timequeries, memory=z,
    #     #                               tgt_key_padding_mask=~mask)
        
    #     # output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
    #     # zero for padded area
    #     output[~mask.T] = 0
    #     output = output.permute(1, 2, 3, 0)
        
    #     batch["output"] = output
    #     return batch

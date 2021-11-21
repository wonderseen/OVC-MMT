# Implement this VSE_Imagine_Enc Module for Encode Imagine Module, 
# which will notonly return vse_loss but also the weighted encode hidden states
from re import T
import torch
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F
from multi_head import MultiHeadAttention

from ..utils.utils import l2norm

class ImagineAttn(nn.Module):
    def __init__(
            self,
            method,
            context_size,
            shared_embedding_size
        ):
        super(ImagineAttn, self).__init__()
        self.method = method
        self.embedding_size = shared_embedding_size
        self.context_size = context_size
        self.mid_dim = self.context_size

        self.ctx2ctx = nn.Linear(self.context_size, self.context_size, bias=False)
        self.emb2ctx = nn.Linear(self.embedding_size, self.context_size, bias=False)
        
        if self.method == 'mlp':
            self.mlp = nn.Linear(self.mid_dim, 1, bias=False)
            self.score = self.score_mlp
        if self.method == 'dot':
            self.score = self.score_dot
        '''
        if self.method == 'dot':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        '''

    def forward(
            self,
            image_vec,
            decoder_hidden,
            ctx_mask=None
        ):
        '''
        Input:
            image_vec: A normalized image vector at the shared space. Size(B,E), E is the shared_embedding_size
            decoer_hidden: A normalized embedded vector from the decoder hidden state at shared space. Size(T,B,E)
            context_mask: The mask applied to filter out the hidden states that don't contribute. The size is (T,B)
        Output:
            attention_weights: The vector that is used to compute the attention weighted sum of decoder hidden state.
            The size should be (B,T) 
        '''

        # --- Create variable to store attention energies
        attn_energies = self.score(image_vec.unsqueeze(1), decoder_hidden.transpose(0, 1))
        if ctx_mask is not None:
            self.mask = (1 - ctx_mask.transpose(0, 1).data).byte().unsqueeze(1) # Convert the mask to the size(B*1*T)
            attn_energies.data.masked_fill_(self.mask.type(torch.bool), -float('inf'))

        # --- Normalize energies to weights in range 0 to 1, resize to B x 1 x T
        return F.softmax(attn_energies, dim=-1)

    def score_dot(
            self,
            image_vec,
            decoder_hidden
        ):
        """
        Input:
            image_vec: A normalized image vector at the shared space. Size(B,1,E), E is the shared_embedding_size
            decoer_hidden: A normalized embedded vector from the decoder hidden state at shared space. Size(B,T,C)
        Output:
            attention_weights: The vector that is used to compute the attention weighted sum of decoder hidden state.
            The size should be (B,1,T) 
        """
        ctx_ = self.ctx2ctx(decoder_hidden).permute(0, 2, 1) #  B*T*C -> B*C*T
        im_ = self.emb2ctx(image_vec) # B*1*C

        # --- Apply the l2norm to ctx and im before comutingt the energies
        # ctx_ = l2norm(ctx_,dim=1)
        # im_ = l2norm(im_,dim=2)

        energies = torch.bmm(im_, ctx_) 
        return energies

    def score_mlp(self, image_vec, decoder_hidden):
        """
        Input:
            image_vec: A normalized image vector at the shared space. Size(B,1,E), E is the shared_embedding_size
            decoer_hidden: A normalized embedded vector from the decoder hidden state at shared space. Size(B,T,C)
        Output:
            attention_weights: The vector that is used to compute the attention weighted sum of decoder hidden state.
            The size should be (B,1,T) 
        """
        ctx_ = self.ctx2ctx(decoder_hidden) # B*T*C
        im_ = self.emb2ctx(image_vec) # B*1*C

        energies = self.mlp(F.tanh(ctx_ + im_)).permute(0, 2, 1) # B*1*T
        return energies


class ImagineAttn_bta(nn.Module):
    def __init__(
            self,
            method,
            context_size: int,
            shared_embedding_size: int,
            im_size: int,
            n_head: int = 1,
            dropout: float = 0.1
        ):
        super(ImagineAttn_bta, self).__init__()
        self.method = method
        self.shared_embedding_size = shared_embedding_size
        self.context_size = context_size
        self.im_size = im_size

        self.bta_attn = MultiHeadAttention(
            d_model=self.shared_embedding_size,
            d_q=self.context_size,
            d_k=self.im_size,
            h=n_head,
            transform_K=True,
            transform_Q=True,
            transform_V=True
        )
        
    def forward(self, object_vecs, decoder_hidden, ctx_mask=None, key_mask=None):
        '''
        Input:
            image_vec: A normalized image vector at the shared space. Size(B,E), E is the shared_embedding_size
            decoer_hidden: A normalized embedded vector from the decoder hidden state at shared space. Size(T,B,E)
            context_mask: The mask applied to filter out the hidden states that don't contribute. The size is (T,B)
        Output:
            attention_weights: The vector that is used to compute the attention weighted sum of decoder hidden state.
            The size should be (B,T) 
        '''

        # --- Create variable to store attention energies
        unnorm_attned_vecs, norm_attned_vecs, attn_pool, att = self.bta_attn(decoder_hidden, object_vecs, ctx_mask=ctx_mask, key_mask=key_mask)

        return unnorm_attned_vecs, norm_attned_vecs, attn_pool, att


class VSE_Imagine_Enc(nn.Module):
    def __init__(
            self,
            attn_type,
            im_size,
            hidden_size,
            shared_embedding_size,
            dropout_im_emb = 0.0,
            dropout_txt_emb = 0.0, 
            activation_vse: bool = True
        ):
        super(VSE_Imagine_Enc, self).__init__()
        # --- Initialize the parameters
        self.attn_type = attn_type
        self.im_size = im_size
        self.hidden_size = hidden_size
        self.shared_embedding_size = shared_embedding_size
        self.dropout_im_emb = 0.0
        self.dropout_txt_emb = 0.0
        self.activation_vse = activation_vse

        # --- Initialize the layers
        self.imagine_attn = ImagineAttn(self.attn_type, self.hidden_size, self.shared_embedding_size)

        # --- initialize the image emebedding layer
        self.im_embedding = nn.Linear(self.im_size, self.shared_embedding_size)
        if self.dropout_im_emb > 0:
            self.im_embedding_dropout = nn.Dropout(self.dropout_im_emb)

        # --- initialize the text embedding layer
        self.text_embedding = nn.Linear(self.hidden_size, self.shared_embedding_size)
        if self.dropout_txt_emb > 0:
            self.txt_embedding_dropout = nn.Dropout(self.dropout_txt_emb)

    def forward(self, im_var, decoder_hiddens, criterion_vse=None, context_mask=None):
        """
            Learn the shared space and compute the VSE Loss
            Input:
                im_var: The image features with size (B, D_im)
                decoder_hiddens: The decoder hidden states for each time step of the decoder. Size is (T, B, H), H is the hidden size, T is the decoder_hiddens. 
                criterion_vse: The criterion to compute the loss.
            Output: 
                loss_vse: The loss computed for the visual-text shared space learning.
        """

        # --- Embed the image fetures to the shared space
        im_emb_vec = self.im_embedding(im_var)

        if self.activation_vse:
            im_emb_vec = F.tanh(im_emb_vec)
        
        if self.dropout_im_emb > 0:
            im_emb_vec = self.im_embedding_dropout(im_emb_vec)

        # --- Normalize the image embedding vectors
        im_emb_vec = l2norm(im_emb_vec)

        # --- Compute the weighted sum of attentions
        attn_weights = self.imagine_attn(im_emb_vec, decoder_hiddens, ctx_mask=context_mask)
        context_vec = attn_weights.bmm(decoder_hiddens.transpose(0, 1)).squeeze(1)

        text_emb_vec = self.text_embedding(context_vec)
        if self.activation_vse:
            text_emb_vec = F.tanh(text_emb_vec)

        if self.dropout_txt_emb > 0:
            text_emb_vec = self.txt_embedding_dropout(text_emb_vec)

        # --- Apply l2 norm to the text_emb_vec
        text_emb_vec = l2norm(text_emb_vec)

        # --- Compute the loss
        if criterion_vse is not None:
            loss_vse = criterion_vse(im_emb_vec, text_emb_vec) * 0.

        return loss_vse, context_vec
        
    def get_emb_vec(self, im_var, decoder_hiddens, ctx_mask=None):
        # --- Embed the image fetures to the shared space
        im_emb_vec = self.im_embedding(im_var)
        if self.activation_vse:
            im_emb_vec = F.tanh(im_emb_vec)

        # --- Normalize the image embedding vectors
        im_emb_vec = l2norm(im_emb_vec)

        # --- Compute the weighted sum of attentions
        attn_weights = self.imagine_attn(im_emb_vec, decoder_hiddens, ctx_mask=ctx_mask)
        context_vec = attn_weights.bmm(decoder_hiddens.transpose(0, 1)).squeeze(1)
        text_emb_vec = self.text_embedding(context_vec)
        if self.activation_vse:
            text_emb_vec = F.tanh(text_emb_vec)

        # --- Apply l2 norm to the text_emb_vec
        text_emb_vec = l2norm(text_emb_vec)
        return im_emb_vec, text_emb_vec

    def get_imagine_weights(self, im_var, decoder_hiddens, ctx_mask=None):
        # --- Embed the image fetures to the shared space
        im_emb_vec = self.im_embedding(im_var)
        if self.activation_vse:
            im_emb_vec = F.tanh(im_emb_vec)

        # --- Normalize the image embedding vectors
        im_emb_vec = l2norm(im_emb_vec)

        # --- Compute the weighted sum of attentions
        attn_weights = self.imagine_attn(im_emb_vec, decoder_hiddens, ctx_mask=ctx_mask)
        return attn_weights


class VSE_Imagine_Enc_bta(nn.Module):
    def __init__(
            self,
            attn_type,
            im_size,
            im_obj_size,
            hidden_size,
            shared_embedding_size,
            dropout_im_emb=0.0,
            dropout_txt_emb=0.0, 
            activation_vse = True
        ):
        super(VSE_Imagine_Enc_bta, self).__init__()
        # --- Initialize the parameters
        self.use_globale_feature = True
        self.attn_type = attn_type
        self.im_size = im_size
        self.im_obj_size = im_obj_size
        self.hidden_size = hidden_size
        self.shared_embedding_size = shared_embedding_size
        self.dropout_im_emb = 0.
        self.dropout_txt_emb = 0.
        self.activation_vse = activation_vse
        self.img_down_size = shared_embedding_size

        # --- Initialize the layers
        self.imagine_attn = ImagineAttn_bta(self.attn_type, self.hidden_size, self.shared_embedding_size, self.img_down_size, 1)
        # self.imagine_attn2 = ImagineAttn_bta(self.attn_type, self.img_down_size,self.shared_embedding_size, self.hidden_size,1)
        
        # --- initialize the image emebedding layer
        self.im_embedding = nn.Linear(self.im_size, self.img_down_size)
        if self.use_globale_feature:
            self.back_embedding = nn.Linear(self.im_size, self.img_down_size)
            # initialize the text embedding layer shared_embedding_size
            # self.text_embedding = nn.Linear(self.hidden_size + self.img_down_size, self.hidden_size)
            self.text_embedding = nn.Linear(self.hidden_size + self.img_down_size, self.shared_embedding_size)

        else:
            # self.text_embedding = nn.Linear(self.img_down_size + self.img_down_size, self.shared_embedding_size)
            # self.text_embedding = nn.Linear(self.img_down_size, self.shared_embedding_size)       
            self.text_embedding = nn.Linear(self.hidden_size, self.shared_embedding_size)

        self.gate = nn.Linear(self.shared_embedding_size, 1)
        # self.im_bn = nn.BatchNorm1d(self.shared_embedding_size, affine=True)
        # self.back_bn = nn.BatchNorm1d(self.shared_embedding_size, affine=True)
        # self.layernorm = nn.LayerNorm(self.im_size, eps=1e-05, elementwise_affine=True)
        # self.layernorm2 = nn.LayerNorm(self.img_down_size, eps=1e-05, elementwise_affine=True)

    def forward(self,
            im_var,
            im_bta_var,
            decoder_hiddens,
            source_vector=None,
            criterion_vse=None,
            context_mask=None,
            bta_mask=None,):
        """
            Learn the shared space and compute the VSE Loss
            Input:
                im_var: The image features with size (B, D_im)
                decoder_hiddens: The decoder hidden states for each time step of the decoder. Size is (T, B, H), H is the hidden size, T is the decoder_hiddens. 
                criterion_vse: The criterion to compute the loss.
            Output: 
                loss_vse: The loss computed for the visual-text shared space learning.
        """
        
        loss_vse, text_emb_vec, context_vec, _, _, attn_weights, im_bta_var = self.get_all(
            im_var,
            im_bta_var,
            decoder_hiddens,
            source_vector,
            bta_mask,
            criterion_vse,
            context_mask)
        return loss_vse, text_emb_vec, context_vec, im_bta_var, attn_weights

    def get_all(
            self,
            im_var,
            im_bta_var,
            decoder_hiddens,
            source_vector=None,
            bta_mask=None,
            criterion_vse=None,
            context_mask=None):
        """
            Learn the shared space and compute the VSE Loss
            Input:
                im_var: The image features with size (B, D_im)
                decoder_hiddens: The decoder hidden states for each time step of the decoder. Size is (T, B, H), H is the hidden size, T is the decoder_hiddens. 
                criterion_vse: The criterion to compute the loss.
            Output: 
                loss_vse: The loss computed for the visual-text shared space learning.
        """
        if bta_mask is not None:
            im_bta_var = im_bta_var[:, :, :2048]
        else:
            if im_bta_var.shape[-1] == 2050:
                bta_mask = im_bta_var[:, :, -1].to(torch.int)# im_bta_var.sum(axis=-1).to(torch.bool)
                im_bta_var = im_bta_var[:, :, :-2]
            else:
                bta_mask = im_bta_var.sum(axis=-1) != 0.

        im_bta_var = im_bta_var * bta_mask[:, :, None].float()
        

        # --- Embed the image fetures to the shared space
        im_emb_vec, im_var, im_bta_var = self.get_join_im_emb_vec(im_bta_var, im_var, bta_mask)

        # --- Compute the weighted sum of attentions
        # _, text_emb_vec_seq, attn_weights_pool, attn_weights = self.imagine_attn2(decoder_hiddens, im_bta_var, ctx_mask=context_mask)
        # context_vec = attn_weights.bmm(im_bta_var).squeeze(1)

        _, text_emb_vec_seq, attn_weights_pool, attn_weights = self.imagine_attn(
            im_bta_var,
            decoder_hiddens,
            ctx_mask=context_mask,
            key_mask=bta_mask)

        # text_emb_vec = text_emb_vec_seq * context_mask[:, :, None]
        # text_emb_vec = text_emb_vec.sum(1) / context_mask.sum(1).unsqueeze(1)
        
        text_emb_vec = attn_weights_pool.bmm(decoder_hiddens).squeeze(1) # B 1024
        if source_vector is not None:
            # context_vec = text_emb_vec
            # In our experiments, with or without the image whole representation is not 
            # important, as they have similar performances
            if self.use_globale_feature:
                context_vec = torch.cat((im_emb_vec, text_emb_vec), -1)
            else:
                context_vec = text_emb_vec
            context_vec = F.tanh(self.text_embedding(context_vec))
            
        else:
            # raise NotImplementedError('')
            context_vec = torch.cat((im_var, text_emb_vec), -1)
            context_vec = F.tanh(self.text_embedding(context_vec))

        ## Apply l2 norm to the text_emb_vec
        Norm_context_vec, Norm_im_emb_vec = self.get_norm_vec(context_vec), self.get_norm_vec(im_bta_var)

        # Compute the loss (duplicated for ovc)
        loss_vse = torch.tensor(0.)
        # if criterion_vse is not None:
        #     loss_vse = criterion_vse(Norm_im_emb_vec, Norm_context_vec)

        return loss_vse, text_emb_vec, context_vec, Norm_context_vec, \
            Norm_im_emb_vec, attn_weights, text_emb_vec_seq.transpose(0, 1)

    def get_norm_vec(self, vec):
        return l2norm(vec)

    def get_emb_vec(self, im_var, im_bta_var, decoder_hiddens, context_mask=None):
        _, _, _, Norm_im_emb_vec, Norm_context_vec, _, _ = self.get_all(
            im_var,
            im_bta_var,
            decoder_hiddens,
            context_mask=context_mask)

        return Norm_im_emb_vec, Norm_context_vec

    def get_imagine_weights(self, im_var, im_bta_var, decoder_hiddens, context_mask=None):
        _, _, _, _, _, attn_weights, _ = self.get_all(im_var, im_bta_var, decoder_hiddens, context_mask=context_mask)
        return attn_weights

    def get_join_im_emb_vec(self, im_bta_var, im_var, bta_mask=None):
        # im_bta_var = self.layernorm(im_bta_var)
        # im_bta_var = im_bta_var[:, :random.randint(-15, -1)]
        # im_bta_var = F.tanh(self.im_embedding(im_bta_var))
        # im_bta_var = self.im_embedding(im_bta_var)
        im_bta_var = F.tanh(self.im_embedding(im_bta_var)) 
        
        if self.use_globale_feature:
            im_var = self.back_embedding(im_var)
            # if im_var.shape[0] == 1:
            #     im_var = self.back_bn(im_var.repeat(2, 1))[0][None, :]
            # else: im_var = self.im_bn(im_var)
            
            im_bta_var_max = im_bta_var.max(dim=1)[0]
            # gate_emb = torch.sigmoid(self.gate(torch.cat((im_bta_var_max, im_var), -1)))
            # gate = gate_emb.squeeze(-1).unsqueeze(-1)
            # if im_bta_var.shape[0] == 1: im_bta_var_max = self.im_bn(im_bta_var_max.repeat(2, 1))[0][None, :]
            # else: im_bta_var_max = self.im_bn(im_bta_var_max)
            im_emb_vec = im_var
            im_emb_vec = F.tanh(im_emb_vec)
        else:
            gate = torch.sigmoid(self.gate(im_bta_var))
            gate = gate * bta_mask.float()[:, :, None]
            im_bta_var_geted = gate * im_bta_var
            # if bta_mask is not None:
            #     im_bta_var_geted += -1.0e9*(1.0-bta_mask.float())[:, :, None]
            # im_emb_vec = im_bta_var_geted.max(dim=1)[0]
            im_emb_vec = im_bta_var_geted.sum(dim=1)[0] / (gate.sum(dim=1)[:, None] + 1e-4)
        return im_emb_vec, im_var, im_bta_var

    def get_join_im_emb_vec2(self, im_bta_var, im_var):
        # if self.dropout_im_emb > 0:
        #     im_bta_var = im_bta_var[:, :random.randint(-10, -1)]
        # im_bta_var = im_bta_var[:, :random.randint(-10, -1)]

        im_emb_vec = F.tanh(self.back_embedding(torch.cat((im_bta_var[:, 0], im_var), -1)))
        return im_emb_vec, im_var, im_bta_var
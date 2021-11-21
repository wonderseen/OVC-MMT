import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from ..layers import LIUMCVC_Encoder2
from ..layers import NMT_Decoder
from ..layers import VSE_Imagine_Enc_bta

SOS_token = 2
EOS_token = 3
use_cuda = torch.cuda.is_available()


class NMT_AttentionImagine_Seq2Seq_Beam_V16(nn.Module):
    def __init__(self,
                 src_size:int, 
                 tgt_size:int, 
                 im_feats_size:int, 
                 im_obj_size:int, 
                 src_embedding_size:int, 
                 tgt_embedding_size:int, 
                 hidden_size:int, 
                 shared_embedding_size:int, 
                 loss_w:float = 0.0, 
                 beam_size:int = 1, 
                 attn_model = 'dot',
                 n_layers:int = 1, 
                 dropout_ctx:float =0.0, 
                 dropout_emb:float =0.0, 
                 dropout_out:float =0.0, 
                 dropout_rnn_enc:float = 0.0, 
                 dropout_rnn_dec:float = 0.0, 
                 dropout_im_emb:float = 0.0, 
                 dropout_txt_emb:float = 0.0, 
                 activation_vse:bool = True, 
                 tied_emb:bool =False,
                 init_split:float = 0.5):

        super(NMT_AttentionImagine_Seq2Seq_Beam_V16, self).__init__()

        # Define all the parameters
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.im_feats_size = im_feats_size
        self.im_obj_size = im_obj_size
        self.src_embedding_size = src_embedding_size
        self.tgt_embedding_size = tgt_embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.shared_embedding_size = self.hidden_size
        self.beam_size = beam_size
        self.loss_w = loss_w
        self.tied_emb = tied_emb
        self.dropout_im_emb = 0.
        self.dropout_txt_emb = 0.
        self.activation_vse = activation_vse
        self.attn_model = attn_model
        self.init_split = init_split

        ## Define all the parts. 
        # self.encoder = LIUMCVC_Encoder2(
        #     self.src_size,
        #     self.src_embedding_size,
        #     self.hidden_size // 2,
        #     self.n_layers,
        #     dropout_rnn=dropout_rnn_enc,
        #     dropout_ctx=dropout_ctx,
        #     dropout_emb=dropout_emb,
        #     pos_emb=False)

        self.encoder = LIUMCVC_Encoder2(
            self.src_size,
            self.src_embedding_size,
            self.hidden_size,
            self.n_layers,
            dropout_rnn=dropout_rnn_enc,
            dropout_ctx=dropout_ctx,
            dropout_emb=dropout_emb,
            pos_emb=False)

        # self.vse_imagine = VSE_Imagine_Enc_bta(
        #     self.attn_model,
        #     self.im_feats_size,
        #     self.im_obj_size,
        #     self.hidden_size,
        #     self.shared_embedding_size,
        #     self.dropout_im_emb,
        #     self.dropout_txt_emb,
        #     self.activation_vse)

        self.vse_imagine = VSE_Imagine_Enc_bta(
            self.attn_model,
            self.im_feats_size,
            self.im_obj_size,
            2 * self.hidden_size,
            self.shared_embedding_size,
            self.dropout_im_emb,
            self.dropout_txt_emb,
            self.activation_vse)

        self.text_only = False   
        self.image_level = False

        self.conf_thresh = 0.48
        self.only_mask_irrevant_object = False # corresponding to HM in Table 3 

        self.add_ranking_loss = True if not self.text_only else False
        self.max_ranking_rate = 1. - self.loss_w if self.add_ranking_loss else 0.
        self.ranking_rate = 1. - self.loss_w
        self.vi_tradeoff = 0.0 if not self.text_only else 0.

        self.type_decoder = 2 if not self.text_only else 1
        if self.type_decoder == 1:
            self.txt_down = nn.Linear(self.hidden_size, self.hidden_size)
            self.decoder = NMT_Decoder(
                self.tgt_size,
                self.tgt_embedding_size,
                self.hidden_size,
                2 * self.hidden_size,
                self.n_layers,
                dropout_rnn=dropout_rnn_dec,
                dropout_out=dropout_out,
                dropout_emb=0.0,
                tied_emb=tied_emb)
        else:
            self.txt_down = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.txt_gate = nn.Linear(2 * self.tgt_embedding_size + self.hidden_size, 1)
            # self.img_down = nn.Linear(self.hidden_size, m_hidden_size*self.hidden_size)

            self.decoder = NMT_Decoder(
                self.tgt_size,
                self.tgt_embedding_size,
                self.hidden_size,
                2 * self.hidden_size,
                self.n_layers,
                dropout_rnn=dropout_rnn_dec,
                dropout_out=dropout_out,
                dropout_emb=0.0,
                tied_emb=tied_emb)

        self.relu = nn.ReLU(inplace=True)

        ## Initilaize the layers with xavier method
        self.reset_parameters()
        self.true_false = True
        self.step = 0

    def reset_parameters(self):
        param_nums = 0
        print('=== all the parameter of Model ====')
        for name, param in self.named_parameters():
            param_num = 1
            print(name, end=' = ')
            for si in param.data.size():
                print(si, end=' * ')
                param_num *= si
            param_nums += param_num
            print('=', param_num)
            if param.requires_grad and 'bias' not in name and param.data.dim() > 1:
                nn.init.kaiming_normal_(param.data)
                # nn.init.xavier_uniform(param.data, gain=1./np.sqrt(param.size()[-1]))
        print('=========== parameter numbers = ', param_nums)
        
    def forward(
        self,
        src_var,
        src_lengths,
        tgt_var,
        vi_var,
        im_var,
        im_bta_var,
        teacher_force_ratio:float = 0.8,
        max_length:int = 80,
        criterion_mt=None,
        criterion_vse=None):
        '''
        Feed forward the input variable and compute the loss. tgt_var is always provided
        Input: 
            src_var: The minibatch input sentence indexes representation with size (B*W_s)
            src_lengths: The list of lenths of each sentence in the minimatch, the size is (B)
            im_var: The minibatch of the paired image ResNet Feature vecotrs, with the size(B*I), I is the image feature size.
            im_bta_var:
            teacher_force_ratio: A scalar between 0 and 1 which defines the probability ot conduct the teacher_force traning.
            tgt_var: The output sentence groundtruth, if provided it will be used to help guide the training of the network. The Size is (B*W_t)
                     If not, it will just generate a target sentence which is shorter thatn max_length or stop when it finds a EOS_Tag.
            max_length: A integer value that specifies the longest sentence that can be generated from this network.     
        Output:  
            loss: Total loss which is the sum of loss_mt and loss_vse
            loss_mt: The loss for seq2seq machine translation
            loss_vse: The loss for visual-text embedding space learning          
        '''
        ## Define the batch_size and input_length
        batch_size = src_var.size()[0]
        tgt_l = tgt_var.size()[1]
        l2_distance = 0
        tgt_mask = (tgt_var != 0).float()

        ## --- shrinke the vi weight
        # self.ranking_rate = 0.99 * self.ranking_rate if self.ranking_rate < self.max_ranking_rate else self.max_ranking_rate
        # self.vi_tradeoff = 0.99 * self.vi_tradeoff

        if self.image_level:
            im_bta_var = im_var.unsqueeze(1)

        ## Update the self.tgt_l
        self.tgt_l = tgt_l

        padding_mask = im_bta_var.sum(axis=-1) != 0.
        padding_mask = padding_mask.to(torch.float) 

        ## get object masking for training or inference
        # assert im_bta_var.shape[-1] == 2050, 'the shape of image feature vector should be 2050 when training'
        if im_bta_var.shape[-1] == 2050: # at training
            # bta_mask = im_bta_var[:,:,-1].to(torch.bool) # im_bta_var.sum(axis=-1).to(torch.bool)
            bta_mask = (im_bta_var[:, :, -1] >= self.conf_thresh).to(torch.bool)
        else: # at inference, all padding object features are preprocessed as all zeros, so we obtain the bta_mask by:
            bta_mask = im_bta_var.sum(axis=-1) != 0.

        # ## example: mask some irrelevant objects
        # bta_mask = bta_mask.to(torch.float)
        # bta_mask = bta_mask + padding_mask * (1. - bta_mask) * ( torch.rand(bta_mask.size()) > torch.rand(bta_mask.size())).cuda().to(torch.float)
        # bta_mask = bta_mask.to(torch.bool)

        # --- Encoder src_var
        encoder_outputs, context_mask, _ = self.encoder(src_var, src_lengths)
        source_vector = encoder_outputs.transpose(0, 1) * context_mask[:, :, None].transpose(0,1) 
        source_vector = source_vector.sum(1) / context_mask.transpose(0, 1).sum(1)[:, None]

        # --- Prepare the Input and Output Variables for Decoder
        loss_mt_mean, loss_vi_mean, (loss_mt, loss_vi, loss_vse) = self.image_transfer_and_decode(
            tgt_l, teacher_force_ratio,
            vi_var, tgt_var, source_vector,
            encoder_outputs, context_mask, im_bta_var,
            bta_mask, im_var, criterion_vse, criterion_mt)
        loss_lo = loss_mt

        # --- Accelerate the early training to obtain a acceptable MT model
        self.step += 1    
        if self.step < 4000: 
            return loss_mt_mean, loss_mt_mean, loss_vse

        # self.true_false = not self.true_false
        # if self.true_false and self.add_ranking_loss:
        #     return loss_mt_mean, loss_mt_mean, loss_vse

        ## --- masking part
        bta_mask = bta_mask.to(torch.float)

        if self.only_mask_irrevant_object:
            loss_lr = loss_lo
        else:
            ## --- mask relevant objects
            # setting 1: only mask one object
            translated_random_mask = padding_mask * bta_mask * (torch.rand(bta_mask.size())).cuda()
            translated_random_max  = torch.max(translated_random_mask, axis=-1)[0]
            translated_random_mask = (translated_random_mask == translated_random_max[:, None]).to(torch.float)
            lr_bta_mask = padding_mask - translated_random_mask

            # # setting 2: mask several objects
            # lr_bta_mask = padding_mask - padding_mask * bta_mask * ( torch.rand(bta_mask.size()) > torch.rand(bta_mask.size())).cuda().to(torch.float)
            # lr_bta_mask = lr_bta_mask.to(torch.bool)
            # print(lr_bta_mask)

            lr_mt_mean, lr_vi_mean, (lr_mt, lr_vi, _) = self.image_transfer_and_decode(
                                                                tgt_l, teacher_force_ratio,
                                                                vi_var, tgt_var, source_vector,
                                                                encoder_outputs, context_mask,
                                                                im_bta_var, lr_bta_mask,
                                                                im_var,
                                                                criterion_vse, criterion_mt)
            loss_lr = lr_mt

        ## --- mask irrelevant objects
        # setting 1: mask several objects
        # li_bta_mask = bta_mask + padding_mask * (1. - bta_mask) * ( torch.rand(bta_mask.size()) > torch.rand(bta_mask.size())).cuda().to(torch.float)
        # li_bta_mask = li_bta_mask.to(torch.bool)

        # setting 2: only mask an object
        translated_random_mask = padding_mask * (1. - bta_mask) * (torch.rand(bta_mask.size())).cuda()
        translated_random_max  = torch.max(translated_random_mask, axis=-1)[0]
        translated_random_mask = (translated_random_mask == translated_random_max[:, None]).to(torch.float)
        li_bta_mask = padding_mask - translated_random_mask

        li_mt_mean, li_vi_mean, (li_mt, li_vi, _) = self.image_transfer_and_decode(
                                                    tgt_l, teacher_force_ratio,
                                                    vi_var, tgt_var, source_vector,
                                                    encoder_outputs, context_mask,
                                                    im_bta_var, li_bta_mask, im_var,
                                                    criterion_vse, criterion_mt)
        loss_li = li_mt

        # --- compute objectives
        lio_diff = loss_li - loss_lo.detach()
        lro_diff = loss_lr - loss_lo.detach()

        lio_loss = self.relu(lio_diff * lio_diff - 0.1) # margin as 0.2
        lro_loss = self.relu(0.1 - lro_diff) # we expect lro_diff be positive but < 0.1
        loss_rk = lro_loss + lio_loss # [bsz, seq_len]

        loss = self.ranking_rate * loss_rk + loss_lo + self.vi_tradeoff * loss_vi
        # loss = self.ranking_rate * loss_rk + (1. - self.ranking_rate) * loss_lo
        # loss = self.ranking_rate * loss_rk + (loss_lu + loss_lc + loss_lo) / 3.0
        # loss = self.ranking_rate * loss_rk.mean() + (1. - self.ranking_rate) * (loss_lo.mean() + loss_lc.mean() ) / 3.0

        # --- token-level loss
        loss = (loss * tgt_mask).sum(-1) / tgt_mask.sum(-1).float()
        loss_mean = loss.mean()

        ## (optional) parameter weight regularization
        # weight_norm = torch.norm(self.vse_imagine.im_embedding.weight, p=2, dim=-1)
        # loss += torch.pow(1. - weight_norm, 2).mean()
        # l2_distance = 5. * (l2_distance / tgt_mask.sum(-1)).mean()
        # loss += l2_distance


        if random.uniform(0, 1) < 0.002:
            print(lro_loss[0])
            # print(lio_loss[0])

        return loss_mean, loss_mt_mean, loss_vse

    def image_transfer_and_decode(self,
            tgt_l,
            teacher_force_ratio,
            vi_var,
            tgt_var,
            source_vector,
            encoder_outputs,
            context_mask,
            im_bta_var,
            bta_mask, 
            im_var,
            criterion_vse, criterion_mt,
            reduce: bool = False
        ):
        batch_size = tgt_var.size()[0]
        loss_mt = []
        loss_vi = []
        loss_vse = torch.tensor(0.)

        if self.text_only:
            encoder_concat = self.txt_down(source_vector)
            vse_text_vec = encoder_concat
        else:
            encoder_outputs = encoder_outputs * context_mask[:, :, None]
            encoder_concat = encoder_outputs.transpose(0, 1).sum(axis=1) / context_mask.transpose(0, 1).sum(axis=1)[:, None]
            encoder_concat2 = self.txt_down(encoder_concat)
            loss_vse, encoder_concat1, vse_text_vec, im_bta_var, attn_weights = self.vse_imagine(
                    im_var,
                    im_bta_var,
                    encoder_outputs.transpose(0, 1),
                    source_vector,
                    bta_mask=bta_mask,
                    context_mask=context_mask.transpose(0, 1)
                )
                
            if not self.image_level:
                encoder_outputs = F.tanh(encoder_outputs)

            gate = F.sigmoid(self.txt_gate(torch.cat((encoder_concat2, vse_text_vec), axis=-1)))
            encoder_concat = gate * F.tanh(encoder_concat2) + (1.0 - gate) * F.tanh(vse_text_vec)

            # if self.type_decoder != 1:
            #     _, encoder_outputs_late, attn_weights = self.so_attention(im_bta_var.transpose(0, 1), encoder_outputs.transpose(0,1),
            #         ctx_mask=context_mask.transpose(0, 1), key_mask=bta_mask.transpose(0, 1))
            #     encoder_outputs = F.tanh(encoder_outputs + self.img_down(encoder_outputs_late.transpose(0, 1)))#encoder_outputs + F.tanh(encoder_outputs_late.transpose(0,1))

        ## Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for _ in range(batch_size)]))

        encoder_concat = encoder_concat.unsqueeze(0)

        decoder_hidden = (encoder_concat)

        ## Initialize the output
        if use_cuda: decoder_input = decoder_input.cuda()
        if tgt_var is not None: tgt_mask = (tgt_var != 0).float()

        ## Initialize Decoder Hidden With Weighted Sum from the interaction with Images
        # decoder_hidden = F.tanh(self.decoderini(encoder_concat)).unsqueeze(0)
        # decoder_hiddens = Variable(torch.zeros(tgt_l, batch_size, self.hidden_size))
        # if use_cuda: decoder_hiddens = decoder_hiddens.cuda()
        

        ## A decreasing schedule of vi tradeoff
        # self.vi_tradeoff = 0.9998 * self.vi_tradeoff
        # if self.vi_tradeoff <= 0.5: self.vi_tradeoff = 0.5

        idx = torch.linspace(0, decoder_input.shape[0], steps=decoder_input.shape[0]).long()
        tgt_mask = Variable(tgt_mask, requires_grad=False)

        is_teacher = random.random() < teacher_force_ratio
        if is_teacher: 
            for di in range(tgt_l):
                decoder_output, encoder_concat = self.decoder(
                    decoder_input,
                    encoder_concat,
                    encoder_outputs,
                    ctx_mask=context_mask)
                loss_n = criterion_mt(decoder_output, tgt_var[:, di])
                loss_mt += [loss_n * tgt_mask[:, di]]
                loss_vi += [loss_n * vi_var[:, di] * tgt_mask[:, di]]

                ## cross-entropy 
                # decoder_output = F.softmax(decoder_output, dim=-1)
                # decoder_output = decoder_output.scatter_add(
                #     1,
                #     tgt_var[:,di][:, None],
                #     0.1 * (1. - vi_var[:, di][:, None])
                # )
                # loss_n = criterion_mt(decoder_output, tgt_var[:, di])
                # loss_mt += [loss_n]
                # loss_vi += self.vi_tradeoff * loss_n * vi_var[:, di]

                ## cosine distance
                # _, top1 = decoder_output.data.topk(1)
                # predicted_embedding = self.decoder.embedding(top1).view(batch_size, -1)
                # raw_embedding = self.decoder.embedding(tgt_var[:, di]).view(batch_size, -1)
                # feature1 = predicted_embedding / predicted_embedding.norm(dim=-1)[:, None]
                # feature2 = raw_embedding / raw_embedding.norm(dim=-1)[:, None]
                # cosine = (1. - feature1.mm(feature2.t())).sum(-1)
                # loss_mt += [cosine]

                ## norm distance
                # _, top1 = decoder_output.data.topk(1)
                # predicted_embedding = self.decoder.embedding(top1).view(batch_size, -1)
                # raw_embedding = self.decoder.embedding(tgt_var[:, di]).view(batch_size, -1)
                # l2_distance = self.relu((predicted_embedding-raw_embedding).norm(dim=-1) - 0.01)
                # loss_mt += [l2_distance]

                # decoder_hiddens[di] = decoder_hidden
                # text_embedding_sets[di] = text_embedding_di
                
                decoder_input = tgt_var[:, di]
        else:
            for di in range(tgt_l):
                decoder_output, encoder_concat = self.decoder(
                                                decoder_input,
                                                encoder_concat,
                                                encoder_outputs,
                                                ctx_mask=context_mask)
                loss_n = criterion_mt(decoder_output, tgt_var[:, di])
                loss_mt += [loss_n * tgt_mask[:, di]]
                loss_vi += [loss_n * vi_var[:, di] * tgt_mask[:, di]]

                # decoder_output = F.softmax(decoder_output, dim=-1)
                # decoder_output = decoder_output.scatter_add(
                #     1,
                #     tgt_var[:,di][:, None],
                #     0.1*(1.-vi_var[:, di][:, None])
                # )
                # loss_n = criterion_mt(decoder_output, tgt_var[:, di])
                # loss_mt += [loss_n]
                # loss_vi += [self.vi_tradeoff*loss_n*vi_var[:, di]]

                ## Normalize The Text Embedding Vector
                # decoder_hiddens[di] = decoder_hidden
                # text_embedding_sets[di] = text_embedding_di


                _, top1 = decoder_output.data.topk(1)
                decoder_input = Variable(top1)
                if use_cuda:
                    decoder_input = decoder_input.cuda()
                

                ## cosine
                # _, top1 = decoder_output.data.topk(1)
                # predicted_embedding = self.decoder.embedding(top1).view(batch_size, -1)
                # raw_embedding = self.decoder.embedding(tgt_var[:, di]).view(batch_size, -1)
                # feature1 = predicted_embedding/predicted_embedding.norm(dim=-1)[:, None]
                # feature2 = raw_embedding/raw_embedding.norm(dim=-1)[:, None]
                # cosine = (1.-feature1.mm(feature2.t())).sum(-1)
                # loss_mt += [cosine]

                ## norm distance
                # _, top1 = decoder_output.data.topk(1)
                # predicted_embedding = self.decoder.embedding(top1).view(batch_size, -1)
                # raw_embedding = self.decoder.embedding(tgt_var[:, di]).view(batch_size, -1)
                # l2_distance = self.relu((predicted_embedding-raw_embedding).norm(dim=-1) - 0.01)
                # loss_mt += [l2_distance]

        for i, m in enumerate(loss_mt):
            loss_mt[i] = loss_mt[i][:, None]
            loss_vi[i] = loss_vi[i][:, None]
        loss_mt = torch.cat(loss_mt, dim=1) # bsz, seq_len
        loss_vi = torch.cat(loss_vi, dim=1)

        ## Average the machine translation loss
        loss_mt_mean = loss_mt.sum(-1) / tgt_mask.sum(-1)
        loss_vi_mean = loss_vi.sum(-1) / tgt_mask.sum(-1)
        return loss_mt_mean.mean(), loss_vi_mean.mean(), (loss_mt, loss_vi, loss_vse)

    def _validate_args(
            self,
            src_var,
            tgt_var,
            max_length
        ):
        batch_size = src_var.size()[0]
        if tgt_var is None:
            tgt_l = max_length
        else:
            tgt_l = tgt_var.size()[1]

        return batch_size, tgt_l

    def beamsearch_decode(
            self,
            src_var,
            src_lengths,
            im_var,
            im_bta_var,
            beam_size=1,
            max_length=80,
            tgt_var=None
        ):
        
        tgt_l = max_length
        if tgt_var is not None:
            tgt_l = tgt_var.size()[1]
        
        batch_size = src_var.size()[0]

        self.tgt_l = tgt_l
        self.final_sample = []
        self.beam_size = beam_size

        if self.image_level:
            im_bta_var = im_var.unsqueeze(1)

        im_bta_var = im_bta_var[:, :, :2048]
        bta_mask = im_bta_var.sum(axis=-1) != 0.

        ## Encode the Sentences. 
        encoder_outputs, context_mask, source_vector = self.encoder(src_var, src_lengths)

        if self.text_only:
            encoder_concat = self.txt_down(source_vector)
            vse_text_vec = encoder_concat
            attn_weights = None
        else:
            encoder_outputs = encoder_outputs * context_mask[:, :, None]
            encoder_concat = encoder_outputs.transpose(0, 1).sum(axis=1) / context_mask.transpose(0, 1).sum(axis=1)[:, None]
            encoder_concat2 = self.txt_down(encoder_concat)
            _, encoder_concat1, vse_text_vec, im_bta_var, attn_weights = self.vse_imagine(
                                                                            im_var,
                                                                            im_bta_var,
                                                                            encoder_outputs.transpose(0, 1),
                                                                            source_vector,
                                                                            bta_mask=bta_mask,
                                                                            context_mask=context_mask.transpose(0, 1))
            if not self.image_level:
                encoder_outputs = F.tanh(encoder_outputs) 

            gate = F.sigmoid(self.txt_gate(torch.cat((encoder_concat2, vse_text_vec), axis=-1)))
            encoder_concat = gate * F.tanh(encoder_concat2) + (1.0 - gate) * F.tanh(vse_text_vec)
            
            # if self.type_decoder != 1:
            #     _, encoder_outputs_late, attn_weights = self.so_attention(im_bta_var.transpose(0,1), encoder_outputs.transpose(0,1),
            #         ctx_mask=context_mask.transpose(0,1), key_mask=bta_mask)       

        ## Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))

        encoder_concat = encoder_concat.unsqueeze(0)

        decoder_hidden = (encoder_concat)

        if use_cuda:
            decoder_input = decoder_input.cuda()


        if beam_size == 1:
            decoder_translation_list = []
            for _ in range(tgt_l):
                decoder_output,encoder_concat = self.decoder(
                    decoder_input,
                    encoder_concat,
                    encoder_outputs,
                    ctx_mask=context_mask)

                _, top1 = decoder_output.data.topk(1)
                ## Append the current prediction to decoder_translation
                decoder_translation_list.append(top1[:,0])
                decoder_input = Variable(top1)
                if use_cuda:
                    decoder_input = decoder_input.cuda()
            
            ## Compute the translation_prediction
            for b in range(batch_size):
                current_list = []
                for i in range(tgt_l):
                    current_translation_token = decoder_translation_list[i][b]
                    if current_translation_token == EOS_token:
                        break
                    current_list.append(current_translation_token)
                self.final_sample.append(current_list)
        
        if beam_size > 1:
            self.final_sample = self.beamsearch(
                encoder_outputs,
                encoder_outputs,
                context_mask,
                decoder_input,
                decoder_hidden,
                beam_size,
                tgt_l)

        return self.final_sample, attn_weights

    def beamsearch(
            self,
            encoder_outputs,
            encoder_outputs_late,
            context_mask,
            decoder_input,
            decoder_hidden,
            beam_size,
            max_length,
            avoid_double=True,
            avoid_unk=False
        ):
        ## Batch_Size
        batch_size = encoder_outputs.size(1)
        n_vocab = self.tgt_size

        ## Define Mask to apply to pdxs.view(-1) to fix indices
        nk_mask = torch.arange(batch_size * beam_size).long() #[0:batch_size*beam_size]
        if use_cuda:
            nk_mask = nk_mask.cuda()
        pdxs_mask = (nk_mask / beam_size) * beam_size

        ## Tile indices to use in the loop to expand first dim
        tile = nk_mask / beam_size

        ## Define the beam
        beam = torch.zeros((max_length, batch_size, beam_size)).long()
        if use_cuda:
            beam = beam.cuda()

        ## Create encoder outptus, context_mask with batch_dimension = batch_size*beam_size
        encoder_outputs_di = encoder_outputs[:, tile, :]
        encoder_outputs_late_di = encoder_outputs_late[:, tile, :]
        context_mask_di = context_mask[:, tile] 

        ## Define a inf numbers to assign to 
        inf = -1e5

        (encoder_concat) = decoder_hidden

        for di in range(max_length):
            if di == 0:
                decoder_output, encoder_concat = self.decoder(
                    decoder_input,
                    encoder_concat,
                    encoder_outputs,
                    ctx_mask=context_mask
                )
                ## nll and topk have the shape [batch, topk]
                nll, topk = decoder_output.data.topk(k=beam_size, sorted=False) 
                beam[0] = topk
            else:
                cur_tokens = beam[di-1].view(-1) ## Get the input tokens to the next step
                fini_idxs = (cur_tokens == EOS_token).nonzero() ## The index that checks whether the beam has terminated
                n_fini = fini_idxs.numel() ## Verify if all the beams are terminated
                if n_fini == batch_size * beam_size:
                    break

                ## Get the decoder for the next iteration(batch_size*beam_size,1)
                with torch.no_grad():
                    decoder_input = Variable(cur_tokens)
                encoder_concat = encoder_concat[:, tile, :] ## This operation will create a decoder_hidden states with size [batch_size*beam_size,H]

                decoder_output, encoder_concat = self.decoder(
                    decoder_input,
                    encoder_concat,
                    encoder_outputs_di,
                    ctx_mask=context_mask_di)
                decoder_output = decoder_output.data

                ## Suppress probabilities of previous tokens at current time step, which avoids generating repeated word. 
                if avoid_double:
                    decoder_output.view(-1).index_fill_(0, cur_tokens + (nk_mask * n_vocab), inf)

                ## Suppress probabilities of unk word.
                if avoid_unk:
                    decoder_output[:, UNK_token] = inf

                """
                Favor finished hyps to generate <eos> again
                Their nll scores will not increase further and they will always be kept in the beam.
                This operation assures the future generation for those finished hypes will always pick EOS_token. 
                """
                if n_fini > 0:
                    fidxs = fini_idxs[:,0]
                    decoder_output.index_fill_(0, fidxs, inf)
                    decoder_output.view(-1).index_fill_(0, fidxs * self.tgt_size + EOS_token, 0)

                ## Update the current score
                nll = (nll.unsqueeze(2) + decoder_output.view(batch_size, -1, n_vocab)).view(batch_size, -1) #[batch, beam * n_vocab]

                ## Pick the top beam_size best scores
                nll, idxs = nll.topk(beam_size, sorted=False) # nll, idxs [batch_size, beam_size]

                ## previous indices into the beam and current token indices
                pdxs = idxs / n_vocab #[batch_size, beam_size]

                ## Update the previous token in beam[di]
                beam[di] = idxs % n_vocab

                ## Permute all hypothesis history according to new order
                beam[:di] = beam[:di].gather(2, pdxs.repeat(di, 1, 1))

                ## Compute correct previous indices
                ## Mask is nedded since we are in flatten regime
                tile = pdxs.view(-1) + pdxs_mask
        ## Put an explicit <eos> to ensure that every sentence end in the end
        beam[max_length - 1] = EOS_token

        ## Find lengths by summing tokens not in (pad, bos, eos)
        lens = (beam.transpose(0, 2) > 3).sum(-1).t().float().clamp(min=1)

        ## Normalize Scores by length
        nll /= lens.float()
        top_hyps = nll.topk(1, sorted=False)[1].squeeze(1)
        
        ## Get best hyp for each sample in the batch
        hyps = beam[:, range(batch_size), top_hyps].cpu().numpy().T

        final_sample = []
        
        for b in range(batch_size):
            current_list = []
            for i in range(max_length):
                current_translation_token = hyps[b][i]
                if current_translation_token == EOS_token:
                    break
                current_list.append(current_translation_token)
            final_sample.append(current_list)

        return final_sample

    ###########################Function Defined Below is for Image Retrieval##############
    def embed_sent_im_eval(
            self,
            src_var,
            src_lengths,
            tgt_var,
            im_feats,
            im_bta_feats
        ):
        """
            Embed the Target Sentences to the shared space
            Input: 
                source_sent: The Source Sent Index Variable with size(B,W), W is just the length of the sentence
                target_sent: The Target Sent Index Variable with size(B,W) W is the length of the sentence
                im_feats: The Image Features with size (B, D), D is the dimension of image feature.
            Output:
                txt_embedding.data: The embedded sentence tensor with size (B, SD), SD is the dimension of shared embedding
                space. 
                im_embedding.data: The embedded image tensor with size (B, SD), SD is the dimension of the shared embedding space
        """
        # Define the batch_size and input_length
        tgt_l = tgt_var.size()[1]
        self.tgt_l = tgt_l

        # Encoder src_var
        encoder_outputs, context_mask = self.encoder(src_var, src_lengths)

        im_embedding, text_embedding = self.vse_imagine.get_emb_vec(
            im_feats, 
            im_bta_var, 
            encoder_outputs.transpose(0, 1),
            ctx_mask=context_mask.transpose(0, 1)
        )
        return im_embedding.data, text_embedding.data

    def embed_sent_im_test(
            self,
            src_var,
            src_lengths,
            im_feats,
            bta_im_feats,
            max_length=80
        ):
        """
            Embed the Target Sentences to the shared space
            Input: 
                source_sent: The Source Sent Index Variable with size(B,W), W is just the length of the sentence
                target_sent: The Target Sent Index Variable with size(B,W) W is the length of the sentence
                im_feats: The Image Features with size (B,D), D is the dimension of image feature.
                bta_im_feats:
            Output:
                txt_embedding.data: The embedded sentence tensor with size (B, SD), SD is the dimension of shared embedding
                space. 
                im_embedding.data: The embedded image tensor with size (B, SD), SD is the dimension of the shared embedding space
        """
        tgt_l = max_length
        self.tgt_l = tgt_l

        ## Encoder src_var
        encoder_outputs, context_mask = self.encoder(src_var, src_lengths)

        ## Get the embedded vectors from vse_imagine
        im_embedding, text_embedding = self.vse_imagine.get_emb_vec(
            im_feats,
            bta_im_feats,
            encoder_outputs.transpose(0,1),
            context_mask=context_mask)
        
        ## another text_embedding may be here. 
        return im_embedding.data, text_embedding.data

    def get_imagine_attention_eval(
            self,
            src_var,
            src_lengths,
            tgt_var,
            im_feats,
            im_bta_var
        ):
        """
            Get the attention_weights for validation dataset when tgt_var is available.
            Input: 
                source_var: The Source Sent Index Variable with size(B,W), W is just the length of the sentence
                target_var: The Target Sent Index Variable with size(B,W) W is the length of the sentence
                im_feats: The Image Features with size (B,D), D is the dimension of image feature.
            Output:
                output_translation: List of index for translations predicted by the seq2seq model
                attention_weights: (B,T)
        """
        tgt_l = tgt_var.size()[1]
        self.tgt_l = tgt_l

        # Encoder src_var
        encoder_outputs, context_mask = self.encoder(src_var, src_lengths)

        # Get the attention weights
        attn_weights = self.vse_imagine.get_imagine_weights(
            im_feats,
            im_bta_var,
            encoder_outputs,
            context_mask=context_mask)

        return attn_weights.data

    def get_imagine_attention_test(
            self,
            src_var,
            src_lengths,
            im_feats,
            im_bta_feats,
            max_length=80
        ):
        """
            Get the attention_weights for validation dataset when tgt_var is available.
            Input: 
                source_var: The Source Sent Index Variable with size(B,W), W is just the length of the sentence
                target_var: The Target Sent Index Variable with size(B,W) W is the length of the sentence
                im_feats: The Image Features with size (B,D), D is the dimension of image feature.
            Output:
                output_translation: List of index for translations predicted by the seq2seq model
                attention_weights: (B,T)
        """
        # Define the batch_size and input_length
        tgt_l = max_length

        # Update the self.tgt_l
        self.tgt_l = tgt_l

        # Encoder src_var
        encoder_outputs, context_mask = self.encoder(src_var, src_lengths)

        # Get the attention weights
        attn_weights = self.vse_imagine.get_imagine_weights(
            im_feats,
            im_bta_feats,
            encoder_outputs,
            context_mask=context_mask)

        return attn_weights.data
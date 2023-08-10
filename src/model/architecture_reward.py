#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import time
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from . import modules
from torch import nn
from . import tf_decoder
# from .model_hyper import HyperGCN
# from one_peace.models import from_pretrained
from .rewards import get_rewards

# Yongsik Part
from eval_metric.coco_eval import calculate_eval_matric


# torch.set_printoptions(profile="full")

class VAE_Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE_Encoder, self).__init__()
        self.num_layer = 2

        self.lstm = nn.LSTM(x_dim, h_dim, num_layers=self.num_layer, batch_first=True, bidirectional=False)
        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # output layer
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x, (_,_) = self.lstm(x)
        x = self.fc1(x)

        mu = F.relu(self.mu(x))
        logvar = F.relu(self.logvar(x))

        z = reparameterization(mu, logvar)
        return z, mu, logvar
    
    
def reparameterization(mu, logvar):
    std = torch.exp(logvar/2)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE_Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE_Decoder, self).__init__()
        
        self.num_layer=2
        self.lstm = nn.LSTM(z_dim, h_dim, num_layers=self.num_layer, batch_first=True, bidirectional=False)
        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        # output layer
        self.fc3 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        z, (_,_) = self.lstm(z)
        z = self.fc1(z)
        x_reconst = F.sigmoid(self.fc3(z))
        return x_reconst


class MyArch(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            json_directory
    ):
        super(MyArch, self).__init__()
        self.device =device
        self.json_directory = json_directory

        self.max_length = args.max_length
        self.history_length = args.history_length
        self.alpha = args.alpha
        self.beta = args.beta
        self.modals = args.modals
        
        self.fusion_type = args.fusion_type
        self.visual_type = args.visual_type
        self.audio_type = args.audio_type
        self.use_RL = args.use_RL
        self.use_manager = args.use_manager

        if args.act == 'relu':
            self.act = nn.ReLU()

        self.gpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='!', bos_token='#')
        self.embedding_layer = self.gpt_model.get_input_embeddings()
        self.word_dimension = self.gpt_model.config.hidden_size  # 1280
        
        if self.fusion_type == 'tf_decoder':
            if args.audio_type == 'wav2vec2':
                self.audio_feature_dimension = 768
            elif args.audio_type == 'wavlm':
                self.audio_feature_dimension = 1024
                
            if args.visual_type == 'landmark':
                self.visual_feature_dimension = 196 * 7
            elif args.visual_type == 'face_image':
                self.visual_feature_dimension = 512
            
            if ('a' in self.modals) and ('v' in self.modals) and ('l' in self.modals):
                self.fusion_layer = tf_decoder.TransformerDecoderLayer(d_model=self.word_dimension, nhead=8, batch_first=True)
                self.transformer_fusion = tf_decoder.TransformerDecoder(self.fusion_layer, num_layers=6)
            else:
                self.fusion_layer = nn.TransformerDecoderLayer(d_model=self.word_dimension, nhead=8, batch_first=True)
                self.transformer_fusion = nn.TransformerDecoder(self.fusion_layer, num_layers=6)
            
            if 'a' in self.modals:
                self.audio_projection_layer = nn.Linear(self.audio_feature_dimension, self.word_dimension, bias=False)
                self.audio_projection_layer.weight = torch.nn.init.xavier_uniform_(self.audio_projection_layer.weight)
            
            if 'v' in self.modals:
                self.visual_projection_layer = nn.Linear(self.visual_feature_dimension, self.word_dimension, bias=False)
                self.visual_projection_layer.weight = torch.nn.init.xavier_uniform_(self.visual_projection_layer.weight)
        

        self.loss_function = nn.CrossEntropyLoss()
        
        if self.use_manager:
            hidden_dim=512
            latent_dim=100
            self.vae_encoder = VAE_Encoder(x_dim=self.word_dimension, h_dim=hidden_dim, z_dim=latent_dim)
            self.vae_decoder = VAE_Decoder(x_dim=self.word_dimension, h_dim=hidden_dim, z_dim=latent_dim)

    def forward(self, inputs, label, metric_log=False, epoch=0):
        textual = inputs[0]  # token || context
        acoustic = inputs[1]   # [batch, max_length, audio_dim] || audio path
        visual = inputs[2]  # [batch, 1, visual_dim] || image path
        history = inputs[3]

        labels = label[0]

        textual['input_ids'] = torch.squeeze(textual['input_ids'], dim=1)  # [batch_size, padding_size]
        textual['attention_mask'] = torch.squeeze(textual['attention_mask'], dim=1)
        history['input_ids'] = torch.squeeze(history['input_ids'], dim=1)  # [batch_size, padding_size]
        history['attention_mask'] = torch.squeeze(history['attention_mask'], dim=1)
        labels['input_ids'] = torch.squeeze(labels['input_ids'], dim=1)
        labels['attention_mask'] = torch.squeeze(labels['attention_mask'], dim=1)
        
        inputs_embeds = self.embedding_layer.weight.data[textual['input_ids']]  # torch.Size([batch, max_len, word_dim])
        history_embeds = self.embedding_layer.weight.data[history['input_ids']]  # torch.Size([batch, history_len, word_dim])
        labels_embeds = self.embedding_layer.weight.data[labels['input_ids']]   # torch.Size([batch, max_len, word_dim])
        
        
        if self.fusion_type == 'tf_decoder':
            # ==== step 1. multimodal feature ====
            if 'a' in self.modals:
                acoustic_feature = self.audio_projection_layer(acoustic)  # torch.Size([batch, audio_pad_size, word_dim])
            if 'v' in self.modals:
                visual_feature = self.visual_projection_layer(visual)  # torch.Size([batch, 1, word_dim])
                visual_feature = torch.unsqueeze(visual_feature, dim=1) # [batch, 1, visual_dim]
            
            # ==== step 2. fusion ====
            if ('a' in self.modals) and ('v' in self.modals) and ('l' in self.modals):
                feature = self.transformer_fusion(tgt=inputs_embeds, memory=acoustic_feature, memory2=visual_feature, tgt_key_padding_mask=textual['attention_mask'])
            elif ('a' in self.modals)  and ('l' in self.modals):
                feature = self.transformer_fusion(tgt=inputs_embeds, memory=acoustic_feature, tgt_key_padding_mask=textual['attention_mask'])
            elif ('v' in self.modals) and ('l' in self.modals):
                feature = self.transformer_fusion(tgt=inputs_embeds, memory=visual_feature, tgt_key_padding_mask=textual['attention_mask'])
                
            inputs_embeds = feature  # torch.Size([batch, max_len, word_dim])
            
            concat_inputs = torch.cat([history_embeds, inputs_embeds, labels_embeds], dim=1)  # torch.Size([batch, history_len + max_len + max_len, word_dim])
            concat_mask = torch.cat([history['attention_mask'], textual['attention_mask'], labels['attention_mask']], dim=1)
            
        else:
            concat_inputs = torch.cat([history_embeds, inputs_embeds, labels_embeds], dim=1)  # torch.Size([batch, history_len + max_len + max_len, word_dim])
            concat_mask = torch.cat([history['attention_mask'], textual['attention_mask'], labels['attention_mask']], dim=1)
            
        # ==== step 3. utterance level train ====
        if self.use_manager:
            utterance_input = torch.cat([history_embeds, inputs_embeds], dim=1)
            # print("utterance_input:", utterance_input.shape)
            z, mu, logvar = self.vae_encoder(utterance_input)
            # print("mu:", mu.shape)
            # print("logvar:", logvar.shape)
            x_reconst = self.vae_decoder(z)
            
            ''' https://github.com/natashamjaques/neural_chat/blob/master/model/utils/probability.py '''
            # reconst_loss1 = 0.5 * torch.sum(-torch.log(torch.tensor(2.0) * np.pi) - torch.log(logvar) - ((z - mu).pow(2) / logvar), dim=1)
            reconst_loss1 = 0.5 * torch.sum(-torch.log(torch.tensor(2.0) * np.pi) - logvar[:,-1,:] - ((z[:,-1,:] - mu[:,-1,:]).pow(2) / logvar[:,-1,:].exp()), dim=1)
            # print("x_reconst:", x_reconst.shape)
            # print("utterance_input:", utterance_input.shape)
            # print("reconst_loss1: ",reconst_loss1)
            reconst_loss2 = F.mse_loss(x_reconst, utterance_input, reduction='mean')
            # print("reconst_loss2: ",reconst_loss2)

            # kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
            
            concat_inputs = torch.cat([x_reconst, labels_embeds], dim=1)

            # manager_loss = reconst_loss + kl_div
            manager_loss = -self.alpha * reconst_loss1.sum() + reconst_loss2
        else:
            manager_loss = 0
        
        # ==== step 4. Generate next sentence ====
        outputs = self.gpt_model(inputs_embeds=concat_inputs,
                                 attention_mask=concat_mask,
                                 )
        sft_idx = textual['input_ids'].shape[-1] + history['input_ids'].shape[-1]
            
        out_logit = outputs.logits[:, sft_idx-1:-1].contiguous().view(-1, 50257)
        
        worker_loss = self.loss_function(out_logit, labels['input_ids'][:, :].contiguous().view(-1))
        
        # ==== step 5. Get loss with Reward ====
        
        if self.use_RL:
            # print(self.tokenizer.batch_decode(torch.argmax(outputs.logits[:, sft_idx-1:-1].contiguous(), dim=2), skip_special_tokens=True))
            output_sentence = self.tokenizer.batch_decode(torch.argmax(outputs.logits[:, sft_idx-1:-1].contiguous(), dim=2), skip_special_tokens=True)
            
            pre_sentence = self.tokenizer.batch_decode(textual['input_ids'], skip_special_tokens=True)
            rewards = get_rewards(pre_sentence=pre_sentence, output_sentence=output_sentence)
            total_loss = (manager_loss +  self.beta * worker_loss) * (1/rewards)
        else:
            total_loss = (manager_loss +  self.beta * worker_loss)

        if metric_log:
            attention_mask = torch.cat([history['attention_mask'], textual['attention_mask']], dim=1)
                
            output = self.gpt_model.generate(max_length=self.max_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            inputs_embeds=torch.cat([history_embeds, inputs_embeds], dim=1),
                                            attention_mask=attention_mask,
                                            num_beams=4,
                                            do_sample=True,
                                            top_k=50,
                                            top_p=0.90,
                                            )
            
            # eval_result = self.get_eval_matric(output, labels['input_ids'])
            outputs_sentence = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            ref_sentence = self.tokenizer.batch_decode(labels['input_ids'], skip_special_tokens=True)

            eval_result = calculate_eval_matric(outputs_sentence, ref_sentence)
            
            self.save_output(outputs_sentence, epoch)
        else:
            eval_result = None

        return total_loss, eval_result

    def inference(self, inputs, greedy=False):
        textual = inputs[0]  # token || context
        acoustic = inputs[1]   # [batch, max_length, audio_dim] || audio path
        visual = inputs[2]  # [batch, 1, visual_dim] || image path
        history = inputs[3]
        
        # ==== step 0. preprocess ====
        textual['input_ids'] = torch.unsqueeze(textual['input_ids'], dim=0)
        textual['attention_mask'] = torch.unsqueeze(textual['attention_mask'], dim=0)
        acoustic = torch.unsqueeze(acoustic, dim=0)
        visual = torch.unsqueeze(visual, dim=0)
        history['input_ids'] = torch.unsqueeze(history['input_ids'], dim=0)
        history['attention_mask'] = torch.unsqueeze(history['attention_mask'], dim=0)
        
        history['input_ids'] = torch.squeeze(history['input_ids'], dim=1)  # [batch_size, padding_size]
        history['attention_mask'] = torch.squeeze(history['attention_mask'], dim=1)
        history_embeds = self.embedding_layer.weight.data[history['input_ids']]  # torch.Size([batch, history_len, word_dim])
        
        
        if self.fusion_type == 'tf_decoder':
            # ==== step 0. preprocess ====
            textual['input_ids'] = torch.squeeze(textual['input_ids'], dim=1)  # [batch_size, padding_size]
            textual['attention_mask'] = torch.squeeze(textual['attention_mask'], dim=1)
            
            # ==== step 1. multimodal feature ====
            inputs_embeds = self.embedding_layer.weight.data[textual['input_ids']]  # torch.Size([batch, max_len, word_dim])
            if 'a' in self.modals:
                acoustic_feature = self.audio_projection_layer(acoustic)  # torch.Size([batch, audio_pad_size, word_dim])
            if 'v' in self.modals:
                visual_feature = self.visual_projection_layer(visual)  # torch.Size([batch, 1, word_dim])
                visual_feature = torch.unsqueeze(visual_feature, dim=1) # [batch, 1, visual_dim]
            
            # ==== step 2. fusion ====
            if ('a' in self.modals) and ('v' in self.modals) and ('l' in self.modals):
                feature = self.transformer_fusion(tgt=inputs_embeds, memory=acoustic_feature, memory2=visual_feature, tgt_key_padding_mask=textual['attention_mask'])
            elif ('a' in self.modals)  and ('l' in self.modals):
                feature = self.transformer_fusion(tgt=inputs_embeds, memory=acoustic_feature, tgt_key_padding_mask=textual['attention_mask'])
            elif ('v' in self.modals) and ('l' in self.modals):
                feature = self.transformer_fusion(tgt=inputs_embeds, memory=visual_feature, tgt_key_padding_mask=textual['attention_mask'])
                
            inputs_embeds = feature  # torch.Size([batch, max_len, word_dim])
            
            concat_inputs = torch.cat([history_embeds, inputs_embeds], dim=1)  # torch.Size([batch, history_len + max_len + max_len, word_dim])
            concat_mask = torch.cat([history['attention_mask'], textual['attention_mask']], dim=1)
            
        else:
            concat_inputs = torch.cat([history_embeds, inputs_embeds], dim=1)  # torch.Size([batch, history_len + max_len + max_len, word_dim])
            concat_mask = torch.cat([history['attention_mask'], textual['attention_mask']], dim=1)
            
        # ==== step 3. utterance level train ====
        if self.use_manager:
            utterance_input = torch.cat([history_embeds, inputs_embeds], dim=1)
            z, mu, logvar = self.vae_encoder(utterance_input)
            x_reconst = self.vae_decoder(z)
            
            concat_inputs = x_reconst
        
        # ==== step 4. Generate next sentence ====
        if greedy:
            output = self.gpt_model.generate(max_length=self.max_length,
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             inputs_embeds=concat_inputs,
                                             attention_mask=concat_mask,
                                             )
        else:
            output = self.gpt_model.generate(max_length=self.max_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            inputs_embeds=concat_inputs,
                                            attention_mask=concat_mask,
                                            num_beams=4,
                                            do_sample=True,
                                            top_k=50,
                                            top_p=0.90,
                                            )
        
        return output

    def get_eval_matric(self, output, ref):
        '''
        output: metric dictionary
            {'Bleu_1': 0.1353352829659427, 
            'Bleu_2': 0.00013533528303361024, 
            'Bleu_3': 1.3533528305616618e-05, 
            'Bleu_4': 4.2796774227674215e-06, 
            'METEOR': 0.14814814814814814, 
            'ROUGE_L': 0.45864661654135336, 
            'CIDEr': 0.0, 
            'SPICE': 0.0}
        '''
        outputs_sentence = self.tokenizer.batch_decode(
            output, skip_special_tokens=True)
        ref_sentence = self.tokenizer.batch_decode(
            ref, skip_special_tokens=True)

        eval_result = calculate_eval_matric(outputs_sentence, ref_sentence)

        return eval_result

    def save_output(self, output, epoch):
        json_name = f'{self.modals}_manager:{self.use_manager}_RL:{self.use_RL}_alpha:{str(self.alpha)}_{epoch}epoch_result.json'
        # 1. 기존 데이터 불러오기
        try:
            with open(self.json_directory + json_name, 'r') as json_file:
                existing_data = json.load(json_file)
        except FileNotFoundError:
            existing_data = []

        # 2. 새 데이터 추가
        new_entry = {
            'output': output
        }

        existing_data.append(new_entry)

        # 3. 업데이트된 데이터 저장
        with open(self.json_directory + json_name, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
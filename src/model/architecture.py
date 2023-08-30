#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from . import tf_decoder
from .scores import get_scores


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
        self.modals = args.modals
        
        self.visual_type = args.visual_type
        self.audio_type = args.audio_type
        self.use_score = args.use_score
        self.use_query = args.use_query
        
        if args.act == 'relu':
            self.act = nn.ReLU()

        self.generator_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='*', bos_token='#')
        self.embedding_layer = self.generator_model.get_input_embeddings()
        self.word_dimension = self.generator_model.config.hidden_size  # 1280
        
        if ('a' in self.modals) and ('v' in self.modals) and ('l' in self.modals):
            self.fusion_layer = tf_decoder.TransformerDecoderLayer(d_model=self.word_dimension, nhead=8, batch_first=True)
            self.transformer_fusion = tf_decoder.TransformerDecoder(self.fusion_layer, num_layers=6)
        if (('a' in self.modals) and ('l' in self.modals)) or (('v' in self.modals) and ('l' in self.modals)):
            self.fusion_layer = nn.TransformerDecoderLayer(d_model=self.word_dimension, nhead=8, batch_first=True)
            self.transformer_fusion = nn.TransformerDecoder(self.fusion_layer, num_layers=6)
        
        if 'a' in self.modals:
            if args.audio_type == 'wav2vec2':
                self.audio_feature_dimension = 768
            elif args.audio_type == 'wavlm':
                self.audio_feature_dimension = 1024
                
            self.audio_projection_layer = nn.Linear(self.audio_feature_dimension, self.word_dimension, bias=False)
            self.audio_projection_layer.weight = torch.nn.init.xavier_uniform_(self.audio_projection_layer.weight)
            
        if 'v' in self.modals:
            if args.visual_type == 'landmark':
                self.visual_feature_dimension = 196 * 7  # landmark dimension * number of landmark
            elif args.visual_type == 'face_image':
                self.visual_feature_dimension = 512
                
            self.visual_projection_layer = nn.Linear(self.visual_feature_dimension, self.word_dimension, bias=False)
            self.visual_projection_layer.weight = torch.nn.init.xavier_uniform_(self.visual_projection_layer.weight)
            
        self.loss_function = nn.CrossEntropyLoss()


    def forward(self, inputs, label, metric_log=False):
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
        
        if self.use_query:
            history_embeds = torch.cat([history_embeds, inputs_embeds], dim=1)
            history['attention_mask'] = torch.cat([history['attention_mask'], textual['attention_mask']], dim=1)
        
        # ==== step 1. multimodal feature ====
        if 'a' in self.modals:
            acoustic_feature = self.audio_projection_layer(acoustic)  # torch.Size([batch, audio_pad_size, word_dim])
        if 'v' in self.modals:
            visual_feature = self.visual_projection_layer(visual)  # torch.Size([batch, 1, word_dim])
            visual_feature = torch.unsqueeze(visual_feature, dim=1) # [batch, 1, visual_dim]
        
        # ==== step 2. fusion ====
        if ('a' in self.modals) and ('v' in self.modals) and ('l' in self.modals):
            feature = self.transformer_fusion(tgt=inputs_embeds, memory=acoustic_feature, memory2=visual_feature, tgt_key_padding_mask=textual['attention_mask'])
            inputs_embeds = feature  # torch.Size([batch, max_len, word_dim])
        elif ('a' in self.modals)  and ('l' in self.modals):
            feature = self.transformer_fusion(tgt=inputs_embeds, memory=acoustic_feature, tgt_key_padding_mask=textual['attention_mask'])
            inputs_embeds = feature  # torch.Size([batch, max_len, word_dim])
        elif ('v' in self.modals) and ('l' in self.modals):
            feature = self.transformer_fusion(tgt=inputs_embeds, memory=visual_feature, tgt_key_padding_mask=textual['attention_mask'])
            inputs_embeds = feature  # torch.Size([batch, max_len, word_dim])

        concat_inputs = torch.cat([history_embeds, inputs_embeds, labels_embeds], dim=1)  # torch.Size([batch, history_len + max_len + max_len, word_dim])
        concat_mask = torch.cat([history['attention_mask'], textual['attention_mask'], labels['attention_mask']], dim=1)    
        
        # ==== step 3. Generate next sentence ====
        outputs = self.generator_model(inputs_embeds=concat_inputs,
                                       attention_mask=concat_mask,
                                       )
        if self.use_query:
            sft_idx = textual['input_ids'].shape[-1] + history['input_ids'].shape[-1] + textual['input_ids'].shape[-1]
        else:
            sft_idx = textual['input_ids'].shape[-1] + history['input_ids'].shape[-1]
        
            
        out_logit = outputs.logits[:, sft_idx-1:-1].contiguous().view(-1, 50257)
        
        loss = self.loss_function(out_logit, labels['input_ids'][:, :].contiguous().view(-1))
        
        # ==== step 4. Get loss with score ====
        
        if self.use_score:
            output_sentence = self.tokenizer.batch_decode(torch.argmax(outputs.logits[:, sft_idx-1:-1].contiguous(), dim=2), skip_special_tokens=True)
            # print(output_sentence)
            
            pre_sentence = self.tokenizer.batch_decode(textual['input_ids'], skip_special_tokens=True)
            rewards = get_scores(pre_sentence=pre_sentence, output_sentence=output_sentence)
            loss *= (1/rewards)

        if metric_log:
            attention_mask = torch.cat([history['attention_mask'], textual['attention_mask']], dim=1)
                
            output = self.generator_model.generate(max_length=self.max_length,
                                                   pad_token_id=self.tokenizer.pad_token_id,
                                                   inputs_embeds=torch.cat([history_embeds, inputs_embeds], dim=1),
                                                   attention_mask=attention_mask,
                                                   num_beams=4,
                                                   do_sample=True,
                                                   top_k=50,
                                                   top_p=0.90,
                                                   )
            
            outputs_sentence = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            ref_sentence = self.tokenizer.batch_decode(labels['input_ids'], skip_special_tokens=True)
            
            self.save_output(outputs_sentence, ref_sentence)

        return loss

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

        textual['input_ids'] = torch.squeeze(textual['input_ids'], dim=1)  # [batch_size, padding_size]
        textual['attention_mask'] = torch.squeeze(textual['attention_mask'], dim=1)        
        history['input_ids'] = torch.squeeze(history['input_ids'], dim=1)  # [batch_size, padding_size]
        history['attention_mask'] = torch.squeeze(history['attention_mask'], dim=1)
        
        inputs_embeds = self.embedding_layer.weight.data[textual['input_ids']]  # torch.Size([batch, max_len, word_dim])
        history_embeds = self.embedding_layer.weight.data[history['input_ids']]  # torch.Size([batch, history_len, word_dim])
        
        if self.use_query:
            history_embeds = torch.cat([history_embeds, inputs_embeds], dim=1)
            history['attention_mask'] = torch.cat([history['attention_mask'], textual['attention_mask']], dim=1)
            
        # ==== step 1. multimodal feature ====
        if 'a' in self.modals:
            acoustic_feature = self.audio_projection_layer(acoustic)  # torch.Size([batch, audio_pad_size, word_dim])
        if 'v' in self.modals:
            visual_feature = self.visual_projection_layer(visual)  # torch.Size([batch, 1, word_dim])
            visual_feature = torch.unsqueeze(visual_feature, dim=1) # [batch, 1, visual_dim]
        
        # ==== step 2. fusion ====
        if ('a' in self.modals) and ('v' in self.modals) and ('l' in self.modals):
            feature = self.transformer_fusion(tgt=inputs_embeds, memory=acoustic_feature, memory2=visual_feature, tgt_key_padding_mask=textual['attention_mask'])
            inputs_embeds = feature  # torch.Size([batch, max_len, word_dim])
        elif ('a' in self.modals)  and ('l' in self.modals):
            feature = self.transformer_fusion(tgt=inputs_embeds, memory=acoustic_feature, tgt_key_padding_mask=textual['attention_mask'])
            inputs_embeds = feature  # torch.Size([batch, max_len, word_dim])
        elif ('v' in self.modals) and ('l' in self.modals):
            feature = self.transformer_fusion(tgt=inputs_embeds, memory=visual_feature, tgt_key_padding_mask=textual['attention_mask'])
            inputs_embeds = feature  # torch.Size([batch, max_len, word_dim])
        
        concat_inputs = torch.cat([history_embeds, inputs_embeds], dim=1)  # torch.Size([batch, history_len + max_len + max_len, word_dim])
        concat_mask = torch.cat([history['attention_mask'], textual['attention_mask']], dim=1)
            
        # ==== step 3. Generate next sentence ====
        if greedy:
            output = self.generator_model.generate(max_length=self.max_length,
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             inputs_embeds=concat_inputs,
                                             attention_mask=concat_mask,
                                             )
        else:
            output = self.generator_model.generate(max_length=self.max_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            inputs_embeds=concat_inputs,
                                            attention_mask=concat_mask,
                                            num_beams=4,
                                            do_sample=True,
                                            top_k=50,
                                            top_p=0.90,
                                            )
        
        return output


    def save_output(self, output_sentence, ref_sentence, epoch):
        json_name = f'epoch:{epoch}_result.json'
        # 1. load data
        try:
            with open(self.json_directory + json_name, 'r') as json_file:
                existing_data = json.load(json_file)
        except FileNotFoundError:
            existing_data = []

        # 2. add new data
        new_entry = {
            'output_sentence': output_sentence,
            'ref_sentence': ref_sentence
        }

        existing_data.append(new_entry)

        # 3. saver updated data
        with open(self.json_directory + json_name, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
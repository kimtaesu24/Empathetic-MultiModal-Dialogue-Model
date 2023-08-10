import numpy as np
import pandas as pd
import torch
import ast
import json
import os

from model import modules
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image

class MELD_Decoder_Dataset(Dataset):
    def __init__(self, data_path, device, args, mode='train'):
        self.data_path = data_path
        self.device = device
        self.modals = args.modals
        if args.audio_type == 'wav2vec2':
            self.audio_feature_path = self.data_path + 'audio_feature/wav2vec2/' + mode
        elif args.audio_type == 'wavlm':
            self.audio_feature_path = self.data_path + 'audio_feature/wavlm/' + mode
        self.visual_type = args.visual_type
        self.landmark_num = 7
        
        self.max_length = args.max_length
        self.history_length = args.history_length
        self.audio_padding = args.audio_pad_size
        self.fusion_type = args.fusion_type

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='*', bos_token='#')
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'

        self.label_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='*', bos_token='#')

        if mode == 'train':
            self.FA = pd.read_csv(self.data_path + 'new_train_FA_matched.csv')
            self.fer = pd.read_csv(self.data_path + 'new_train_FER_matched.csv')
            self.landmark = pd.read_csv(self.data_path + 'new_train_LM_matched.csv')
        else:
            self.FA = pd.read_csv(self.data_path + 'new_valid_FA_matched.csv')
            self.fer = pd.read_csv(self.data_path + 'new_valid_FER_matched.csv')
            self.landmark = pd.read_csv(self.data_path + 'new_valid_LM_matched.csv')

        self.history_path = self.data_path + mode
        # self.T_padding = max(len(i) for i in self.fer['T_list'].apply(eval))  # 459
        self.manual_index = 0
        
        self.mode = mode

    def __len__(self):
        length = 0
        for idx in range(len(self.FA) - 1):
            if (self.FA['Dialogue_ID'][idx] == self.FA['Dialogue_ID'][idx+1]):  # same dialogue
                # next utterance
                if (self.FA['Utterance_ID'][idx] == (self.FA['Utterance_ID'][idx+1] - 1)):
                    length += 1
        return length

    def __getitem__(self, idx):
        if idx == 0:
            self.manual_index = 0  # initialize

        idx += self.manual_index
        # next dialogue appear OR empty uttrance appear
        while ((self.FA['Dialogue_ID'][idx] != self.FA['Dialogue_ID'][idx+1]) or (self.FA['Utterance_ID'][idx] != (self.FA['Utterance_ID'][idx+1] - 1))):
            self.manual_index += 1
            idx += 1

        dia_id = self.FA['Dialogue_ID'][idx]
        utt_id = self.FA['Utterance_ID'][idx]
        
        # extract textual feature
        context = ' '.join(ast.literal_eval(self.FA['word'][idx])).lower() + '.'
        response = ' '.join(ast.literal_eval(self.FA['word'][idx+1])).lower() + '.'

        tokens = self.tokenizer(context + self.tokenizer.eos_token,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt'
                                )

        # extract audio feature
        audio_feature = torch.tensor(0)
        if 'a' in self.modals:
            waveform = torch.load(self.audio_feature_path+'/dia{}_utt{}_16000.pt'.format(dia_id, utt_id))
            waveform = torch.squeeze(waveform, dim=0)
            audio_feature = modules.audio_pad(waveform, self.audio_padding)
            audio_feature = audio_feature.to(self.device)

        # extract visual feature
        visual_feature = torch.tensor(0)
        if 'v' in self.modals:
            if self.visual_type == 'landmark':
                landmark_set = torch.tensor(ast.literal_eval(self.landmark['landmark_list'][idx]))
                if len(landmark_set) >= self.landmark_num:
                    visual_feature = landmark_set[[round(i * len(landmark_set)/7) for i in range(7)]] # number of landmark inputs is 7
                    visual_feature = torch.flatten(visual_feature)  # [self.landmark_num, landmark_dim] -> [self.landmark_num * landmark_dim]
                else:
                    visual_feature = modules.pad(torch.flatten(landmark_set[:]), self.landmark_num * landmark_set[0].shape[0])
            elif self.visual_type == 'face_image':
                src_path = f"{self.data_path}/{self.mode}/dia{dia_id}/utt{utt_id}"
                dirListing = os.listdir(src_path)
                image_path = src_path+'/{:06d}.jpg'.format((len(dirListing)-3)//2)
                try:
                    img = Image.open(image_path)
                    # print(img.size)
                    # img_cropped = self.mtcnn(img)
                    normalized_image = self.transform(img)
                    # print(img_cropped.shape)
                    visual_feature = self.resnet(normalized_image.unsqueeze(0).to(self.device))
                    # print(visual_feature.shape)
                    visual_feature = torch.squeeze(visual_feature, dim=0)
                except:
                    visual_feature = torch.zeros(512)  # doesn't exist speaker face
        # get dialogue history
        with open(f"{self.history_path}/dia{dia_id}/utt{utt_id}/dia{dia_id}_utt{utt_id}_history.json", "r") as json_file:
            historys = json.load(json_file)

        input_historys = ""
        for utt_hist in historys:
            input_historys += utt_hist+self.tokenizer.eos_token

        input_historys_tokens = self.tokenizer(input_historys,
                                               padding='max_length',
                                               max_length=self.history_length,
                                               truncation=True,
                                               return_attention_mask=True,
                                               return_tensors='pt'
                                               )
        
        # get label
        tokens_labels = self.label_tokenizer(response + self.label_tokenizer.eos_token,
                                             padding='max_length',
                                             max_length=self.max_length,
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_tensors='pt'
                                             )

        inputs = [tokens.to(self.device),
                  audio_feature.to(self.device),
                  visual_feature.to(self.device),
                  input_historys_tokens.to(self.device),
                  ]

        labels = [tokens_labels.to(self.device)]

        return inputs, labels

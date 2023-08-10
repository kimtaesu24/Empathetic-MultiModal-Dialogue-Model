import numpy as np
import pandas as pd
import torch
import ast
import json
import os
import torchvision.transforms as transforms

from model import modules
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

class EC_Decoder_Dataset(Dataset):
    def __init__(self, data_path, device, args, mode='train'):
        self.data_path = data_path  # '/home2/dataset/english_conversation/'
        self.device = device
        self.modals = args.modals
        if args.audio_type == 'wav2vec2':
            self.audio_feature_path = self.data_path + 'audio_feature/wav2vec2/' + mode
        elif args.audio_type == 'wavlm':
            self.audio_feature_path = self.data_path + 'audio_feature/wavlm/' + mode
        self.visual_type = args.visual_type
        self.landmark_num = 7
           
        self.max_length =args.max_length
        self.history_length = args.history_length
        self.audio_pad_size = args.audio_pad_size
        self.fusion_type = args.fusion_type

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='!', bos_token='#')
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'

        self.label_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='!', bos_token='#')

        if mode == 'train':
            self.text_data = pd.read_csv(f'{self.data_path}/{mode}/text/text_data.csv')
        else:
            self.text_data = pd.read_csv(f'{self.data_path}/{mode}/text/text_data.csv')

        self.history_path = f'{self.data_path}{mode}/text/history'
        self.manual_index = 0
        
        self.transform = transforms.Compose([
            transforms.Resize((160,160)),        # Resize the image to the desired size
            transforms.ToTensor()                   # Convert the image to a PyTorch tensor
        ])
        # self.mtcnn = MTCNN()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.visual_list = os.listdir(f'{self.data_path}/{mode}/speaker_image/')
        self.mode = mode

    def __len__(self):
        total_data = len(os.listdir(f'{self.data_path}{self.mode}/speaker_image'))
        total_dia = max(self.text_data['Dialogue_ID'])
        # for idx in range(len(os.listdir(f'{self.data_path}{self.mode}/speaker_image')) - 1):
        #     if (self.text_data['Dialogue_ID'][idx] == self.text_data['Dialogue_ID'][idx+1]):  # same dialogue
        #         length += 1
        return total_data - total_dia

    def __getitem__(self, idx):
        if idx == 0:
            self.manual_index = 0  # initialize

        idx += self.manual_index
        # next dialogue appear OR empty uttrance appear
        while (self.text_data['Dialogue_ID'][idx] != self.text_data['Dialogue_ID'][idx+1]):
            self.manual_index += 1
            idx += 1

        data = self.visual_list[idx]
        
        dia_id = data.split('_')[0][3:]
        utt_id = data.split('_')[1][3:]
        
        # dia_id = self.text_data['Dialogue_ID'][idx]
        # utt_id = self.text_data['Utterance_ID'][idx]
        
        # extract textual feature
        context = ' '.join(self.text_data['Utterance'][(self.text_data['Dialogue_ID'] == int(dia_id)) & (self.text_data['Utterance_ID'] == int(utt_id))]).lower() + '.'
        response = ' '.join(self.text_data['Utterance'][(self.text_data['Dialogue_ID'] == int(dia_id)) & (self.text_data['Utterance_ID'] == int(utt_id)+1)]).lower() + '.'

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
            waveform = torch.load(self.audio_feature_path+'/dia{}_utt{}.pt'.format(dia_id, utt_id))
            waveform = torch.squeeze(waveform, dim=0)
            audio_feature = modules.audio_pad(waveform, self.audio_pad_size)

        # extract visual feature
        visual_feature = torch.tensor(0)
        if 'v' in self.modals:
            if self.visual_type == 'landmark':
                src_path = f"{self.data_path}/{self.mode}/landmark/dia{dia_id}_utt{utt_id}/"
                dirListing = os.listdir(src_path)
                if len(dirListing) >= self.landmark_num:
                    tensor_list = []
                    for i in range(self.landmark_num):
                        tensor = torch.load(src_path + dirListing[round(i * len(dirListing)/self.landmark_num)])
                        tensor_list.append(torch.tensor(tensor.flatten(), dtype=torch.float32))  # [2,96] -> [landmark_dim]
                    visual_feature = torch.cat(tensor_list, dim=0)
                else:
                    tensor_list = []
                    for lm in dirListing:
                        tensor = torch.load(src_path + lm)
                        tensor_list.append(torch.tensor(tensor.flatten(), dtype=torch.float32))  # [2,96] -> [landmark_dim]
                    visual_feature = torch.cat(tensor_list, dim=0)
                    visual_feature = modules.pad(visual_feature, self.landmark_num * tensor_list[0].shape[0])
            elif self.visual_type == 'face_image':
                src_path = f"{self.data_path}/{self.mode}/speaker_image/dia{dia_id}_utt{utt_id}"
                dirListing = os.listdir(src_path)
                image_path = src_path+f'/{format(dirListing[(len(dirListing))//2])}'  # middle file in the directory
                img = Image.open(image_path)
                # print(img.size)
                # img_cropped = self.mtcnn(img)
                normalized_image = self.transform(img)
                # print(img_cropped.shape)
                visual_feature = self.resnet(normalized_image.unsqueeze(0).to(self.device))
                # print(visual_feature.shape)
                visual_feature = torch.squeeze(visual_feature, dim=0)

        # get dialogue history
        with open(f"{self.history_path}/dia{dia_id}_utt{utt_id}/dia{dia_id}_utt{utt_id}_history.json", "r") as json_file:
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

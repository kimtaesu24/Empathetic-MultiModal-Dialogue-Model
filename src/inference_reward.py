import torch
import sys
import pandas as pd
import ast
import json
import argparse
import torchvision.transforms as transforms
import os
import warnings

from transformers import AutoTokenizer
from model.architecture_reward import MyArch
from utils.util import log_args
from model import modules
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# sys.path.insert(0, '/home2/s20235100/Conversational-AI/MyModel/src/model/')
warnings.filterwarnings(action='ignore')

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='*', bos_token='#')
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'left'

def load_EC_data(data_path, args, dia_id, utt_id, device, mode='test'):
    #########################################################################
    modals = args.modals
    if args.audio_type == 'wav2vec2':
        audio_feature_path = data_path + 'audio_feature/wav2vec2/' + mode
    elif args.audio_type == 'wavlm':
        audio_feature_path = data_path + 'audio_feature/wavlm/' + mode
    visual_type = args.visual_type
        
    max_length =args.max_length
    history_length = args.history_length
    audio_pad_size = args.audio_pad_size
    fusion_type = args.fusion_type
    
    text_data = pd.read_csv(f'{data_path}/{mode}/text/text_data.csv')
    history_path = f'{data_path}{mode}/text/history'
    
    transform = transforms.Compose([
            transforms.Resize((160,160)),        # Resize the image to the desired size
            transforms.ToTensor()                   # Convert the image to a PyTorch tensor
        ])
    # self.mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
    #########################################################################
    context = ' '.join(text_data['Utterance'][(text_data['Dialogue_ID'] == int(dia_id)) & (text_data['Utterance_ID'] == int(utt_id))]).lower() + '.'

    tokens = tokenizer(context + tokenizer.eos_token,
                        padding='max_length',
                        max_length=max_length,
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                        )

    # extract audio feature
    audio_feature = torch.tensor(0)
    if 'a' in modals:
        waveform = torch.load(audio_feature_path+'/dia{}_utt{}.pt'.format(dia_id, utt_id))
        waveform = torch.squeeze(waveform, dim=0)
        audio_feature = modules.audio_pad(waveform, audio_pad_size)

    # extract visual feature
    visual_feature = torch.tensor(0)
    if 'v' in modals:
        if visual_type == 'landmark':
            landmark_set = torch.tensor(ast.literal_eval(landmark['landmark_list'][idx]))
            visual_feature = landmark_set[len(landmark_set)//2]
        elif visual_type == 'face_image':
            src_path = f"{data_path}/{mode}/speaker_image/dia{dia_id}_utt{utt_id}"
            dirListing = os.listdir(src_path)
            image_path = src_path+f'/{format(dirListing[(len(dirListing))//2])}'  # middle file in the directory
            img = Image.open(image_path)
            # print(img.size)
            # img_cropped = self.mtcnn(img)
            normalized_image = transform(img)
            # print(img_cropped.shape)
            visual_feature = resnet(normalized_image.unsqueeze(0).to(device))
            # print(visual_feature.shape)
            visual_feature = torch.squeeze(visual_feature, dim=0)

    # get dialogue history
    with open(f"{history_path}/dia{dia_id}_utt{utt_id}/dia{dia_id}_utt{utt_id}_history.json", "r") as json_file:
        historys = json.load(json_file)

    input_historys = ""
    for utt_hist in historys:
        input_historys += utt_hist+tokenizer.eos_token

    input_historys_tokens = tokenizer(input_historys,
                                            padding='max_length',
                                            max_length=history_length,
                                            truncation=True,
                                            return_attention_mask=True,
                                            return_tensors='pt'
                                            )

    inputs = [tokens.to(device),
                audio_feature.to(device),
                visual_feature.to(device),
                input_historys_tokens.to(device),
                ]
    
    historys.append(context)
    
    return inputs, historys
    
def load_MELD_data(data_path, args, dia_id, utt_id, device, mode='test'):
    #########################################################################
    modals = args.modals
    if args.audio_type == 'wav2vec2':
        audio_feature_path = data_path + 'audio_feature/wav2vec2/' + mode
    elif args.audio_type == 'wavlm':
        audio_feature_path = data_path + 'audio_feature/wavlm/' + mode
    visual_type = args.visual_type
        
    max_length =args.max_length
    history_length = args.history_length
    audio_pad_size = args.audio_pad_size
    fusion_type = args.fusion_type
    
    FA = pd.read_csv(data_path + f'new_{mode}_FA_matched.csv')
    # fer = pd.read_csv(data_path + f'new_{mode}_FER_matched.csv')
    landmark = pd.read_csv(data_path + f'new_{mode}_LM_matched.csv')
    history_path = data_path + mode
    #########################################################################

    idx = FA[(FA['Dialogue_ID'] == dia) & (FA['Utterance_ID'] == utt)].index[0]

    context = ' '.join(ast.literal_eval(FA['word'][idx])).lower() + '.'

    tokens = tokenizer(context + tokenizer.eos_token,
                       padding='max_length',
                       max_length=max_length,
                       truncation=True,
                       return_attention_mask=True,
                       return_tensors='pt'
                       )

    # extract audio feature
    audio_feature = torch.tensor(0)
    if 'a' in modals:
        waveform = torch.load(audio_feature_path+'/dia{}_utt{}_16000.pt'.format(dia_id, utt_id))
        waveform = torch.squeeze(waveform, dim=0)
        audio_feature = modules.audio_pad(waveform, audio_pad_size)
        audio_feature = audio_feature.to(device)

    # extract visual feature
    visual_feature = torch.tensor(0)
    if 'v' in modals:
        if visual_type == 'landmark':
            landmark_set = torch.tensor(ast.literal_eval(landmark['landmark_list'][idx]))
            visual_feature = landmark_set[len(landmark_set)//2]
        elif visual_type == 'face_image':
            src_path = f"{data_path}/{mode}/dia{dia_id}/utt{utt_id}"
            dirListing = os.listdir(src_path)
            image_path = src_path+'/{:06d}.jpg'.format((len(dirListing)-3)//2)
            visual_feature = torch.tensor([])

    with open(f"{history_path}/dia{dia}/utt{utt}/dia{dia}_utt{utt}_history.json", "r") as json_file:
        historys = json.load(json_file)

    input_historys = ""
    for utt_hist in historys:
        input_historys += utt_hist+tokenizer.eos_token

    input_historys_tokens = tokenizer(input_historys,
                                      padding='max_length',
                                      max_length=history_length,
                                      truncation=True,
                                      return_attention_mask=True,
                                      return_tensors='pt'
                                      )

    inputs = [tokens.to(device),
              audio_feature.to(device),
              visual_feature.to(device),
              input_historys_tokens.to(device),
              ]

    historys.append(context)

    return inputs, historys


def show_meld_sample(model, args, device):
    dia_utt = [[13, 6],
               [16, 8],
               [37, 6],
               [65, 9],
               [99, 7],
               [114, 4],
               [121, 4],
               [127, 4],
               [131, 7],
               ]
    for d_u in dia_utt:
        dia = d_u[0]
        utt = d_u[1]
        print(f'Test Data: dia{dia}_utt{utt}.mp4')
        

        data_path = '/home2/dataset/MELD/'
        inputs, historys = load_MELD_data(data_path, args, dia, utt, device, mode='test')        
        # for i, k in enumerate(historys):
        #     print(f'{i}-th sentence: {k}')
        
        outputs = model.inference(inputs, greedy=False)
        # print(outputs)
        sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Response: {}".format(sentence))
        print()
        
def show_ec_sample(model, args, device):
    dia_utt = [
            #    [13, 6],
            #    [16, 8],
            #    [37, 6],
            #    [65, 9],
            #    [99, 7],
            #    [114, 4],
            #    [121, 4],
            #    [127, 4],
            #    [131, 7],
               ]
    for d_u in dia_utt:
        dia = d_u[0]
        utt = d_u[1]
        print(f'Test Data: dia{dia}_utt{utt}.mp4')
        
        data_path = '/home2/dataset/english_conversation/'
        inputs, historys = load_EC_data(data_path, args, dia, utt, device, mode='test')
        # for i, k in enumerate(historys):
        #     print(f'{i}-th sentence: {k}')
        
        outputs = model.inference(inputs, greedy=False)
        # print(outputs)
        sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Response: {}".format(sentence))
        print()


if __name__ == '__main__':
    checkpoint = 50
    weight_path = f"../checkpoint/EC/tf_decoder/x_full_model/{str(checkpoint)}_epochs.pt"
    print('checkpoint path at:', weight_path)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='EC', help='select dataset for training')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--act', default='relu', help='type of activation function')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--max_length', type=int, default=60, help='maximum length of utterance')
    parser.add_argument('--history_length', type=int, default=256, help='maximum length of dialogue history')
    parser.add_argument('--audio_pad_size', type=int, default=350, help='time domain padding size for audio')
    parser.add_argument('--alpha', type=float, default=0.001, help='weight for manager component')
    parser.add_argument('--beta', type=float, default=1.0, help='weight for woker component')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--decay_rate', type=float, default=0.98, help='decay rate of learning rate')
    parser.add_argument('--save_at_every', type=int, default=10, help='save checkpoint')
    parser.add_argument('--metric_at_every', type=int, default=10, help='calculate metric scores')
    parser.add_argument('--LLM_freeze', action='store_true', default=False, help='freeze language decoder or not')
    parser.add_argument('--audio_type', default='wav2vec2', help='audio feature extract type |wavlm')
    parser.add_argument('--fusion_type', default='tf_decoder', help='modal fusion method |graph')
    parser.add_argument('--visual_type', default='face_image', help='visual feature type |landmark')
    parser.add_argument('--modals', default='avl', help='modals to fusion')
    parser.add_argument('--use_manager', action='store_true', default=False, help='use reinforcement learning on training or not')
    parser.add_argument('--use_RL', action='store_true', default=False, help='use reinforcement learning on training or not')
    parser.add_argument('--resume', default=None, help='resume train with checkpoint path or not')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode for wandb')
    args = parser.parse_args()
    log_args(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MyArch(args, device).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    print('checkpoint model has loaded')

    show_meld_sample(model, args, device)
    # show_ec_sample(model, args, device)

import torch
import pandas as pd
import json
import argparse
import torchvision.transforms as transforms
import os
import warnings
import natsort

from transformers import AutoTokenizer
from model.architecture import MyArch
from utils.util import log_args
from model import modules
from tqdm import tqdm

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
    landmark_num = 7
        
    max_length =args.max_length
    history_length = args.history_length
    audio_pad_size = args.audio_pad_size
    
    text_data = pd.read_csv(f'{data_path}/{mode}/text/text_data.csv')
    history_path = f'{data_path}{mode}/text/history'
        
    #########################################################################
    context = ' '.join(text_data['Utterance'][(text_data['Dialogue_ID'] == int(dia_id)) & (text_data['Utterance_ID'] == int(utt_id))])

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
            src_path = f"{data_path}/{mode}/landmark/dia{dia_id}_utt{utt_id}/"
            dirListing = os.listdir(src_path)
            if len(dirListing) >= landmark_num:
                tensor_list = []
                for i in range(landmark_num):
                    tensor = torch.load(src_path + dirListing[round(i * len(dirListing)/landmark_num)])
                    tensor_list.append(torch.tensor(tensor.flatten(), dtype=torch.float32))  # [2,96] -> [landmark_dim]
                visual_feature = torch.cat(tensor_list, dim=0)
            else:
                tensor_list = []
                for lm in dirListing:
                    tensor = torch.load(src_path + lm)
                    tensor_list.append(torch.tensor(tensor.flatten(), dtype=torch.float32))  # [2,96] -> [landmark_dim]
                visual_feature = torch.cat(tensor_list, dim=0)
                visual_feature = modules.pad(visual_feature, landmark_num * tensor_list[0].shape[0])
        elif visual_type == 'face_image':
            src_path = f"{data_path}/{mode}/face_feature/dia{dia_id}_utt{utt_id}/"
            dirListing = os.listdir(src_path)
            visual_feature = torch.load(src_path + f'/{format(dirListing[(len(dirListing))//2])}')
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


def show_ec_sample(model, args, device):
    mode = 'test'
    csv_name = f'./{args.modals}_score:{args.use_score}_query:{args.use_query}_visual:{args.visual_type}_audio:{args.audio_type}'
    
    dataset = natsort.natsorted(os.listdir(f'/home2/dataset/english_conversation/{mode}/speaker_image/'))

    for idx in tqdm(range(len(dataset))):
        # dia = d_u[0]
        # utt = d_u[1]
        # print(f'Test Data: dia{dia}_utt{utt}.mp4')
        data = dataset[idx]
    
        dia = data.split('_')[0][3:]
        utt = data.split('_')[1][3:]
        
        data_path = '/home2/dataset/english_conversation/'
        inputs, historys = load_EC_data(data_path, args, dia, utt, device, mode=mode)
        # for i, k in enumerate(historys):
        #     print(f'{i}-th sentence: {k}')
        
        outputs = model.inference(inputs, greedy=False)
        sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 기존 데이터를 불러오기
        try:
            existing_data = pd.read_csv(f'{csv_name}.csv')
        except FileNotFoundError:
            existing_data = pd.DataFrame()

        # 새로운 데이터 생성
        new_data = pd.DataFrame({'Dialogue_ID': [dia], 'Utterance_ID': [utt], 'output': [sentence]})

        # 기존 데이터와 새로운 데이터를 합치기
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)

        # 누적된 데이터를 CSV 파일로 저장
        combined_data.to_csv(f'{csv_name}.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='MSC', help='select dataset for training')
    parser.add_argument('--act', default='relu', help='type of activation function')
    parser.add_argument('--max_length', type=int, default=60, help='maximum length of utterance')
    parser.add_argument('--history_length', type=int, default=256, help='maximum length of dialogue history')
    parser.add_argument('--audio_pad_size', type=int, default=350, help='time domain padding size for audio')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--audio_type', default='wavlm', help='audio feature extract type |wav2vec2')
    parser.add_argument('--visual_type', default='face_image', help='visual feature type |landmark')
    parser.add_argument('--modals', default='avl', help='modals to fusion')
    parser.add_argument('--use_score', action='store_true', default=False, help='score loss while training or not')
    parser.add_argument('--use_query', action='store_true', default=False, help='use leanable query on training or not')
    parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint for load')
    args = parser.parse_args()
    log_args(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_path = f"../checkpoint/{args.data_name}/{args.modals}_score:{args.use_score}_query:{args.use_query}_visual:{args.visual_type}_audio:{args.audio_type}/{str(args.checkpoint)}_epochs.pt"
    print(f"model: {weight_path}")
    
    
    model = MyArch(args, device, json_directory=None).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    
    # show_meld_sample(model, args, device)
    show_ec_sample(model, args, device)
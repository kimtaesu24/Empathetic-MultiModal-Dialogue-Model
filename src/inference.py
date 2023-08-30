import torch
import sys
import pandas as pd
import ast
import json
import argparse
import torchvision.transforms as transforms
import os
import warnings
import natsort

from transformers import AutoTokenizer, AutoModelForCausalLM
from model.architecture import MyArch
from utils.util import log_args
from model import modules
from facenet_pytorch import InceptionResnetV1
from PIL import Image
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
            visual_feature = torch.load(src_path + dirListing[0])
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
    landmark_num = 7
        
    max_length =args.max_length
    history_length = args.history_length
    audio_pad_size = args.audio_pad_size
    fusion_type = args.fusion_type
    
    FA = pd.read_csv(data_path + f'new_{mode}_FA_matched.csv')
    # fer = pd.read_csv(data_path + f'new_{mode}_FER_matched.csv')
    landmark = pd.read_csv(data_path + f'new_{mode}_LM_matched.csv')
    history_path = data_path + mode
    #########################################################################

    idx = FA[(FA['Dialogue_ID'] == dia_id) & (FA['Utterance_ID'] == utt_id)].index[0]

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
            if len(landmark_set) >= landmark_num:
                visual_feature = landmark_set[[round(i * len(landmark_set)/7) for i in range(7)]] # number of landmark inputs is 7
                visual_feature = torch.flatten(visual_feature)  # [self.landmark_num, landmark_dim] -> [self.landmark_num * landmark_dim]
            else:
                visual_feature = modules.pad(torch.flatten(landmark_set[:]), landmark_num * landmark_set[0].shape[0])
        elif visual_type == 'face_image':
            src_path = f"{data_path}/{mode}/dia{dia_id}/utt{utt_id}"
            dirListing = os.listdir(src_path)
            image_path = src_path+'/{:06d}.jpg'.format((len(dirListing)-3)//2)
            visual_feature = torch.tensor([])

    with open(f"{history_path}/dia{dia_id}/utt{utt_id}/dia{dia_id}_utt{utt_id}_history.json", "r") as json_file:
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
        
def show_ec_sample(model, args, device, csv_name):
    mode = 'test'
    dataset = natsort.natsorted(os.listdir(f'/home2/dataset/english_conversation/{mode}/speaker_image/'))
    total_data = len(dataset)
    print(total_data)

    for idx in range(total_data):
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
    parser.add_argument('--audio_type', default='wavlm', help='audio feature extract type |wav2vec2')
    parser.add_argument('--fusion_type', default='tf_decoder', help='modal fusion method |graph')
    parser.add_argument('--visual_type', default='landmark', help='visual feature type |face_image')
    parser.add_argument('--modals', default='avl', help='modals to fusion')
    parser.add_argument('--use_manager', action='store_true', default=False, help='use reinforcement learning on training or not')
    parser.add_argument('--use_RL', action='store_true', default=False, help='use reinforcement learning on training or not')
    parser.add_argument('--use_query', action='store_true', default=False, help='use reinforcement learning on training or not')
    parser.add_argument('--resume', default=None, help='resume train with checkpoint path or not')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode for wandb')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    log_args(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    weight_path = f"../checkpoint/EC/{args.modals}_manager:False_RL:{args.use_RL}_query:{args.use_query}_visual:{args.visual_type}_audio:{args.audio_type}_alpha:0.001/{str(checkpoint)}_epochs.pt"
    print(f"model: {weight_path}")
    csv_name = f'./{args.modals}_RL:{args.use_RL}_visual:{args.visual_type}_audio:{args.audio_type}_alpha:0.001'
    
    model = MyArch(args, device, json_directory=None).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    
    # show_meld_sample(model, args, device)
    show_ec_sample(model, args, device, csv_name)
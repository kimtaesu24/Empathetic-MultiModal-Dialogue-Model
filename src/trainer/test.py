import torch
import pickle
import datetime

from data_loader.MELD_loader_decoder import MELD_Decoder_Dataset
from data_loader.ECData_loader_decoder import EC_Decoder_Dataset
from torch.utils.data import DataLoader
from model.architecture import MyArch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_metric.coco_eval import calculate_eval_matric

import os


class MyTester:
    def __init__(self, device, data_path):
        self.device = device
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='*', bos_token='#')
        self.mode = 'test'
        print(f'This is running for get [{self.mode}] data metric result!!!!!!')
        
    def train_with_hyper_param(self, args):
        json_directory = f"./{self.mode}_output/{args.data_name}/Dialogpt/"
        try:
            if not os.path.exists(json_directory):
                os.makedirs(json_directory)
        except OSError:
            print("Error: Failed to create the directory.")
            exit()

        model = MyArch(args, self.device, json_directory).to(self.device)
        ##################################################################    
        checkpoint = 100
        weight_path = f"../checkpoint/EC/{args.modals}_manager:False_RL:{args.use_RL}_visual:{args.visual_type}_audio:{args.audio_type}_alpha:0.001/{str(checkpoint)}_epochs.pt"
        print(f"model: {weight_path}")
        model = MyArch(args, self.device, json_directory).to(self.device)
        model.load_state_dict(torch.load(weight_path))
        ###################################################################
        if args.data_name == 'EC':
            test_dataset = EC_Decoder_Dataset(self.data_path, mode=self.mode, device=self.device, args=args)
            
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        test_batch_len = len(test_dataloader)

        total_bleu_1 = 0
        total_bleu_2 = 0
        total_bleu_3 = 0
        total_bleu_4 = 0
        total_meteor = 0
        total_rouge = 0
        with torch.no_grad():
            model.eval()
            for inputs, labels in tqdm(test_dataloader, position=1, leave=False, desc='batch'):
                loss, eval_result = model(inputs, labels, metric_log=True)
                
                total_bleu_1 += eval_result['Bleu_1']
                total_bleu_2 += eval_result['Bleu_2']
                total_bleu_3 += eval_result['Bleu_3']
                total_bleu_4 += eval_result['Bleu_4']

                total_meteor += eval_result['METEOR']
                total_rouge += eval_result['ROUGE_L']

                

            output_metric_dict = {'Bleu-1 (epoch)': total_bleu_1/test_batch_len,
                                  'Bleu-2 (epoch)': total_bleu_2/test_batch_len,
                                  'Bleu-3 (epoch)': total_bleu_3/test_batch_len,
                                  'Bleu-4 (epoch)': total_bleu_4/test_batch_len,
                                  'METEOR (epoch)': total_meteor/test_batch_len,
                                  'ROUGE_L (epoch)': total_rouge/test_batch_len,
                                  }
            print(output_metric_dict)


        return model
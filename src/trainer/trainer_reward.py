import torch
import pickle
import datetime

from data_loader.MELD_loader_decoder import MELD_Decoder_Dataset
# from data_loader.MELD_loader_graph import MELD_Graph_Dataset
# from data_loader.ECData_loader_graph import EC_Graph_Dataset
from data_loader.ECData_loader_decoder import EC_Decoder_Dataset
from torch.utils.data import DataLoader
from model.architecture_reward import MyArch
from tqdm import tqdm
from transformers import AutoTokenizer

import os
import wandb


class MyTrainer:
    def __init__(self, device, data_path):
        self.device = device
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='*', bos_token='#')

    def train_with_hyper_param(self, args):
        # checkpoint save dir create
        directory = f"../checkpoint/{args.data_name}/{args.modals}_manager:{args.use_manager}_RL:{args.use_RL}_visual:{args.visual_type}_audio:{args.audio_type}_alpha:{args.alpha}/"
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Failed to create the directory.")
            exit()
        
        ###########################################################    
        # json_output save dir create
        json_directory = f"./valid_output/{args.data_name}/{args.modals}_manager:{args.use_manager}_RL:{args.use_RL}_visual:{args.visual_type}_audio:{args.audio_type}_alpha:{args.alpha}/"
        try:
            if not os.path.exists(json_directory):
                os.makedirs(json_directory)
        except OSError:
            print("Error: Failed to create the directory.")
            exit()
        ###########################################################

        model = MyArch(args, self.device, json_directory).to(self.device)
        if args.LLM_freeze:
            for parameters in model.gpt_model.parameters():
                parameters.requires_grad = False
            # for name, parameters in model.named_parameters():
            #     print(name, parameters.requires_grad)
            
        if args.resume is not None:
            model.load_state_dict(torch.load(args.resume))
            model.to(self.device)
            print('checkpoint Model has loaded')

        if args.data_name == 'MELD':
            train_dataset = MELD_Decoder_Dataset(self.data_path, mode='train', device=self.device, args=args)
            valid_dataset = MELD_Decoder_Dataset(self.data_path, mode='valid', device=self.device, args=args)
        if args.data_name == 'EC':
            train_dataset = EC_Decoder_Dataset(self.data_path, mode='train', device=self.device, args=args)
            valid_dataset = EC_Decoder_Dataset(self.data_path, mode='valid', device=self.device, args=args)
            
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

        train_batch_len = len(train_dataloader)
        valid_batch_len = len(valid_dataloader)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.decay_rate)

        if not args.debug:  # code for debug mode
            wandb.init(project=f"DialoGen")
            # d = datetime.datetime.today()
            # wandb.run.name = model_name + d.strftime('%c')

        pbar = tqdm(range(args.epochs), position=0, leave=False, desc='epoch')
        for epoch in pbar:
            total_train_loss = 0

            total_valid_loss = 0
            total_valid_bleu_1 = 0
            total_valid_bleu_2 = 0
            total_valid_bleu_3 = 0
            total_valid_bleu_4 = 0
            total_valid_meteor = 0
            total_valid_rouge = 0
            # total_valid_cider = 0
            # total_valid_spice = 0
            
            # training
            model.train()
            prog_bar = tqdm(train_dataloader, position=1,leave=False, desc='batch')
            for i, (inputs, labels) in enumerate(prog_bar):
                optimizer.zero_grad()
                
                loss, eval_result = model(inputs, labels)
                
                prog_bar.set_postfix({'loss': loss.item()})

                loss.backward()
                optimizer.step()

                # log
                total_train_loss += loss.item()

                if not args.debug:  # code for debugging
                    if i % (100//args.batch_size) == 0:
                        wandb.log({'train_loss': loss.item()})
            
            # validation
            with torch.no_grad():
                model.eval()
                metric_log = (epoch+1) % args.metric_at_every == 0
                for inputs, labels in tqdm(valid_dataloader, position=1, leave=False, desc='batch'):
                    
                    loss, eval_result = model(inputs, labels, metric_log=metric_log, epoch=epoch+1)

                    total_valid_loss += loss.item()

                    if metric_log:
                        total_valid_bleu_1 += eval_result['Bleu_1']
                        total_valid_bleu_2 += eval_result['Bleu_2']
                        total_valid_bleu_3 += eval_result['Bleu_3']
                        total_valid_bleu_4 += eval_result['Bleu_4']

                        total_valid_meteor += eval_result['METEOR']
                        total_valid_rouge += eval_result['ROUGE_L']
                        # total_valid_cider += eval_result['CIDEr']
                        # total_valid_spice += eval_result['SPICE']
                    
            if not args.debug:  # code for debugging
                output_loss__dict = {'train_loss (epoch)': total_train_loss/train_batch_len,
                                     'valid_loss (epoch)': total_valid_loss/valid_batch_len
                                     }
                wandb.log(output_loss__dict)
                print(output_loss__dict)
                
                if metric_log:
                    output_metric_dict = {'valid_Bleu-1 (epoch)': total_valid_bleu_1/valid_batch_len,
                                          'valid_Bleu-2 (epoch)': total_valid_bleu_2/valid_batch_len,
                                          'valid_Bleu-3 (epoch)': total_valid_bleu_3/valid_batch_len,
                                          'valid_bleu-4 (epoch)': total_valid_bleu_4/valid_batch_len,
                                          'valid_METEOR (epoch)': total_valid_meteor/valid_batch_len,
                                          'valid_ROUGE_L (epoch)': total_valid_rouge/valid_batch_len,
                                        #   'valid_CIDEr (epoch)': total_valid_cider/valid_batch_len,
                                        #   'valid_SPICE (epoch)': total_valid_spice/valid_batch_len,
                                          }
                    wandb.log(output_metric_dict)
                    print(output_metric_dict)

            # save checkpoint
            if (epoch+1) % args.save_at_every == 0:
                if args.resume is not None:
                    torch.save(model.state_dict(),f"{directory}/resume_{str(epoch+1)}_epochs.pt")
                else:
                    torch.save(model.state_dict(),f"{directory}/{str(epoch+1)}_epochs.pt")
                pbar.write('Checkpoint model has saved at Epoch: {:02} '.format(epoch+1))

            scheduler.step()  # per epochs
            pbar.update()
        pbar.close()

        return model

import torch

from data_loader.MELD_loader_decoder import MELD_Decoder_Dataset
from data_loader.ECData_loader_decoder import EC_Decoder_Dataset
from torch.utils.data import DataLoader
from model.architecture import MyArch
from tqdm import tqdm
from transformers import AutoTokenizer
from eval_metric.coco_eval import calculate_eval_matric

import os
import wandb


class MyTrainer:
    def __init__(self, device, data_path):
        self.device = device
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", pad_token='*', bos_token='#')

    def train_with_hyper_param(self, args):
        # save dir create
        checkpoint_directory = f"../checkpoint/{args.data_name}/{args.modals}_score:{args.use_RL}_query:{args.use_query}_visual:{args.visual_type}_audio:{args.audio_type}/"
        json_directory = f"./valid_output/{args.data_name}/{args.modals}_score:{args.use_RL}_query:{args.use_query}_visual:{args.visual_type}_audio:{args.audio_type}/"
        try:
            if not os.path.exists(checkpoint_directory):
                os.makedirs(checkpoint_directory)
            if not os.path.exists(json_directory):
                os.makedirs(json_directory)
        except OSError:
            print("Error: Failed to create the directory.")
            exit()
                
        model = MyArch(args, self.device, json_directory).to(self.device)
        if args.LLM_freeze:
            for parameters in model.generator_model.parameters():
                parameters.requires_grad = False
            # for name, parameters in model.named_parameters():
            #     print(name, parameters.requires_grad)
            
        if args.resume is not None:
            model.load_state_dict(torch.load(args.resume))
            model.to(self.device)
            print('checkpoint Model has loaded')

        # if args.data_name == 'MELD':
        #     train_dataset = MELD_Decoder_Dataset(self.data_path, mode='train', device=self.device, args=args)
        #     valid_dataset = MELD_Decoder_Dataset(self.data_path, mode='valid', device=self.device, args=args)
        if args.data_name == 'MSC':
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
            r = "'score'" if args.use_score else ''
            q = "'query'" if args.use_query else ''
            wandb.run.name = f"{args.data_name} ~ '{args.modals}' {r} {q} [{args.visual_type}] [{args.audio_type}]"

        pbar = tqdm(range(args.epochs), position=0, leave=False, desc='epoch')
        for epoch in pbar:
            total_train_loss = 0
            total_valid_loss = 0
            
            # training
            model.train()
            prog_bar = tqdm(train_dataloader, position=1,leave=False, desc='batch')
            for i, (inputs, labels) in enumerate(prog_bar):
                optimizer.zero_grad()
                
                loss, eval_result = model(inputs, labels)
                
                prog_bar.set_postfix({'loss': loss.item()})
                
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                if not args.debug:  # code for debug mode
                    if i % (100//args.batch_size) == 0:
                        wandb.log({'train_loss': loss.item()})
            
            # validation
            with torch.no_grad():
                model.eval()
                metric_log = (epoch+1) % args.metric_at_every == 0
                for inputs, labels in tqdm(valid_dataloader, position=1, leave=False, desc='batch'):
                    loss, eval_result = model(inputs, labels, metric_log=metric_log, epoch=epoch+1)
                    total_valid_loss += loss.item()

            # log
            output_loss__dict = {'train_loss (epoch)': total_train_loss/train_batch_len,
                                 'valid_loss (epoch)': total_valid_loss/valid_batch_len
                                 }
            
            if metric_log:
                # load json
                # output_sentence = json['outputs_sentence']
                # ref_sentence = json['ref_sentence']
                eval_result = calculate_eval_matric(outputs_sentence, ref_sentence)
                output_metric_dict = {'valid_Bleu-1 (epoch)': eval_result['Bleu_1'],
                                      'valid_Bleu-2 (epoch)': eval_result['Bleu_2'],
                                      'valid_Bleu-3 (epoch)': eval_result['Bleu_3'],
                                      'valid_bleu-4 (epoch)': eval_result['Bleu_4'],
                                      'valid_METEOR (epoch)': eval_result['METEOR'],
                                      'valid_ROUGE_L (epoch)': eval_result['ROUGE_L'],
                                      'valid_CIDEr (epoch)': eval_result['CIDEr'],
                                      'valid_SPICE (epoch)': eval_result['SPICE'],
                                      }
                    
            if not args.debug:  # code for debug mode
                wandb.log(output_loss__dict)
                print(output_loss__dict)
                if metric_log:
                    wandb.log(output_metric_dict)
                    print(output_metric_dict)
            
            # save checkpoint
            if (epoch+1) % args.save_at_every == 0:
                if args.resume is not None:
                    torch.save(model.state_dict(),f"{checkpoint_directory}/resume_{str(epoch+1)}_epochs.pt")
                else:
                    torch.save(model.state_dict(),f"{checkpoint_directory}/{str(epoch+1)}_epochs.pt")
                pbar.write('Checkpoint model has saved at Epoch: {:02} '.format(epoch+1))

            scheduler.step()  # per epochs
        pbar.close()

        return model

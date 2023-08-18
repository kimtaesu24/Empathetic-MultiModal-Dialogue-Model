import torch
import os
import argparse

from utils.util import set_random_seed, log_args
from trainer.trainer_reward import MyTrainer
from trainer.test_reward import MyTester
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(action='ignore')


def run_mymodel(device, data_path, args):
    trainer = MyTrainer(device=device,
                        data_path=data_path,
                        )
    trainer.train_with_hyper_param(args=args)
    
def test_mymodel(device, data_path, args):
    tester = MyTester(device=device,
                        data_path=data_path,
                        )
    tester.train_with_hyper_param(args=args)

def main(args):
    # Step 0. Initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed(seed=args.seed, device=device)

    # Step 1. Load datasets
    if args.data_name == 'MELD':
        data_path = '/home2/dataset/MELD/'
        # data_path = '/path/to/MELD'
    elif args.data_name == 'MSC':
        data_path = '/home2/dataset/english_conversation/'
        # data_path = '/path/to/MSC'

    # Step 2. Run (train and evaluate) the specified model
    if args.test:
        test_mymodel(device=device,
                    data_path=data_path,
                    args=args)
    else:
        run_mymodel(device=device,
                    data_path=data_path,
                    args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_name', default='MSC', help='select dataset for training')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--act', default='relu', help='type of activation function')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--max_length', type=int, default=60, help='maximum length of utterance')
    parser.add_argument('--history_length', type=int, default=256, help='maximum length of dialogue history')
    parser.add_argument('--audio_pad_size', type=int, default=350, help='time domain padding size for audio')
    parser.add_argument('--alpha', type=float, default=0.001, help='weight for manager component')
    parser.add_argument('--beta', type=float, default=1.0, help='weight for woker component')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--decay_rate', type=float, default=0.98, help='decay rate of learning rate')
    parser.add_argument('--save_at_every', type=int, default=5, help='save checkpoint')
    parser.add_argument('--metric_at_every', type=int, default=5, help='calculate metric scores')
    parser.add_argument('--LLM_freeze', action='store_true', default=False, help='freeze language decoder or not')
    parser.add_argument('--audio_type', default='wavlm', help='audio feature extract type |wav2vec2')
    parser.add_argument('--fusion_type', default='tf_decoder', help='modal fusion method |graph')
    parser.add_argument('--visual_type', default='face_image', help='visual feature type |landmark')
    parser.add_argument('--modals', default='avl', help='modals to fusion')
    parser.add_argument('--use_manager', action='store_true', default=False, help='use reinforcement learning on training or not')
    parser.add_argument('--use_RL', action='store_true', default=False, help='use reward on training or not')
    parser.add_argument('--use_query', action='store_true', default=False, help='use leanable query on training or not')
    parser.add_argument('--resume', default=None, help='resume train with checkpoint path or not')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode for wandb')
    parser.add_argument('--test', action='store_true', default=False, help='get test data result')
    
    args = parser.parse_args()
    
    log_args(args)
    main(args)

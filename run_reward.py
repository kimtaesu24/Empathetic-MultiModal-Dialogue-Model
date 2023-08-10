import os
import fire
import sys
import json
from dotmap import DotMap
from train_reward import main

os.chdir('./src')


def main_wrapper(json_name=''):
    param_path = f'../hyperparameter/param_reward_{json_name}.json'

    with open(param_path, 'r') as in_file:
        param = DotMap(json.load(in_file))

    main(data_name=param.data_name,
         seed=param.seed,
         fps=param.fps,
         epochs=param.epochs,
         act=param.act,
         batch_size=param.batch_size,
         learning_rate=param.learning_rate,
         max_length=param.max_length,
         history_length=param.history_length,
         audio_pad_size=param.audio_pad_size,
         alpha=param.alpha,
         beta=param.beta,
         dropout=param.dropout,
         decay_rate=param.decay_rate,
         save_at_every=param.save_at_every,
         metric_at_every=param.metric_at_every,
         resume=param.resume,
         debug=param.debug,
         LLM_freeze=param.LLM_freeze,
         audio_extract=param.audio_extract,
         fusion_type=param.fusion_type,
         visual_feature=param.visual_feature,
         )


if __name__ == "__main__":
    sys.exit(fire.Fire(main_wrapper))

# import numpy as np

# from torchMoji.api.botmoji import Botmoji

import torch
from transformers import pipeline

def reward_bot_deepmoji(model, conversations):
    '''
    [{'label': 'POSITIVE', 'score': 0.9988656044006348}, {'label': 'NEGATIVE', 'score': 0.9991950392723083}]
    '''
    sentiment_list = model(conversations)
    
    rewards = 0
    for dic in sentiment_list:
        if dic['label'] == 'POSITIVE':
            rewards += 1
    return rewards / len(conversations)
    
if __name__=="__main__":
    sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
    reward_bot_deepmoji(sentiment_analysis, ["I love this!", "i hate you"])
    

# def reward_bot_deepmoji(conversations):
#     """Allocates reward based on deepmoji sentiment of bot utterance"""
#     # Init deepmoji just once
#     if 'botmoji' not in globals():
#         print('Loading deepmoji')
#         global botmoji
#         botmoji = Botmoji()

#     num_convs = len(conversations)
#     episode_len = (len(conversations[0]) - 1) // 2
#     # Flattened bot responses
#     bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    
#     # Run deepmoji
#     reward_multiplier = _get_reward_multiplier()
#     bot_emojis = botmoji.encode_multiple(bot_responses)
#     rewards = np.dot(bot_emojis, reward_multiplier)

#     for i, resp in enumerate(bot_responses):
#         if '<unk>' in bot_responses[i]:
#             rewards[i] = -0.5

#     rewards = rewards.reshape(num_convs, episode_len)
#     return rewards


# def _get_emojis():
#     # All emojis in the order returned by deepmoji
#     EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: :pensive: " + \
#              ":ok_hand: :blush: :heart: :smirk: :grin: :notes: :flushed: " + \
#              ":100: :sleeping: :relieved: :relaxed: :raised_hands: " + \
#              ":two_hearts: :expressionless: :sweat_smile: :pray: " + \
#              ":confused: :kissing_heart: :heartbeat: :neutral_face: " + \
#              ":information_desk_person: :disappointed: :see_no_evil: " + \
#              ":tired_face: :v: :sunglasses: :rage: :thumbsup: :cry: " + \
#              ":sleepy: :yum: :triumph: :hand: :mask: :clap: :eyes: :gun: " + \
#              ":persevere: :smiling_imp: :sweat: :broken_heart: " + \
#              ":yellow_heart: :musical_note: :speak_no_evil: :wink: :skull: " + \
#              ":confounded: :smile: :stuck_out_tongue_winking_eye: :angry: " + \
#              ":no_good: :muscle: :facepunch: :purple_heart: " + \
#              ":sparkling_heart: :blue_heart: :grimacing: :sparkles:"
#     EMOJIS = EMOJIS.split(' ')
#     return EMOJIS


# def _get_emojis_to_rewards_dict():
#     # How detected emojis map to rewards
#     emojis_to_rewards = {
#         # very strongly positive
#         ':kissing_heart:': 1, ':thumbsup:': 1, ':ok_hand:': 1,
#         ':smile:': 1,

#         # strongly positive
#         ':blush:': 0.75, ':wink:': 0.75, ':muscle:': 0.75,
#         ':grin:': 0.75, ':heart_eyes:': 0.75, ':100:': 0.75,

#         # positive
#         ':smirk:': 0.5, ':stuck_out_tongue_winking_eye:': 0.5,
#         ':sunglasses:': 0.5, ':relieved:': 0.5, ':relaxed:': 0.5,
#         ':blue_heart:': 0.5, ':two_hearts:': 0.5, ':heartbeat:': 0.5,
#         ':yellow_heart:': 0.5,

#         # negative
#         ':disappointed:': -0.5, ':eyes:': -0.5,
#         ':expressionless:': -0.5, ':sleeping:': -0.5,
#         ':grimacing:': -0.5,

#         # strongly negative
#         ':neutral_face:': -0.75, ':confused:': -0.75,
#         ':triumph:': -0.75, ':confounded:': -0.75,

#         # very strongly negative
#         ':unamused:': -1, ':angry:': -1,  # removing ':hand:': -1 due to ambiguity
#         ':rage:': -1
#     }
#     return emojis_to_rewards


# def _get_reward_multiplier():
#     EMOJIS = _get_emojis()
#     emojis_to_rewards = _get_emojis_to_rewards_dict()
#     reward_multiplier = np.zeros(len(EMOJIS))
#     for emoji, reward_val in emojis_to_rewards.items():
#         loc = EMOJIS.index(emoji)
#         reward_multiplier[loc] = reward_val
#     return reward_multiplier
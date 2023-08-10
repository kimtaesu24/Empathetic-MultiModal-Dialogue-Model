import numpy as np
import torch

from sentence_transformers import SentenceTransformer



def reward_similarity(model, conversations):
    """Allocates reward for coherence between user input and bot response in
    Universal Sentence Encoder embedding space"""
    
    user_embed = model.encode(conversations[0])
    bot_embed = model.encode(conversations[1])
    similarity = cosine_similarity(user_embed, bot_embed)
    
    # rewards = similarity.reshape(num_convs, episode_len)
    rewards = torch.mean(torch.tensor(similarity))

    return rewards


def cosine_similarity(a, b):
    return np.sum(a * b, axis=1) / np.sqrt((np.sum(a * a, axis=1) * np.sum(b * b, axis=1)))

if __name__=="__main__":
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    conversations = ["This is an example sentence", "Each sentence is converted"]
    reward_similarity(model, conversations)
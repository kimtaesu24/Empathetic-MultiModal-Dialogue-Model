import torch

def reward_question(conversations):
    """Allocates reward for any bot utterance that asks questions."""
    # num_convs = len(conversations)
    # episode_len = (len(conversations[0]) - 1) // 2
    # rewards = np.zeros(num_convs * episode_len)
    rewards = 0

    # Flattened responses
    # bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    question_words = ['who', 'what', 'why', 'where', 'how', 'when']

    for i, resp in enumerate(conversations):
        resp = resp.lower()
        if any(q in resp for q in question_words) and '?' in resp:
            rewards += 1

    # rewards = rewards.reshape(num_convs, episode_len)
    return rewards / len(conversations)
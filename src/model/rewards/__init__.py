import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

from .question import reward_question
from .semantic_similarity import reward_similarity
from .sentiment import reward_bot_deepmoji
from .toxicity import reward_for_toxicity

from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

toxicity_tokenizer = AutoTokenizer.from_pretrained("martin-ha/toxic-comment-model")
toxicity_model = AutoModelForSequenceClassification.from_pretrained("martin-ha/toxic-comment-model")
toxicity_pipeline = pipeline("text-classification", model=toxicity_model, tokenizer=toxicity_tokenizer, device=device)

similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)

sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english", device=device)

def get_rewards(pre_sentence, output_sentence):
    toxicity = reward_for_toxicity(toxicity_pipeline, output_sentence)
    question = reward_question(output_sentence)
    similarity = reward_similarity(similarity_model, [pre_sentence, output_sentence])
    sentiment = reward_bot_deepmoji(sentiment_analysis, output_sentence)
    
    # print("toxicity: ",toxicity)
    # print("question: ",question)
    # print("similarity: ",similarity)
    # print("sentiment: ",sentiment)

    rewards = toxicity + question + similarity + sentiment
    
    # print(rewards)
    return rewards

if __name__=="__main__":
    rewards = get_rewards(pre_sentence=["Yes! You are so smart! I love you."], output_sentence=["I love you too."])
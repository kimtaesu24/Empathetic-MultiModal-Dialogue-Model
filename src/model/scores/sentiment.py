from transformers import pipeline

def score_bot_sentiment(model, conversations):
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
    score_bot_sentiment(sentiment_analysis, ["I love this!", "i hate you"])
    

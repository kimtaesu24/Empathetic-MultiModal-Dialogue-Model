import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

def reward_for_toxicity(toxicity_pipeline, conversation):
    '''
    [{'label': 'toxic', 'score': 0.9607304334640503}, {'label': 'non-toxic', 'score': 0.9967865943908691}]
    '''
    result_list = toxicity_pipeline(conversation)
    
    toxicity = 0
    for dic in result_list:
        if dic['label'] == 'non-toxic':
            toxicity += 1
    return toxicity / len(conversation)


if __name__=="__main__":
    model_path = "martin-ha/toxic-comment-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    print(reward_for_toxicity(model, tokenizer, ['I fucking hate this bull shit world', 'I like this world']))
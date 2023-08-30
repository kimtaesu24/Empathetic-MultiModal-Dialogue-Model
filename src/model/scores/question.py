def score_question(conversations):
    """Allocates reward for any bot utterance that asks questions."""
    scores = 0
    question_words = ['who', 'what', 'why', 'where', 'how', 'when']

    for i, resp in enumerate(conversations):
        resp = resp.lower()
        if any(q in resp for q in question_words) and '?' in resp:
            scores += 1

    return scores / len(conversations)
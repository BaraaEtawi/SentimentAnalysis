from transformers import pipeline, set_seed
import re


generator = pipeline("text-generation", model="openai-community/gpt2")
set_seed(42)

FEW_SHOT_EXAMPLES = [
    ("This film exceeded all my expectations. A masterpiece!", "Positive"),
    ("The pacing was perfect, and I was emotionally invested the whole time.", "Positive"),
    ("An absolute disaster. I wanted to walk out. Complete waste of time.", "Negative"),
]

def build_prompt(review: str, shots: int = 3):
    prompt = "Movie Review Sentiment Analysis:\n"
    prompt += "Analyze each review and respond with ONLY 'Positive' or 'Negative'.\n\n"
    for i in range(min(shots, len(FEW_SHOT_EXAMPLES))):
        text, label = FEW_SHOT_EXAMPLES[i]
        prompt += f"Review: {text}\nSentiment: {label}\n\n"
    prompt += f"Review: {review.strip()}\nSentiment:"
    return prompt

def extract_sentiment(generated: str, original_review: str):
    matches = list(re.finditer(r"Sentiment:\s*(\w+)", generated, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        label = last_match.group(1).strip().lower()
        if label.startswith("pos") or label == "positive":
            return "Positive"
        elif label.startswith("neg") or label == "negative":
            return "Negative"
    
    generated_lower = generated.lower()
    if "positive" in generated_lower:
        return "Positive"
    elif "negative" in generated_lower:
        return "Negative"

    review = original_review.lower()
    
    positives = [
        "amazing", "fantastic", "excellent", "outstanding", "brilliant", "superb", 
        "incredible", "awesome", "perfect", "masterpiece", "great", "good", "love", 
        "wonderful", "best", "favorite", "enjoyed", "loved", "enjoyable", "entertaining"
    ]
    
    negatives = [
        "terrible", "worst", "awful", "horrible", "dreadful", "disaster", "unwatchable", 
        "painful", "waste of time", "wanted to walk out", "bad", "hate", "boring", 
        "waste", "disappointing", "predictable", "confusing", "mess", "annoying", 
        "frustrating", "mediocre", "poor", "weak", "stupid", "okay", "average", 
        "just", "nothing special"
    ]
    
    pos_count = sum(word in review for word in positives)
    neg_count = sum(word in review for word in negatives)
    if pos_count > neg_count:
        return "Positive"
    else:
        return "Negative" 

def classify_review(review: str, shots: int = 3):
    prompt = build_prompt(review, shots)
    result = generator(
        prompt, 
        max_new_tokens=50, 
        temperature=0.7,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )[0]["generated_text"]

    return extract_sentiment(result, review)

from textblob import TextBlob
import random

# Function for positive feedback
def positive_feedback():
    quotes = [
        "Keep up the great work! 🌟",
        "You’re doing amazing! 💪",
        "Stay positive and keep smiling! 😄"
    ]
    return random.choice(quotes)

# Function for negative feedback
def negative_feedback():
    quotes = [
        "Take a deep breath and relax. 😌",
        "It'll get better, just hang in there! 💖",
        "Let’s turn that frown upside down! 😊"
    ]
    return random.choice(quotes)

# Mood-based background color (for fun!)
def get_background_color(mood):
    colors = {
        "happy": "yellow",
        "sad": "blue",
        "angry": "red",
        "neutral": "grey"
    }
    return colors.get(mood, "grey")

# Taking multiple user inputs
def get_user_input():
    print("Enter 'done' when you're finished.")
    text_input = []
    while True:
        text = input("Enter a sentence: ")
        if text.lower() == "done":
            break
        text_input.append(text)
    return text_input

# Analyzing Sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Mapping Sentiments to Mood and Emojis
mood_emojis = {
    "happy": "😃",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐"
}

def get_mood(sentiment_score):
    if sentiment_score > 0.1:
        return "happy"
    elif sentiment_score < -0.1:
        return "sad"
    else:
        return "neutral"

# Main Program
def main():
    user_texts = get_user_input()

    all_sentiment_scores = []
    for sentence in user_texts:
        sentiment_score = analyze_sentiment(sentence)
        all_sentiment_scores.append(sentiment_score)

    average_sentiment = sum(all_sentiment_scores) / len(all_sentiment_scores)
    mood = get_mood(average_sentiment)

    print(f"\nOverall Mood: {mood_emojis[mood]} - {mood.capitalize()}")

    if mood == "happy":
        print(positive_feedback())
    elif mood == "sad":
        print(negative_feedback())

    # Optional: Displaying background color (you can implement this visually in GUI)
    print(f"Background Color for Mood: {get_background_color(mood)}")

if __name__ == "__main__":
    main()


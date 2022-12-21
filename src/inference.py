import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from base import MODEL_PATH
    

def inference(test_text):
    # load tokenizer and model weights
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    online_crime_recognition = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

    # inference
    print(online_crime_recognition(test_text))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("test_text", type=str, help="Text for online crime recognition.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    #test = "это что, как торрент, только только все происходит через программу без всяких сайтов?как узнавать"
    inference(args.test_text)
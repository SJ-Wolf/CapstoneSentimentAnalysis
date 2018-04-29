import requests
import csv
import io
import random


def get_tsv_from_url(url, refresh=True):
    if refresh:
        r = requests.get(url)
        with open('headlines.tsv', 'w', encoding='utf8') as f:
            f.write(r.content.decode('utf8'))
    with open('headlines.tsv', 'r', encoding='utf8') as f:
        text = f.read()
    return text


text = get_tsv_from_url('https://docs.google.com/spreadsheets/d/e/2PACX-1vThJtp6ZLll4fuBWscbDQ49_VAOOTre1qQVqIwTkFAuJQldCMT4MoVx3iCE_hkPv8amS_033LiBq2Lb/pub?gid=0&single=true&output=tsv')


from sentiment_model_demo import get_prediction

our_matches = 0
random_matches = 0
num_headlines = 0

csv_reader = csv.DictReader(io.StringIO(text), delimiter='\t')

for i, row in enumerate(csv_reader):
    our_prediction = get_prediction(row['Headline'])
    assert row['Polarity'] in ('pos', 'neg'), f'Invalid polarity: {row["Polarity"]}'
    if row['Polarity'] == 'pos':
        ground_truth = 1
    else:
        ground_truth = -1
    google_prediction = None
    random_guess = -1 if random.random() < 0.5 else 1
    if our_prediction == ground_truth:
        our_matches += 1
    if random_guess == ground_truth:
        random_matches += 1
    num_headlines += 1
    print(f"Truth: {ground_truth};\tModel: {our_prediction};\tGoogle: {google_prediction};\tHeadline: {row['Headline']}")
    #$print(row['Headline'], row['Polarity'])


print()
print(f"We matched {our_matches} out of {num_headlines} for an accuracy of {100.*our_matches/num_headlines}%")
print(f"Random guessing matched {random_matches} out of {num_headlines} for an accuracy of {100.*random_matches/num_headlines}%")

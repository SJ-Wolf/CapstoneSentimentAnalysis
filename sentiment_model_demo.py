import numpy as np
from keras.models import load_model
import pickle
import sentiment_cnn
import data_helpers
from keras.utils import plot_model


example_text = ['The book was sorta great I guess. Not much to write home about.',
                "I'm not really talking about anything important here.",
                "What an amazing read!",
                "I picked up this book after being inspired by some quotes from Kahlil Gibran. This is a brief text written in a flowery language that masks some simple points. I did not find this book inspiring or insightful, but it was a short read.",
                "I evidently misread the writeup, I thought it was a hardback. It was a cheap paperback. I got it as a present so I couldn't send it back but I'm very dissapointed for the cost!",
                "Miners Fried in Game of Chicken",
                "I strongly dislike this movie. I am disappointed.",
                "Why are global stock markets falling?",
                "Brazil Could Be The Top Performing Market In 2018",
                """Bitcoin miners who've decided to stay in the game amid plunging prices may soon find that the well has run dry.

A 70 percent price drop since the heady days of mid-December has cut profitability to the bone. With the cryptocurrency hitting $6,000 on Tuesday, only the biggest and most efficient can stay above water, but even these are balancing on a knife edge, according to a Gadfly analysis.

Unless you're an outfit running the fastest rigs bought at wholesale prices -- -- 67 percent of all mining power is in the hands of four pools -- chances are you're losing money. The arms race among participants has brought 40 percent more mining power online since Bitcoin prices went above $19,000 on Dec. 18. That's resulted in the rebalancing system built into the digital currency making it 51 percent more difficult to complete a block, according to data from Blockchain.info.

Miners forced to work ever harder for each Bitcoin have shrugged off this escalating requirement for computational power -- up 18-fold in two years -- because a 21-fold price increase over the same period made the cost worth the investment. 

Had Bitcoin stayed at its 50-day moving average of $13,200, then the average miner could expect to print $80 per week in profit at current levels of computation (hash rate) and difficulty. This is based on the very generous assumption that a miner is running Bitmain Technologies Ltd.'s Antminer S9 at 13.5 TH/s (retail price $2,320), one of the most advanced systems available, and the set-up is in China at wholesale prices. 1 Older equipment will have lower returns, and a lot of those mines are still online.

If the price doesn't rise, then the average miner is set to lose $3 per week at current levels. Mining syndicates such as Antpool -- which are probably buying their mines at less than the retail price -- may still be making money, but will be getting returns 90 percent lower than they would at that 50-day moving average.

The only way for miners to return to sustained profits is if Bitcoin prices rise, or some miners turn off the lights, lowering competition. History shows that while the latter is possible, it's unlikely. In fact, those who have plunked down millions of dollars to build their Bitcoin mining operations seem to be playing chicken in the hope that competitors will flinch.

If that happens, they reason, then the bravest miners will be left alone to enjoy the spoils. If it doesn't, then expect a lot to drive off the cliff together.""",
                """As prices plunge, most are losing money. They're all headed for the cliff unless some pull out.""",
                """Why are global stock markets falling?""",
                """Fears of interest rate rises in the US aimed at taming inflation are making markets nervous""",
                """Why are global stock markets falling?
Fears of interest rate rises in the US aimed at taming inflation are making markets nervous

""",
                """STOCKS PLUMMET!! This Could Be The Big One☝️ Buy #Bitcoin Notice Bitcoin and Bitcoin Cash in the Green? #BitcoinRich""",
                """Got a lot of DMs from people wanting to give up on #crypto this past week. My personal opinion is this would be a grave mistake. This market is new, evolving and therefore risky but promising.  Each must determine their own risk tolerance - it's not for everyone.""",
                """When I was a child I loved to draw cartoons and was inspired by the many brilliant works of Disney and other classic animation. It was a lifelong dream that I would one day become an animator myself. How stupid I was, not thinking just how much the would change as I became an adult. Western 2D animation is dead, which is awful enough, but now they've come out with this brainless, mindless mess of pus-yellow-colored human waste. My dreams are dead. Everything I worked so hard at was a waste. A FAILURE. NO ONE in North America has ANY talent anymore. Last year we did have some decent stuff like Zootopia and Finding Dory, but this year it was two leaps forwards, a hundred gigantic leaps back. Now will someone kindly stop the world so I can get off?""",
                """Me and my son Muhammad were shocked how this great and funny 3D movie whose budget is 50 million dollars is so underrated and hated. 
                
                The film centers on Gene, a multi-expressional emoji who lives in a teenager's phone, and who sets out on a journey to become a normal meh emoji like his parents.
                
                My son Muhammad and I laughed and saw this movie too entertaining for kids and adults and the thing that convinces us to see despite the unstoppable attack on it is that it has already reached number 1 in the US Box Office and that many open-minded people did not believe the malicious hatred against a quality movie like this.""",
                """If I was God, and I heard this product was not only being made, not only being promoted, but actually released, then I would invite Satan over to manage the heavens so I could personally eradicate my failure below. This is the sort of product - because this is not truly a movie, as the word "movie" is too suggestive of art - that corporations fawn over. And they did. Believe it or not, three major production studios *fought* to make this happen. 

Of course, they wouldn't spend too much: Minions, a product almost as artless as this one, cost $74 million and runs for 91 minutes. In comparison, The Emoji Movie cost $50 mil and runs for 86 minutes. A 91-minute-long Emoji Movie would cost a mere $52.91 million; Sony cares less than the company that brought us screaming yellow screen- fever. They threw as little as they could at it.

But that's just the math. In order to fully appreciate how apocalyptic this wretched insult to all things sincere is, consider the following; You, the assumed person seeking entertainment, go to the movie theater expecting to take a break or have fun. And while the blatant advertising (Dropbox is an important plot point, there's a pointless scene with Just Dance, apps all around etc.) and banality may be entertainingly laughable, the very same slithery gargoyles that gave you this product get the money. They count their cash, and they think "Hey, that worked." 

So they give you more of the same thing. And more of it. And more, until the idiocy is familiar and the ads the norm. It's already happened to music, with the same notes and lyrics repeated over and over again. Here we have the same situation staring us down, except instead of ass and cash the contents are something else they're trying to sell you. 

The Emoji Movie is an ad that you pay to see. Of course product placement already exists in film. The Lego Movie and Toy Story both feature products as characters, but those films had heart and personality. Here, there is nothing but product placement. Anything resembling humanity is just padding for the next app to appear. How vile for a product that constantly tells you to "express yourself."

Do not watch this thing. Don't bring your kids to see it. Don't watch it ironically. Whatever your beliefs, biases, intentions, anything, do not give companies the thumbs up to feed us mediocre, heartless drivel.""",
"The stock market decline probably isn't over yet",
"Don't Buy Nintendo Co., Ltd Despite the Recent Hype",
"Why are global stock markets falling?",
                "Economic forecast strong for Europe",
"Bitcoin remains in freefall, dives below $7000",
                "Why you shouldn't panic about the market meltdown (yet)",

                ]

with open('vocabulary_inv.pickle', 'rb') as f:
    vocabulary_inv = pickle.load(f)
with open('vocabulary.pickle', 'rb') as f:
    vocabulary = pickle.load(f)

# print(new_test_data)
print('load model')
model = load_model('sentiment_model_5_epochs.h5')
# print('\n')

# print('plot_model')
# plot_model(model, to_file='model.png', show_layer_names=False, show_shapes=False)


if __name__ == "__main__":
    new_test_data = data_helpers.load_additional_data(x_text=example_text, vocabulary=vocabulary, pad_length=592)
    for text, prediction in zip(example_text, model.predict(new_test_data)):
        if prediction >= .5:
            prediction_text = 'Positive'
        else:
            prediction_text = 'Negative'
        print(f"{text[:40]}... {prediction_text} with score of {prediction}.")


def get_prediction(text):
    new_test_data = data_helpers.load_additional_data(x_text=[text], vocabulary=vocabulary, pad_length=592)
    prediction = model.predict(new_test_data)
    if prediction < 0.5:
        return 0
    else:
        return 1

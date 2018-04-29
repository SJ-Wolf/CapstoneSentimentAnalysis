# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:31:01 2017

@author: Daheng
"""
import json
pos_count = 0
neg_count = 0
pos_num = neg_num = 50000
with open('Books_5.json', 'r') as raw:
    with open('rt-polarity-neg.txt', 'w') as neg:
        with open('rt-polarity-pos.txt', 'w') as pos:
            for line in raw:
                rating = json.loads(line)[u'overall']
                if rating in [1,2,4,5]:
                    if rating in [4,5] and pos_count < pos_num:
                        pos_count += 1
                        words = json.loads(line)[u'reviewText']
                        words = words.split()
                        words = words[:400]
                        words = ' '.join(words)
                        pos.write("%s\n" % words)
                    elif rating in [1,2] and neg_count < neg_num:
                        neg_count += 1
                        words = json.loads(line)[u'reviewText']
                        words = words.split()
                        words = words[:400]
                        words = ' '.join(words)
                        neg.write("%s\n" % words)
                    if pos_count == neg_count == pos_num+1:
                        break

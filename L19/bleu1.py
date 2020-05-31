import numpy as np
sum = np.log(6/6) + np.log(4/5) + np.log(2/4) + np.log(1/3)
print('sum=', sum)
#print('sum=', np.exp(sum/4))
bp = np.exp(1-7/6)
print('bp=', bp)
print('bleu=', bp * np.exp(sum/4))

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
# 单个句子的BLEU计算
reference = [['Going', 'to', 'play', 'basketball', 'in', 'the', 'afternoon']]
candidate = ['Going', 'to', 'play', 'basketball', 'the', 'afternoon']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
#score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
#sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)

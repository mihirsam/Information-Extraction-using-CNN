from LineSplit import LineSplit
from WordSplit import WordSplit
from nltk.translate.bleu_score import sentence_bleu

canFile = input("\nEnter Location of Candidate File: ")
refFile = input("\nEnter Location of Reference File: ")

#canFile = './summary_worldwar2.txt'
#refFile = './summary_worldwar2.txt'

canList = WordSplit(LineSplit(canFile))
refList = WordSplit(LineSplit(refFile))

canFinal = []
refFinal = []

for item in canList:
    if item in refList:
        canFinal.append(item)
        refFinal.append(item)
        canList.remove(item)
        refList.remove(item)

canFinal += canList
refFinal += refList

score = sentence_bleu([refFinal], canFinal)
print("BLEU score: ", score)
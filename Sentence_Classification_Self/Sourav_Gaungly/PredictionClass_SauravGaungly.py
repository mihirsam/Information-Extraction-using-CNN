import numpy as np
from doc2vec_gensim import doc2vecs
from LineSplit import LineSplit

name = input("Enter path to txt file : ")
limit = int(input("Enter summary percentage: "))

data = LineSplit(name)
doc2vecs(data)

X = []

from gensim.models.doc2vec import Doc2Vec
import numpy as np

model= Doc2Vec.load("d2v.model")

for i in range(0, len(data)): 
    X.append(model.docvecs[str(i)].reshape(10, 10))    

X = np.asarray(X)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
print(f"X shape : {X.shape}")



from keras.models import load_model

model = load_model('CNN_model_sauravGaungly.h5')
classes = model.predict(X)


abstract = []
history = []
profile = []
career = []
captain = []
offside = []
style = []
president = []



for res in range(len(classes)):
    tmp = list(classes[res])
    
    if tmp.index(max(tmp)) == 0:
        abstract.append(data[res])
    elif tmp.index(max(tmp)) == 1:
        history.append(data[res])
    elif tmp.index(max(tmp)) == 2:
        profile.append(data[res])
    elif tmp.index(max(tmp)) == 3:
        career.append(data[res])
    elif tmp.index(max(tmp)) == 4:
        captain.append(data[res])
    elif tmp.index(max(tmp)) == 5:
        offside.append(data[res])
    elif tmp.index(max(tmp)) == 6:
        style.append(data[res])
    elif tmp.index(max(tmp)) == 7:
        president.append(data[res])
        
        
Final = [profile,
         career,
         captain,
         history,
         abstract,
         offside,
         style,
         president]

Summary = []

for lis in Final:
    if len(lis) != 0:
        for i in range(0, int(len(lis) * (limit/100))+1):
            temp = lis[i].replace("$", ".")
            temp += "."
            Summary.append(temp)
            
            
file = open('./summary_sauravGaungly.txt', 'w+', encoding='utf-8')

for line in Summary:
    file.write(line)
file.close()

print("\nSummary has been saved in 'summary_sauravGaungly.txt'")
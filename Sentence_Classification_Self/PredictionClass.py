import numpy as np
from doc2vec_gensim import doc2vecs
from LineSplit import LineSplit

name = input("Enter path to txt file : ")
limit = int(input("Enter summary percentage: "))
            
data = LineSplit(name)
doc2vecs(data)

X = []
from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v.model")

for i in range(0, len(data)): 
    X.append(model.docvecs[str(i)].reshape(10, 10))    

X = np.asarray(X)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
print(f"X shape : {X.shape}")



from keras.models import load_model

model = load_model('CNN_model_final.h5')
classes = model.predict(X)

abstract = []
advancements = []
asia = []
aftermath = []
casualities = []
europe = []
events = []
mass_slaughter = []

for res in range(len(classes)):
    tmp = list(classes[res])
    
    if tmp.index(max(tmp)) == 0:
        abstract.append(data[res])
    elif tmp.index(max(tmp)) == 1:
        advancements.append(data[res])
    elif tmp.index(max(tmp)) == 2:
        asia.append(data[res])
    elif tmp.index(max(tmp)) == 3:
        aftermath.append(data[res])
    elif tmp.index(max(tmp)) == 4:
        casualities.append(data[res])
    elif tmp.index(max(tmp)) == 5:
        europe.append(data[res])
    elif tmp.index(max(tmp)) == 6:
        events.append(data[res])
    elif tmp.index(max(tmp)) == 7:
        mass_slaughter.append(data[res])
        
Final = [abstract,
        europe,
        asia,
        events,
        mass_slaughter,
        aftermath,
        casualities,
        advancements]

Summary = []

for lis in Final:
    if len(lis) != 0:
        for i in range(0, int(len(lis) * (limit/100))+1):
            Summary.append(lis[i] + ". ")
        
file = open('./summary.txt', 'w+', encoding='utf-8')

for line in Summary:
    file.write(line)
file.close()

print("\nSummary has been saved in 'summary.txt'")
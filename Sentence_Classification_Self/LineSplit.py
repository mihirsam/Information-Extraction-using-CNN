import re

# reading text file
data = open('worldwar.txt', encoding='utf-8', mode='r')
data_lines = data.readlines()

# replacing U.S. with US and spliting lines
cleaned_data = []

for line in data_lines:
        cleaned_data.append(line.replace('U.S.', 'US'))
        
lis = []

for line in cleaned_data:
    if len(set(line)) > 1:
        lis += line.split('.')

# removing unwanted elements and putting cleaned line in cleaned_data
cleaned_data = []
for line in lis:
    if len(line) >= 1:
        if line[0] == " ":
            line = line[1:]
        line = re.sub(r'\[.*?\]', "", line)
        
        if line != '\n':
            cleaned_data.append(line)

# printing each line of cleaned_data
for line in cleaned_data:
    print(line, "\n")
import re

def LineSplit(filename):
    # reading text file
    data = open(filename, encoding='utf-8', mode='r')
    data_lines = data.readlines()

    # replacing U.S. with US and spliting lines
    cleaned_data = []

    for l in range(len(data_lines)):
        flag = 0
        for i in range(len(data_lines[l])):
            try:
                if flag == 0 and data_lines[l][i] == "“":
                    flag = 1
                elif flag == 1 and data_lines[l][i] == "”":
                    flag = 0

                if flag == 1 and data_lines[l][i] == '.':
                    data_lines[l] = data_lines[l][0:i] + "$" + data_lines[l][i+1:]
            except:
                print(data_lines[l])
                break
        cleaned_data.append(data_lines[l])

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

    return cleaned_data
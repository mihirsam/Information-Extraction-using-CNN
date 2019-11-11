def WordSplit(data):
    splitList = []
    for item in data:
        splitList += item.split(' ')
    return splitList
dictionary = {
    0: (2, 9),
    1: (32,8),
    2: (4, 7),
    3: (23, 6),
    5: (4, 5),
    6: (32,4)
}

#print([x for x,y in sorted(dictionary.keys(), key=lambda x: dictionary[x][1][1])])
print(sorted(dictionary.keys(), key=lambda x: dictionary[x], reverse=True))
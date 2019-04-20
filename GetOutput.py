# -*- coding: utf-8 -*-

categories_output = {'alt.atheism' : [],
              'rec.sport.hockey': [],
              'comp.graphics': [],
              'sci.crypt': [],
              'comp.os.ms-windows.misc': [],
              'sci.electronics': [],
              'comp.sys.ibm.pc.hardware': [],
              'sci.med': [],
              'comp.sys.mac.hardware': [],
              'sci.space': [],
              'comp.windows.x': [],
              'soc.religion.christian': [],
              'misc.forsale': [],
              'talk.politics.guns': [],
              'rec.autos': [],
              'talk.politics.mideast': [],
              'rec.motorcycles': [],
              'talk.politics.misc': [],
              'rec.sport.baseball': [],
              'talk.religion.misc': []}

def get_output():
    a = [0 for _ in range(20)]
    b = [list(a) for _ in range(20)]

    position = 0
    while position < 20:
        b[position][position] = 1
        position = position + 1

    position = 0
    for category in categories_output:
        categories_output[category] = b[position]
        position = position + 1

    print(categories_output)
    return categories_output

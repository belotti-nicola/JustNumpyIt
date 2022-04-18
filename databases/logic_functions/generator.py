from math import ceil
from ntpath import join
from operator import xor
from functools import reduce
from os.path import join

fw_test = join('databases','logic_functions','test.csv')
fw_train= join('databases','logic_functions','train.csv')

try:
    n = int(input("insert input dimension... "))
    if n<1:
        raise ValueError
    if n != int(n):
        raise ValueError
except ValueError:
    import sys
    sys.exit('Run the script, inserting a positivae integer != 0.')

n = int(n)    
ntest  = ceil(pow(2,n)/6)
ntrain = pow(2,n) - ntest

lst,lines= [],[]

for i in range(pow(2,n)):
    format_str = f'{i:0{n}b}'
    lst = list(map(int,format_str))
    count_true = lst.count(True)
    lst.append(count_true)
    lines.append([lst])

import random
#needs lst
random.shuffle(lines)

with open(fw_test,'a') as file_train:
    for line in lines[0:ntest]:
        # TypeError: write() argument must be str, not list
        file_train.writ(line)


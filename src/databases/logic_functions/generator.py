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

print(ntest,ntrain)

bin_operands, lst = [], []

for i in range(pow(2,n)):
    format_str = f'{i:0{n}b}'
    bin_operands = list(map(int,format_str))
    bin_result = reduce(lambda x, y: x ^ y, bin_operands)
    lst.append(bin_operands + [bin_result])

from random import shuffle
shuffle(lst)

print(len(lst))
with open(fw_train,'w') as f1:
    for l in lst[0:ntrain:]:
        f1.write(reduce(lambda x,y:str(x)+','+str(y),l)+'\n')

with open(fw_test,'w') as f2:
    for l in lst[ntrain::]:
        f2.write(reduce(lambda x,y:str(x)+','+str(y),l)+'\n')

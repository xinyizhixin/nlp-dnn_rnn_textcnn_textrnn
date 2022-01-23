# -*- coding: utf-8 -*-
import time
now = time.strptime('2016-07-20', '%Y-%m-%d')
print (now)
time.strftime('%Y-%m-%d', now)

import datetime
someDay = datetime.date(1999,2,10)
anotherDay = datetime.date(1999,2,15)
deltaDay = anotherDay - someDay
deltaDay.days

#list用来存储一连串元素的容器，列表用[]来表示，其中元素的类型可不相同。
# 元组类似列表，元组里面的元素也是进行索引计算。列表里面的元素的值可以修改，而元组里面的元素的值不能修改，只能读取。元组的符号是()。
# 集合主要有两个功能，一个功能是进行集合操作，另一个功能是消除重复元素。set()，其中()内可以是列表、字典或字符串，因为字符串是以列表的形式存储的
# Python中的字典dict也叫做关联数组，用大括号{}括起来，在其他语言中也称为map，使用键-值（key-value）存储，具有极快的查找速度，其中key不能重复。

#map/reduce: map将传入的函数依次作用到序列的每个元素，并把结果作为新的list返回；
# reduce把一个函数作用在一个序列[x1, x2, x3...]上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算

myList = [-1, 2, -3, 4, -5, 6, 7]
res=map(abs, myList)
list(res)
# from functools import reduce
#%%
myList = [-1, 2, -3, 4, -5, 6, 7]
from functools import reduce
def powerAdd(a, b):
    return pow(a, 2) + pow(b, 2)
reduce(powerAdd, myList) # 是否是计算平方和？

# filter： filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素
def is_odd(x):
    return x % 3  # 0被判断为False，其它被判断为True
list(filter(is_odd, myList))#去除了3的倍数，3，6

## 1.编码变换

#%%

# utf-8与gbk互相转化需要通过Unicode作为中介
s="我爱北京天安门"  # 默认编码为Unicode
print(s.encode("gbk")) # Unicode可直接转化为gbk
print(s.encode("utf-8")) # Unicode可直接转化为utf-8
print(s.encode("utf-8").decode("utf-8").encode("gb2312"))

f=open('ly.txt','r',encoding='utf-8') # 文件句柄 'w'为创建文件，之前的数据就没了
data=f.read()
print(data)
f.close()
#%%
f=open('test','a',encoding='utf-8') # 文件句柄 'a'为追加文件 append
f.write("\n阿斯达所，\n天安门上太阳升")
f.close()

# 10.浅copy
names=["zhang","Gu","Xiang","Xu"]
names2=names.copy()
print(names,names2) # 此时names2与names指向相同
names[2]="大张"
print(names,names2) # 此时names改变，names2不变

# 12.完整克隆#copy后的list完全独立
import copy
# 浅copy与深copy
'''浅copy与深copy区别就是浅copy只copy一层，而深copy就是完全克隆'''
names = [1, 2, 3, 4, ["zhang", "Gu"], 5]
# names2=copy.copy(names) # 这个跟列表的浅copy一样
names2 = copy.deepcopy(names)  #深copy
names[3] = "斯"
names[4][0] = "张改"
print(names, names2)

# Question 7

### **Question:**
# > **_Write a program which takes 2 digits, X,Y as input and generates a 2-dimensional array. The element value in the i-th row and j-th column of the array should be i _ j.\***
#
# > **_Note: i=0,1.., X-1; j=0,1,¡­Y-1. Suppose the following inputs are given to the program: 3,5_**
#
# > **_Then, the output of the program should be:_**
# [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]
# **Solutions:**
x, y = map(int, input().split(","))
lst = []
for i in range(x):
    tmp = []
    for j in range(y):
        tmp.append(i * j)
    lst.append(tmp)
print(lst)
x, y = map(int, input().split(","))
lst = [[i * j for j in range(y)] for i in range(x)]
print(lst)

# Question 10

### **Question**

# > **_Write a program that accepts a sequence of whitespace separated words as input and prints the words after removing all duplicate words and sorting them alphanumerically._**

# > **_Suppose the following input is supplied to the program:_**
#
# hello world and practice makes perfect and hello world again
#
# > **_Then, the output should be:_**
#
# again and hello makes perfect practice world
# **Solutions:**
word = sorted(
    list(set(input().split()))
)  #  input string splits -> converting into set() to store unique
#  element -> converting into list to be able to apply sort
print(" ".join(word))

# Question 11

### **Question**

# > **_Write a program which accepts a sequence of comma separated 4 digit binary numbers as its input and then check whether they are divisible by 5 or not. The numbers that are divisible by 5 are to be printed in a comma separated sequence._**
#
# > **_Example:_**
#
# 0100,0011,1010,1001
#
# > **_Then the output should be:_**
#
# 1010
data = input().split(",")
data = list(
    filter(lambda i: int(i, 2) % 5 == 0, data)
)  # lambda is an operator that helps to write function of one line
print(",".join(data))
# x='011'
# int(x,2)
# 3

# Question 12
lst = []
for i in range(1000, 3001):
    flag = 1
    for j in str(i):  # every integer number i is converted into string
        if ord(j) % 2!=0:  # ord returns ASCII value and j is every digit of i
            flag = 0  # flag becomes zero if any odd digit found
            break
    if flag == 1:
        lst.append(str(i))  # i is stored in list as string
print(",".join(lst))

# Question 13
word = input()
letter, digit = 0, 0
for i in word:
    if ("a" <= i<= "z") or ("A" <= i <= "Z"):
        letter += 1
    if "0" <= i <= "9":
        digit += 1
print("LETTERS {0}\nDIGITS {1}".format(letter, digit))

word = input()
letter, digits = 0, 0
for i in word:
    if i.isalpha():  # returns True if alphabet
        letter += 1
    elif i.isnumeric():  # returns True if numeric
        digit += 1
print(
    f"LETTERS {letter}\n{digits}"
)  # two different types of formating method is shown in both solution

# Question 15
a = input()
total = int(a) + int(2*a) + int(3*a) + int(4*a)  # N*a=Na, for example  a="23", 2*a="2323",3*a="232323"
print(total)

# Question 18

### **Question:**#
# > **_A website requires the users to input username and password to register. Write a program to check the validity of password input by users._**
#
# > **_Following are the criteria for checking the password:_**
#
# - **_At least 1 letter between [a-z]_**
# - **_At least 1 number between [0-9]_**
# - **_At least 1 letter between [A-Z]_**
# - **_At least 1 character from [$#@]_**
# - **_Minimum length of transaction password: 6_**
# - **_Maximum length of transaction password: 12_**
import re
s = input().split(",")
lst = []
for i in s:
    cnt = 0
    cnt += (6 <= len(i) <= 12)
    cnt += bool(
        re.search("[a-z]", i)
    )  # here re module includes a function re.search() which returns the object information
    cnt += bool(
        re.search("[A-Z]", i)
    )  # of where the pattern string i is matched with any of the [a-z]/[A-z]/[0=9]/[@#$] characters
    cnt += bool(
        re.search("[0-9]", i)
    )  # if not a single match found then returns NONE which converts to False in boolean
    cnt += bool(re.search("[@#$]", i))  # expression otherwise True if found any.
    if cnt == 5:
        lst.append(i)
print(",".join(lst))

# Question 19

### **Question:**
# > **_You are required to write a program to sort the (name, age, score) tuples by ascending order where name is string, age and score are numbers. The tuples are input by console. The sort criteria is:_**
#
# - **_1: Sort based on name_**
# - **_2: Then sort based on age_**
# - **_3: Then sort by score_**
lst=[]
while True:
    s=input().split(',')
    if not s[0]:
        break
    lst.append(tuple(s))
lst.sort(key=lambda x:(x[0],x[1],x[2]))#表示进行排序的优先级依次为x[0],x[1],x[2]
print(lst) # here key is defined by lambda and the data is sorted by element priority 0>1>2 in accending order

# Question 21
### **Question:**
# > **_A robot moves in a plane starting from the original point (0,0).
# The robot can move toward UP, DOWN, LEFT and RIGHT with a given steps. The trace of robot movement is shown as the following:_**
# UP 5
# DOWN 3
# LEFT 3
# RIGHT 2
import math
x, y = 0, 0
while True:
    s = input().split()
    if not s:
        break
    if s[0] == "UP":  # s[0] indicates command
        x -= int(s[1])  # s[1] indicates unit of move
    if s[0] == "DOWN":
        x += int(s[1])
    if s[0] == "LEFT":
        y -= int(s[1])
    if s[0] == "RIGHT":
        y += int(s[1])
        # N**P means N^P
dist = round(
    math.sqrt(x ** 2 + y ** 2)
)  # euclidean distance = square root of (x^2+y^2) and rounding it to nearest integer
print(dist)

# Question 22
# Write a program to compute the frequency of the words from the input. The output should output after sorting the key alphanumerically.
#
# Suppose the following input is supplied to the program:
#
# New to Python or choosing between Python 2 and Python 3? Read Python 2 or Python 3.
#
# Then, the output should be:
# 2:2
# 3.:1
# 3?:1
# New:1
# Python:5
ss = input().split()
dict = {i: ss.count(i) for i in ss}
# sets dictionary as i-> split word & ss.count(i) -> total occurrence of i in ss
dict = sorted(dict.items())
# items() function returns both key & value of dictionary as a list
# and then sorted. The sort by default occurs in order of 1st -> 2nd key
for i in dict:
    print("%s:%d" % (i[0], i[1]))

from collections import Counter
ss = input().split()
ss = Counter(ss)  # returns key & frequency as a dictionary
ss = sorted(ss.items())  # returns as a tuple list
for i in ss:
    print("%s:%d" % (i[0], i[1]))

# Question 42
### **Question:**
# > **_Write a program which can map() and filter() to make a list whose elements are square of even number in [1,2,3,4,5,6,7,8,9,10]._**
def even(x):
    return x % 2 == 0
def squer(x):
    return x * x
li = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
li = map(
    squer, filter(even, li)
)  # first filters number by even number and the apply map() on the resultant elements
print(list(li))
# Question 43
# > **_Write a program which can filter() to make a list whose elements are even number between 1 and 20 (both included)._**
def even(x):
    return x % 2 == 0
evenNumbers = filter(even, range(1, 21))
print(list(evenNumbers))

def divide():
    return 5 / 0

# Question 51
# Question
# Write a function to compute 5/0 and use try/except to catch the exceptions.
try:
    divide()
except ZeroDivisionError as ze:
    print("Why on earth you are dividing a number by ZERO!!")
except:
    print("Any other exception")

# Question 55
# Question
# Write a program which accepts a sequence of words separated by whitespace as input to print the words composed of digits only.
#
# Example: If the following words is given as input to the program:
#
# 2 cats and 3 dogs.
#
# Then, the output of the program should be:
#
# ['2', '3']
import re

email = input()
pattern = "\d+"
ans = re.findall(pattern, email)
print(ans)
#%%
email = input().split()
ans = [word for word in email if word.isdigit()]  # using list comprehension method
print(ans)

# Question 70
# Please write a program to output a random even number between 0 and 10 inclusive using random module and list comprehension.
import random
resp = [i for i in range(0, 11, 2)]
print(random.choice(resp))
# Question 71
import random
resp = random.sample(range(100, 201), 5)
print(resp)

import zlib
t = zlib.compress(b'hello world!hello world!hello world!hello world!')
print(t)
print(zlib.decompress(t))
import datetime
before = datetime.datetime.now()
for i in range(100):
    x = 1 + 1
after = datetime.datetime.now()
execution_time = after - before
print(execution_time.microseconds)

subjects = ["I", "You"]
verbs = ["Play", "Love"]
objects = ["Hockey", "Football"]
for sub in subjects:
    for verb in verbs:
        for obj in objects:
            print("{} {} {}".format(sub, verb, obj))

#Question 84
array = [[[0 for col in range(8)] for col in range(5)] for row in range(3)]
print(array)

li = [12, 24, 35, 70, 88, 120, 155]
li = [li[i] for i in range(len(li)) if i not in (0, 4, 5)]
print(li)

#Question 90
# Question
# Please write a program which count and print the numbers of each character in a string input by console.
# Example: If the following string is given as input to the program:
import string

s = input()
for letter in string.ascii_lowercase:
    cnt = s.count(letter)
    if cnt > 0:
        print("{},{}".format(letter, cnt))

#Question 93
import itertools
print(list(itertools.permutations([1, 2, 3])))


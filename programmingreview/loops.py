# code is suppose to print # 1=5 , fix it
"""
things = str([1, 2, 3, 4, 5])
for thing in things:
    print(thing)


things = [1, 2, 3, 4, 5]
for thing in things:
    print(thing)

"""

# extra proper way to iterate over a list
# this prints index and the value at each index
languages = ["Python", "C++", "Java"]
for i in range(len(languages)):
    print(i, languages[i])

"""
While loops repeat something while a condition is true.
Until loops repeat something while a condition is false.
For loops repeat something for each element of something.

When you use for loops :
For loops over a tuple gives us its items
for loops over a string gives us its characters as strings of length 1
"""
"""
>>> for short_string in 'abc':
...     print(short_string)
...
a
b
c
>>> for item in (1, 2, 3):
...     print(item)
...
1
2
3
"""

"""
Zup function: allows you to loop over multiple lists at the same time.
colors = ["red", "green", "blue", "purple"]
ratios = [0.2, 0.3, 0.1, 0.4]
for color, ratio in zip(colors, ratios):
    print("{}% {}".format(ratio * 100, color))
"""

""" Looping cheatsheat

1) over a single list
for n in numbers:
    print(n)

2) loop over multiple lists
for header, rows in zip(headers, columns):
    print("{}: {}".format(header, ", ".join(rows)))

3) loop over a list while keeping track of the index
for num, line in enumerate(lines):
    print("{0:03d}: {}".format(num, line))
"""

# exercise 1
"""
uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
lowercase = 'abcdefghijklmnopqrstuvwxyz'

Print to console both lists with matching index values.
"""

uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
lowercase = 'abcdefghijklmnopqrstuvwxyz'

for uppercase, lowercase in zip(uppercase, lowercase):
    print("{}: {} ".format(uppercase, ", ".join(lowercase)))

# exercise 3 print to console with indexes and values

uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
lowercase = 'abcdefghijklmnopqrstuvwxyz'

for upper, indexlowerpair in zip(uppercase, enumerate(lowercase, start=1)):
    index, lower = indexlowerpair
    print(index, upper, lower)


"""
Dictionaries consist of key:value pairs.

- variables are stored in a dictorinary with their names as keys, so dictionaries behave like
variables. 
* Dictionaries are not ordered
* setting or getting the value of a key is simple and fast
*dictionaries can't contain the same key twice
* for looping over a dictionary loops over its keys and checking if somthing is in the dictionary
checks if the dictionary contains the key like that. 
- the values() and items() methods return things that behave like lists of values or (key,value) pairs instead
"""
# example
# the following counts hwo many times a word appears in a sentence
sentence = input("Enter a sentence: ")

counts = {}     # {word: count, ...}
for word in sentence.split():
    if word in counts:
        # we have seen this word before
        counts[word] += 1
    else:
        # this is the first time this word occurs
        counts[word] = 1

print()     # display an empty line
for word, count in counts.items():
    if count == 1:
        # "1 times" looks weird
        print(word, "appears once in the sentence")
    else:
        print(word, "appears", count, "times in the sentence")

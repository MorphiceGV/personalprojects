"""
# Exercise 1.1
print("Hello!")
something = input('Enter something:')
print('You entered:' + something)

# exercise 1.2
print('Hello!')
something = input("Enter something: ")
if something == 'hello':
    print("Hello for you too!")
elif something == "hi":
    print('Hi there!')
else:
    print("I don't know what," + something, + "means.")


# exercise 1.3
print("enter your chosen word")
chosenword = input("enter here: ")
# satisfies 1.3
# print(chosenword*10)
# satisfies 1.4
print((chosenword + " ")*10)


#exercise 2.1
print("Hello!")
word1 = input("Enter something: ")
word2 = input("Enter another thing: ")
word3 = input("Enter a third thing: ")
word4 = input("And yet another thing: ")
print("You entered {}, {}, and {}.".format(word1, word2, word3,word4))


# exercise 2.2
message = input("What do you want me to say? ").upper()
print(message, "!!!")
print(message, "!!!")
print(message, "!!!")
"""

#notes
"""
while loops - repeat whiel condition is true
until loops - repeat something until condition is false
for loops - repeat something for each element of something 

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

iterable- something which you can loop through
"""


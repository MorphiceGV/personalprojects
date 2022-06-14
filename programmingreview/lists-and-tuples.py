
"""
Lists are mutable (changeable)
Ingegers, floats, strings and other built in types can't so immutable
[] creates a new list everytime in a loop or just by declaring it 
format: 
things = [1, 2, 3, 4, 5]
"""

"""
Tuples are immutable (can't change after creation)
Can't use append,extend,insert,remove,pop,reverse,sort because they can't be
changed
format: 
thing = (1,2,3)
"""

#Exercises: Lsts and tuples

"""
#1 fix this program
namelist = ('wub_wub', 'RubyPinch', 'go|dfish', 'Nitori')
namelist.append('pb122')
if 'pb122' in namelist:
    print("Now I know pb122!")
"""
# error was wrong syntax brackets
namelist = ['wub_wub', 'RubyPinch', 'go|dfish', 'Nitori']
namelist.append('pb122')
if 'pb122' in namelist:
    print("Now I know pb122!")

# 2 fix this program

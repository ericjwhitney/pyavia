#!/usr/bin/env python3

# Examples of WtDirgraph functions.
# Written by: Eric J. Whitney  Last updated: 8 January 2020

from pyavia import WtDirgraph

wdg = WtDirgraph()
wdg['a':'b'] = 'Here'
print(wdg)  # WtDirgraph({'a': {'b': 'Here'}, 'b': {}})
print(wdg['a':'b'])  # Here
# print(wdg['b':'a'])  # KeyError: 'a'
wdg['a':3.14159] = (22, 7)
wdg['b':'c'] = 'it'
wdg['c':'d'] = 'is.'
path, joined = wdg.trace('a', 'd', op=lambda x, y: ' '.join([x, y]))
print(path, joined)  # ['a', 'b', 'c', 'd'] Here it is.
del wdg['a']
print(wdg)  # WtDirgraph({'b': {'c': 'it'}, 3.14159: {},
# 'c': {'d': 'is.'}, 'd': {}})

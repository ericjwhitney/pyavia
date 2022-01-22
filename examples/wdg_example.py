#!/usr/bin/env python3

# Examples of WtDirgraph functions.
# Written by: Eric J. Whitney  Last updated: 8 January 2020

from pyavia.containers import WtDirGraph

wdg = WtDirGraph()
wdg['a':'b'] = '*** a -> b Connector ***'
print(wdg)  # WtDirgraph({'a': {'b': 'Here'}, 'b': {}})
print(f"'a' -> 'b' Connection?  {wdg['a':'b']}")
# print(f"'b' -> 'a' Connection? {wdg['b':'a']}")  # KeyError: 'a'
wdg['a':3.14159] = (22, 7)
wdg['b':'c'] = '*** b -> c Connector ***'
wdg['c':'d'] = '*** c -> d Connector ***'
path, joined = wdg.trace('a', 'd', op=lambda x, y: ' '.join([x, y]))
print(path, joined)
del wdg['a']
print(wdg)

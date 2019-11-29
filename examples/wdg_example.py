#!/usr/bin/env python3

# Examples of WeightedDirGraph functions.
# Written by: Eric J. Whitney  Last updated: 5 September 2019

from containers import WeightedDirGraph


def main():
    wdg = WeightedDirGraph()
    wdg['a':'b'] = 'Here'
    print(wdg)  # WeightedDirGraph({'a': {'b': 'Here'}, 'b': {}})
    print(wdg['a':'b'])  # Here
    # print(wdg['b':'a'])  # KeyError: 'a'
    wdg['a':3.14159] = (22, 7)
    wdg['b':'c'] = 'it'
    wdg['c':'d'] = 'is.'
    path, joined = wdg.trace('a', 'd', op=lambda x, y: ' '.join([x, y]))
    print(path, joined)  # ['a', 'b', 'c', 'd'] Here it is.
    del wdg['a']
    print(wdg)  # WeightedDirGraph({'b': {'c': 'it'}, 3.14159: {},
    # 'c': {'d': 'is.'}, 'd': {}})


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

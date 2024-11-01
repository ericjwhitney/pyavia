from unittest import TestCase


# ======================================================================

class TestWeightedDirGraph(TestCase):
    def test___init__(self):
        from pyavia.containers import WtDirGraph, wdg_edge
        wdg = WtDirGraph()

        # test basic functions.
        wdg['a':'b'] = 'somevalue'
        self.assertIn('a', wdg)  # Both keys created by link assignment.
        self.assertIn('b', wdg)
        self.assertEqual(wdg['a':'b'], 'somevalue')
        with self.assertRaises(KeyError):
            print(wdg['b':'a'])  # Reverse should not exist.

        # test reverse link.
        wdg['b':'a'] = 1.23
        self.assertEqual(wdg['b':'a'], 1.23)
        self.assertNotEqual(wdg['a':'b'], wdg['b':'a'])

        # test heterogeneous and multiple keys.
        wdg['a':3.14159] = (22, 7)
        wdg[456:True] = 'Yes'
        with self.assertRaises(KeyError):
            wdg[1:2:3] = 4  # Invalid kind of slice index.
        self.assertNotEqual(wdg['a':'b'], wdg['a':3.14159])

        # test key deletion and contains.
        del wdg['a':'b']  # Specific x -> y
        self.assertNotIn(wdg_edge('a', 'b'), wdg)
        self.assertIn(wdg_edge('b', 'a'), wdg)  # Rev. should delete.
        del wdg[456]  # Entire x-key.
        with self.assertRaises(KeyError):
            del wdg[3.14159, 'a']  # Reverse should not exist.
            del wdg[456, True]  # Should already be gone.

        # Can't set path to nowhere.
        with self.assertRaises(KeyError):
            wdg['a':'a'] = 666

        # test construction with forwards dict.
        wdg = WtDirGraph({'a': {'b': 2, 'c': 5}, 'c': {'a': 4}})
        self.assertEqual(wdg['c':'a'], 4)
        with self.assertRaises(KeyError):
            print(wdg['b':'a'])

    def test_trace(self):
        from pyavia.containers import WtDirGraph

        wdg = WtDirGraph()
        wdg[1:2] = 0.5
        wdg[1:3] = 0.2
        wdg[1:4] = 5
        wdg[2:7] = 1
        wdg[2:8] = 3.14159
        wdg[7:-1] = -2

        # Simple paths should be lists with two nodes.
        self.assertEqual(wdg.trace(2, 7), [2, 7])

        # Path to nowhere is invalid.
        with self.assertRaises(KeyError):
            wdg.trace(4, 4)

        # Even simple paths should not be reversible.
        self.assertEqual(wdg.trace(7, 2), None)

        # Check complex forward path.
        path, path_sum = wdg.trace(1, -1, op=lambda x, y: x + y)
        self.assertEqual(path, [1, 2, 7, -1])
        self.assertEqual(path_sum, -0.5)

        # Forward path check (#2 check side-effects of caching).
        path, path_sum = wdg.trace(1, -1, op=lambda x, y: x + y)
        self.assertEqual(path, [1, 2, 7, -1])
        self.assertEqual(path_sum, -0.5)

        # No reverse path should exist.
        path, path_sum = wdg.trace(-1, 1, op=lambda x, y: x + y)
        self.assertIsNone(path)
        self.assertIsNone(path_sum)

        # Add reverse path and confirm it now exists and is different.
        wdg[-1:3] = 5
        wdg[3:1] = 7
        path, path_sum = wdg.trace(-1, 1, op=lambda x, y: x + y)
        self.assertEqual(path, [-1, 3, 1])
        self.assertEqual(path_sum, 12)

        # Forward path check (also checks side-effects rev. cache).
        path, path_sum = wdg.trace(1, -1, op=lambda x, y: x + y)
        self.assertEqual(path, [1, 2, 7, -1])
        self.assertEqual(path_sum, -0.5)

        # Reverse path check (also checks side-effects fwd path cache).
        path, path_sum = wdg.trace(-1, 1, op=lambda x, y: x + y)
        self.assertEqual(path, [-1, 3, 1])
        self.assertEqual(path_sum, 12)

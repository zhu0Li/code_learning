from typing import Optional
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        curr = root
        while curr:
            if curr.left:
                predecessor = nxt = curr.left
                while predecessor.right:
                    predecessor = predecessor.right
                predecessor.right = curr.right
                curr.left = None
                curr.right = nxt
            curr = curr.right

class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        while (root != None):
            if root.left != None:
                most_right = root.left
                while most_right.right != None: most_right = most_right.right
                most_right.right = root.right
                root.right = root.left
                root.left = None
            root = root.right
        return

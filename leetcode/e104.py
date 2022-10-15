# Definition for a binary tree node.
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    class Solution:
        def maxDepth(self, root: Optional[TreeNode]) -> int:
            if root == None:
                return 0
            else:
                l = self.maxDepth(root.left)
                r = self.maxDepth(root.right)
                return max(l, r) + 1

if __name__ == "__main__":
    root = [3,9,20,0,0,15,7]
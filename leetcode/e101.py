from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        def dfs(left,right):
            if not (left or right):
                return True
            if not (left and right):
                return False
            if left.val != right.val:
                return False
            return dfs(left.left, right.right) and dfs(left.right, right.left)
        return dfs(root.left, root.right)

if __name__ == "__main__":
    root = [1, 2, 2, 3, 4, 4, 3]
    print(20<<2)

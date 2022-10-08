from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    pre = -2**31
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        if root == None: return True
        if not self.isValidBST(root.left):
            return False
        if root.val <= self.pre: return False
        self.pre = root.val
        return self.isValidBST(root.right)

if __name__ == "__main__":
    root = [2, 1, 3]
    S = Solution()
    # print(S.isValidBST(root))
    a = None
    if not a:
        print('aaa')
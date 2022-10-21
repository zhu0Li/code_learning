from typing import List, Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
	def buildTree(self, preorder, inorder):
		if not (preorder and inorder):
			return None
		# 根据前序数组的第一个元素，就可以确定根节点
		root = TreeNode(preorder[0])
		# 用preorder[0]去中序数组中查找对应的元素
		mid_idx = inorder.index(preorder[0])
		# 递归的处理前序数组的左边部分和中序数组的左边部分
		# 递归处理前序数组右边部分和中序数组右边部分
		root.left = self.buildTree(preorder[1:mid_idx+1],inorder[:mid_idx])
		root.right = self.buildTree(preorder[mid_idx+1:],inorder[mid_idx+1:])
		return root


class Solution:
	def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
		def myBuildTree(preorder_left: int, preorder_right: int, inorder_left: int, inorder_right: int):
			if preorder_left > preorder_right:
				return None

			# 前序遍历中的第一个节点就是根节点
			preorder_root = preorder_left
			# 在中序遍历中定位根节点
			inorder_root = index[preorder[preorder_root]]

			# 先把根节点建立出来
			root = TreeNode(preorder[preorder_root])
			# 得到左子树中的节点数目
			size_left_subtree = inorder_root - inorder_left
			# 递归地构造左子树，并连接到根节点
			# 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
			root.left = myBuildTree(preorder_left + 1, preorder_left + size_left_subtree, inorder_left,
									inorder_root - 1)
			# 递归地构造右子树，并连接到根节点
			# 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
			root.right = myBuildTree(preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1,
									 inorder_right)
			return root

		n = len(preorder)
		# 构造哈希映射，帮助我们快速定位根节点
		index = {element: i for i, element in enumerate(inorder)}
		return myBuildTree(0, n - 1, 0, n - 1)
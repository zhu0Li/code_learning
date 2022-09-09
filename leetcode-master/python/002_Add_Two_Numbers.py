# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Node_handle():
    def __init__(self):
        self.cur_node = None
	# 查找
    def find(self,node,num,a = 0):
        while node:
            if a == num:
                return node
            a += 1
            node = node.next
	# 增加
    def add(self,data):
        node = ListNode()
        node.val = data
        node.next = self.cur_node
        self.cur_node = node
        return node
	# 打印
    def printNode(self,node):
        while node:
            print ('\nnode: ', node, ' value: ', node.val, ' next: ', node.next)
            node = node.next
	# 删除
    def delete(self,node,num,b = 1):
        if num == 0:
            node = node.next
            return node
        while node and node.next:
            if num == b:
                node.next = node.next.next
            b += 1
            node = node.next
        return node
	# 翻转
    def reverse(self,nodelist):
        list = []
        while nodelist:
            list.append(nodelist.val)
            nodelist = nodelist.next
        result = ListNode()
        result_handle =Node_handle()
        for i in list:
            result = result_handle.add(i)
        return result

class Solution(object):
    # def addTwoNumbers(self, l1, l2):
    #     """
    #     :type l1: ListNode
    #     :type l2: ListNode
    #     :rtype: ListNode
    #     """
    #     last = 0
    #     head = prev = None
    #     while True:
    #         if l2 is None and l1 is None and last == 0:
    #             break
    #         val = last
    #         if l2 is not None:
    #             val += l2.val
    #             l2 = l2.next
    #         if l1 is not None:
    #             val += l1.val
    #             l1 = l1.next
    #         if val >= 10:
    #             val = val % 10
    #             last = 1
    #         else:
    #             last = 0
    #         current = ListNode(val)
    #         if prev is None:
    #             head = current
    #         else:
    #             prev.next = current
    #         prev = current
    #     return head

    def addTwoNumbers(self, l1, l2):
        carry = 0
        # dummy head
        head = curr = ListNode(0)
        while l1 or l2:
            val = carry
            if l1:
                val += l1.val
                l1 = l1.next
            if l2:
                val += l2.val
                l2 = l2.next
            curr.next = ListNode(val % 10)
            curr = curr.next
            carry = int(val / 10)
        if carry > 0:
            curr.next = ListNode(carry)
        return head.next

if __name__ == "__main__":
    l1 = ListNode([1,2,3])
    l2 = ListNode([4,5,6])
    print(l1.val)
    # ListNode = Node_handle()
    # ListNode_2 = Node_handle()
    aa = Solution()
    # # b = Node()
    # a = [1,2,3]
    # b = [4,5,6]
    # for i in a:
    #     l1 = ListNode.add(i)
    # for i in b:
    #     l2 = ListNode_2.add(i)
    # print(ListNode.printNode(l1),ListNode_2.printNode(l2))
    re = aa.addTwoNumbers(l1,l2)
    print(re)
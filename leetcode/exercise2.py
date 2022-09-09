class Node(object):
    def __init__(self):
        self.val = None
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
        node = Node()
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
        result = Node()
        result_handle =Node_handle()
        for i in list:
            result = result_handle.add(i)
        return result
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        if not l1:
            return l2
        elif not l2:
            return l1
        sum = l1.val+l2.val
        if sum > 9:  #相加超过10进位
            return Node(sum-10, self.addTwoNumbers(Node(1, None), self.addTwoNumbers(l1.next, l2.next)))
        else:
            return Node(sum, self.addTwoNumbers(l1.next, l2.next))
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        carry = 0
        # dummy head
        head = curr = Node(0)
        while l1 or l2:
            val = carry
            if l1:
                val += l1.val
                l1 = l1.next
            if l2:
                val += l2.val
                l2 = l2.next
            curr.next = Node(val % 10)
            curr = curr.next
            carry = int(val / 10)
        if carry > 0:
            curr.next = Node(carry)
        return head.next
if __name__ == "__main__":
    # l1 = Node()
    # ListNode_1 = Node_handle()
    # l1_list = [1, 8, 3]
    # for i in l1_list:
    #     l1 = ListNode_1.add(i)
    # ListNode_1.printNode(l1)
    # l1 = ListNode_1.delete(l1,0)
    # ListNode_1.printNode(l1)
    # l1 = ListNode_1.reverse(l1)
    # ListNode_1.printNode(l1)
    # l1 = ListNode_1.find(l1,1)
    # ListNode_1.printNode(l1)
    # a=[1,2,3]
    # b=[4,5,6]
    # lisaa=ListNode()
    # print(lisaa.addTwoNumbers(a,b))
    # re = Node()
    l1 = Node()
    l2 = Node()
    ListNode = Node_handle()
    ListNode_2 = Node_handle()
    aa = Solution()
    # b = Node()
    a = [1,2,3]
    b = [4,5,6]
    for i in a:
        l1 = ListNode.add(i)
    for i in b:
        l2 = ListNode_2.add(i)
    print(ListNode.printNode(l1),ListNode_2.printNode(l2))
    re = aa.addTwoNumbers(l1,l2)
    print(re)

# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        start = head
        end = dummy
        for i in range(n):
            start = start.next
        while start:
            start = start.next
            end = end.next
        end.next = end.next.next
        return dummy.next


if __name__ == "__main__":
    head = ListNode(1)
    for i in range(2,6):
        head = ListNode(head,i)
    print(head.next)
    n = 2
    S = Solution()
    print(S.removeNthFromEnd(head, n))
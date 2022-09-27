from collections import defaultdict
class Solution:
    def CheckPermutation(self, s1: str, s2: str) -> bool:
        # hs1 = defaultdict()
        l1,l2 = len(s1),len(s2)
        if l1 != l2:return False
        if sorted(s1) == sorted(s2): return True
        return False

if __name__ == '__main__':
    s1 = "abc"
    s2 = "bca"
    S = Solution()
    print(S.CheckPermutation(s1, s2))
from typing import List
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 1:
            return list({'()'})
        res = set()
        for i in self.generateParenthesis(n - 1):
            for j in range(len(i) + 2):
                res.add(i[0:j] + '()' + i[j:])
        return list(res)

# class Solution:
#     def generateParenthesis(self, n: int) -> List[str]:
#         res = []
#         def backtrack(S, l, r):
#             if len(S) == 2*n:
#                 res.append(''.join(S))
#                 return
#             if l < n:
#                 S.append('(')
#                 backtrack(S, l+1, r)
#                 S.pop()
#             if r < l:
#                 S.append(')')
#                 backtrack(S, l, r+1)
#                 S.pop()
#         backtrack([],0,0)
#         return res
if __name__ == '__main__':
    n = 3
    S = Solution()
    # a = ['(', ')']*n
    # a.pop(0)
    # print(a)
    print(S.generateParenthesis(n))
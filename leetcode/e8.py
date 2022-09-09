# #普通解法
# class Solution:
#     def myAtoi(self, s: str) -> int:
#         out_min, out_max= -2**31, 2**31-1
#         s=s.strip()
#         # l = len(s)
#         res = 0
#         if not s: return 0
#         i, sign = 0,1
#         if s[i] == '-':
#             sign = -1
#             i+=1
#         elif s[i] == '+':
#             i+=1
#         while i < len(s):
#             if not s[i].isdigit(): break
#             res = res*10 + int(s[i])
#             if sign*res<=out_min:
#                 return out_min
#             elif sign*res>=out_max:
#                 return out_max
#             i=i+1
#         return sign*res

# 自动机 or 确定有限状态机（deterministic finite automaton, DFA）
INT_MAX = 2 ** 31 - 1
INT_MIN = -2 ** 31
class Automaton:
    def __init__(self):
        self.state = 'start'
        self.sign = 1
        self.ans = 0
        self.table = {
            'start': ['start', 'signed', 'in_number', 'end'],
            'signed': ['end', 'end', 'in_number', 'end'],
            'in_number': ['end', 'end', 'in_number', 'end'],
            'end': ['end', 'end', 'end', 'end'],
        }

    def get_col(self, c):
        if c.isspace():
            return 0
        if c == '+' or c == '-':
            return 1
        if c.isdigit():
            return 2
        return 3

    def get(self, c):
        self.state = self.table[self.state][self.get_col(c)]
        if self.state == 'in_number':
            self.ans = self.ans * 10 + int(c)
            self.ans = min(self.ans, INT_MAX) if self.sign == 1 else min(self.ans, -INT_MIN)
        elif self.state == 'signed':
            self.sign = 1 if c == '+' else -1


class Solution:
    def myAtoi(self, str: str) -> int:
        automaton = Automaton()
        for c in str:
            automaton.get(c)
        return automaton.sign * automaton.ans

# # 正则表达
# import re
# class Solution:
#     def myAtoi(self, s: str) -> int:
#         if not s: return 0
#         s = s.lstrip()
#         out_min, out_max = -2 ** 31, 2 ** 31 - 1
#         num_re = re.compile(r'^[\-\+]?\d+')
#         num = num_re.findall(s)
#         print(num)
#         print(*num)
#         res = int(*num)
#         return max(min(res, out_max),out_min)


if __name__ == '__main__':
    s = "-9132"

    S = Solution()
    # print(int('1'))
    print(S.myAtoi(s))
    table = {
        'start': ['start', 'signed', 'in_number', 'end'],
        'signed': ['end', 'end', 'in_number', 'end'],
        'in_number': ['end', 'end', 'in_number', 'end'],
        'end': ['end', 'end', 'end', 'end'],
    }
    aa = 'The edge-guided Feature Pyramid Network (EFPN) model with a side output is proposed for the automatic stratigraphic correlation, which can extract the multi-scale features of well logs.'
    aaa = 'Our deep learning algorithm is capable of capturing the capillary effect and supercritical fluid phenomena.'
    print(
        len(aaa)
    )
    # a = ['12','213','12441']
    # print(*a)
    #
    # print(*a[0])
    # print(int(*a[1]))
    # # print(*a[0][0])
## 贪心哈希
class Solution:
    VALUE_SYMBOLS = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    def intToRoman(self, num: int) -> str:
        res=''
        # while num > 0:
        for n in self.VALUE_SYMBOLS:
            while num >= n[0]:
                res+=n[1]
                num = num - n[0]
                # if num % n[0] != 0:
                #     res+=n[1]*num% n[0]
        return res

# class Solution:
#     def intToRoman(self, num: int) -> str:
#         N = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
#         n = [1000, 900, 500, 400,  100,  90,   50,  40,  10,   9,    5,   4,   1]
#         result = ''
#         for i in range(len(n)):
#             if num >= n[i]:
#                 count = num//n[i]
#                 num -= n[i] * count
#                 result += N[i] * count
#         return result


# ## 暴力匹配
# class Solution():
#     THOUSANDS = ["", "M", "MM", "MMM"]
#     HUNDREDS = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
#     TENS = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
#     ONES = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
#
#     def intToRoman(self, num: int) -> str:
#         return Solution.THOUSANDS[num // 1000] + \
#                Solution.HUNDREDS[num % 1000 // 100] + \
#                Solution.TENS[num % 100 // 10] + \
#                Solution.ONES[num % 10]


if __name__ == "__main__":
    num = [1, 8, 16, 2, 1994, 4, 8, 3, 7]
    # print(19//10,19/10,19%10)
    S = Solution()
    for i in num:
        print(S.intToRoman(i))


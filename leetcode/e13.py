## 模拟
class Solution:
    VALUE_SYMBOLS = [
        (1, "I"),
        (5, "V"),
        (10, "X"),
        (50, "L"),
        (100, "C"),
        (500, "D"),
        (1000, "M"),
    ]
    def romanToInt(self, s: str) -> int:
        res = 0
        mid = 0
        s = s[::-1]
        for i, n in enumerate(s):
            for j, n_ in self.VALUE_SYMBOLS:
                if n==n_:
                    if mid<=j:
                        res += j
                    else:
                        res -=j
                    mid = j
                    break
        return res



if __name__ == '__main__':
    s = "MCMXCIV"
    # print(s.lstrip('I''V'))
    S = Solution()
    print(S.romanToInt(s))
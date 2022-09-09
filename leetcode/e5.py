# def longestPalindrome(s: str):
#         sT = s[::-1]
#         # print(sT)
#         l = 0
#         for j in range(len(s)):
#             for i in range(j+1,len(s)+1):
#                 mid = s[j:i]
#                 if sT.find(mid)>=0 and len(mid)>l:
#                     # print(len(mid)//2)
#                     if len(mid)==1:
#                         l = len(mid)
#                         res = mid
#                     else:
#                         if mid[0:len(mid) // 2] == mid[len(mid)-1 : len(mid) // 2-1:-1] \
#                                 or mid[0:len(mid) // 2] == mid[len(mid)-1:len(mid) // 2:-1]:
#                             l = len(mid)
#                             res = mid
#                     # elif len(mid)>3 and len(mid)%2 == 0 and mid[0:len(mid) // 2] == mid[len(mid) // 2:len(mid)]:
#                     #
#                     #     l = len(mid)
#                     #     res = mid
#         return res


def lP(s: str):
        l = 0
        for j in range(len(s)):
            for i in range(j + 1, len(s) + 1):
                mid = s[j:i]
                if len(mid) > l:
                    if mid[0:len(mid) // 2] == mid[len(mid) - 1: len(mid) // 2 - 1:-1] \
                        or mid[0:len(mid) // 2] == mid[len(mid) - 1:len(mid) // 2:-1] :
                        l = len(mid)
                        res = mid
        return res


def longestPalindrome(s: str) -> str:
        n = len(s)
        if n < 2:
            return s

        max_len = 1
        begin = 0
        # dp[i][j] 表示 s[i..j] 是否是回文串
        dp = [[False] * n for _ in range(n)]
        print(dp)
        for i in range(n):
            dp[i][i] = True
        print(dp)
        # 递推开始
        # 先枚举子串长度
        for L in range(2, n + 1):
            # 枚举左边界，左边界的上限设置可以宽松一些
            for i in range(n):
                # 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                j = L + i - 1
                # 如果右边界越界，就可以退出当前循环
                if j >= n:
                    break

                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]

                # 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if dp[i][j] and j - i + 1 > max_len:
                    max_len = j - i + 1
                    begin = i
        return s[begin:begin + max_len]



if __name__ == '__main__':
    a = "asa"
    print(longestPalindrome(a))
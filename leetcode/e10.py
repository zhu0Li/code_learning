class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        ls, lp = len(s), len(p)
        dp = [[False] * (lp + 1) for _ in range(ls + 1)]
        dp[0][0] = True
        # print(len(dp),len(dp[0]))
        for j in range(1, lp+1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]

        for i in range(1, ls+1):
            for j in range(1, lp+1):
                if p[j - 1] == s[i - 1] or p[j - 1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                else:
                    if p[j - 1] == '*':
                        if p[j - 2] in {s[i - 1], '.'}:
                            dp[i][j] = dp[i][j - 2] or dp[i - 1][j - 2] or dp[i - 1][j]
                        else:
                            dp[i][j] = dp[i][j - 2]
        return dp[ls][lp]
        #     else:
        #         if p[lp-1] == '.':
        #
        #         elif p[lp-1] == '*':
        #
        #         else:
        #             return False

if __name__ == '__main__':
    s = 'aab'
    p = 'c*a*b'
    # print(len([[False] * (10 + 1) for _ in range(5 + 1)]))
    S = Solution()
    print(S.isMatch(s, p))
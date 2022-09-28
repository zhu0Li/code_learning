## 双指针
class Solution:
    def getKthMagicNumber(self, k: int) -> int:
        dp = [0 for _ in range(k)]
        dp[0:4] = [1,3,5,7]
        l,r = 1,1
        i = 4
        while l <= r:
            dp[l] * dp[r]
        return
class Solution:
    def getKthMagicNumber(self, k: int) -> int:
        dp = [1] * (k + 1)
        p3 = p5 = p7 = 1
        for i in range(2, k + 1):
            a, b, c = dp[p3] * 3, dp[p5] * 5, dp[p7] * 7
            v = min(a, b, c)
            dp[i] = v
            if v == a:
                p3 += 1
            if v == b:
                p5 += 1
            if v == c:
                p7 += 1
        print(dp)
        return dp[k]
if __name__ == '__main__':
    k = 11
    S = Solution()
    print(S.getKthMagicNumber(k))
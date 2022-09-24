class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if m==1 or n==1: return 1
        dp = [[1 for _ in range(n)]for _ in range(m)]

        i,j=1,1
        while j<m:
            dp[j][i] = dp[j-1][i] + dp[j][i-1]
            i+=1
            if i >= n:
                i = 1
                j += 1
        return dp[m-1][n-1]

## 空间优化
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if m == 1 or n == 1: return 1
        dp = [1 for _ in range(n)]
        cur = [1 for _ in range(n)]
        i, j = 1, 1
        while j < m:
            # dp[j][i] = dp[j - 1][i] + dp[j][i - 1]
            cur[i] = cur[i-1]+dp[i]
            i += 1
            if i >= n:
                i = 1
                j += 1
                dp = cur[:]
        return cur[-1]

if __name__ == '__main__':
    m = 7
    n = 3
    S = Solution()
    print(S.uniquePaths(m, n))

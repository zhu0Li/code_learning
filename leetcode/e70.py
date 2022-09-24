class Solution:
    def climbStairs(self, n: int) -> int:
        # def dfs(n,index,res):
        #     if index == n:
        #         return res+1
        if n == 1: return 1
        if n == 2: return 2
        dp = [0 for _ in range(n)]
        dp[0:2] = [1,2]
        for i in range(2, n):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]

## 空间优化
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        if n == 2: return 2
        pre,cur=1,2
        for i in range(2, n):
            mid = cur
            cur+=pre
            pre =mid
        return cur

if __name__ == '__main__':
    n = 3
    S = Solution()
    print(S.climbStairs(n))
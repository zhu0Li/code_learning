class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        l1,l2 = len(word1), len(word2)
        dp = [[0]*(l2+1) for _ in range(l1+1)]
        dp[0][:] = [0]+[i+1 for i in range(l2)]
        for i in range(l1):
            dp[i+1][0] = i+1
        for i in range(1,l1+1):
            for j in range(1,l2+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i][j-1], dp[i-1][j]) + 1
        return dp[-1][-1]


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        import functools
        @functools.lru_cache(None)
        def helper(i, j):
            if i == len(word1) or j == len(word2):
                return len(word1) - i + len(word2) - j
            if word1[i] == word2[j]:
                return helper(i + 1, j + 1)
            else:
                inserted = helper(i, j + 1)
                deleted = helper(i + 1, j)
                replaced = helper(i + 1, j + 1)
                return min(inserted, deleted, replaced) + 1

        return helper(0, 0)


if __name__ == '__main__':
    word1 = "horse"
    word2 = "ros"
    S = Solution()
    print(S.minDistance(word1, word2))
from typing import List
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        for i in range(1,m):
            grid[i][0] += grid[i-1][0]
        for i in range(1,n):
            grid[0][i] += grid[0][i-1]
        if m == 1 or n == 1: return grid[-1][-1]
        i, j = 1, 1
        while j < m:
            grid[j][i] += min(grid[j][i-1], grid[j-1][i])
            i += 1
            if i >= n:
                i = 1
                j += 1
        return grid[-1][-1]

if __name__ == '__main__':
    grid = [[0]]
    S = Solution()
    print(S.minPathSum(grid))
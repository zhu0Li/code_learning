#* 动态规划，重点是公式
class Solution:
    def numTrees(self, n: int) -> int:
        res = [1]+[0]*n
        for i in range(1, n+1):
            for j in range(1, i+1):
                res[i] += res[j-1]*res[i-j]
        return res[-1]
from collections import defaultdict
#* 递归

class Solution:
    map_ = defaultdict(int)
    def numTrees(self, n: int) -> int:
        if n==0 or n==1: return 1
        if n in self.map_:
            return self.map_[n]
        cnt=0
        for i in range(1, n+1):
            leftnum = self.numTrees(i-1)
            rightnum = self.numTrees(n-i)
            cnt += leftnum * rightnum
        self.map_[n] = cnt
        return cnt

if __name__ == '__main__':
    n = 3
    S = Solution()
    print(S.numTrees(n))
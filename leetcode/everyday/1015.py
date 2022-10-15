from typing import List
class Solution:
    def buildArray(self, target: List[int], n: int) -> List[str]:
        if n == 0: return []
        mid = [i for i in range(1,n+1)]
        res = []
        for n in mid:
            res.append("Push")
            if n not in target:
                res.append("Pop")
            if n == target[-1]: break
        return res

if __name__ == '__main__':
    target = [2, 3]
    n = 4
    S = Solution()
    print(S.buildArray(target, n))
from typing import List
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        n = len(candidates)
        def dfs(candidates, target, res, res_, i, end):
            if target == 0:
                res.append(res_)
                return
            if target<0:
                return
            for index in range(i, end):
                dfs(candidates, target-candidates[index], res, res_+[candidates[index]], index, end)
        res = []
        res_ = []
        dfs(candidates, target, res, res_, 0, n)
        return res

if __name__ == '__main__':
    candidates = [2,3,6,7]
    target = 7
    S = Solution()
    print(S.combinationSum(candidates, target))
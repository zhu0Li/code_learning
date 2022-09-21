from typing import List
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        def dfs(nums,combination,combinations,size,used,index):
            if index == size:
                combinations.append(combination[:])
                return
            for i in range(size):
                if not used[i]:
                    used[i] = True
                    combination.append(nums[i])

                    dfs(nums,combination ,combinations , size, used,index + 1)

                    used[i] = False
                    combination.pop()
        if len(nums) == 0:
            return []
        used = [False for _ in range(n)]
        combination = []
        combinations = []
        dfs(nums,combination,combinations,n,used,0)
        return combinations

if __name__ == '__main__':
    nums = [1,2,3]
    S = Solution()
    print(S.permute(nums))
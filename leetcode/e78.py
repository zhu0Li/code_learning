from typing import List
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        def backtrack(combination, combinations, size, index, nums):
            if sorted(combination) not in combinations:
                combinations.append(sorted(combination)[:])
            for i in range(index,size):
                if nums[i] not in combination:
                    combination.append(nums[i])
                    backtrack(combination, combinations, size, index + 1, nums)
                    combination.pop()
        combination = []
        backtrack(combination, res, n, 0, nums)
        return res
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        def backtrack(combination, combinations,  index, nums):
            combinations.append(combination[:])
            for i in range(index,len(nums)):
                combination += [nums[i]]
                backtrack(combination, combinations, i + 1, nums)
                combination.pop()
        combination = []
        backtrack(combination, res, 0, nums)
        return res

# class Solution:
#     def subsets(self, nums: List[int]) -> List[List[int]]:
#         idx, n, res, tmp = 0, len(nums), [], []
#         def backtrack(idx, nums, tmp):
#             res.append(tmp)
#             for i in range(idx, n):
#                 backtrack(i + 1, nums, tmp + [nums[i]])
#         backtrack(idx, nums, tmp)
#         return res
#
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        def back_tracking(start, temp):
            res.append(temp[:])
            for i in range(start, len(nums)):
                temp.append(nums[i])
                back_tracking(i+1, temp)
                temp.pop()
        back_tracking(0, [])
        return res

# class Solution:
#     def subsets(self, nums: List[int]) -> List[List[int]]:
#         q=[[]]
#         n=len(nums)
#         for i in range(n):
#             for j in range(len(q)):
#                 q.append(q[j]+[nums[i]])
#         return q
if __name__ == '__main__':
    nums = [4,1,0]
    S = Solution()
    print(S.subsets(nums))
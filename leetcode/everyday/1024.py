from typing import List
class Solution:
    def partitionDisjoint(self, nums: List[int]) -> int:
        n = len(nums)
        # if n == 2: return 1
        l, r, max_index, cur_max = 0, 1, 0, 0
        while r < n:
            if nums[r] > nums[cur_max]: cur_max = r
            if nums[r] < nums[max_index]:
                l=r
                max_index = cur_max
            r += 1
        return l+1

if __name__ == '__main__':
    nums = [3,1,8,4,9,7,12,0,0,12,6,12,6,19,24,90,87,54,92,60,31,59,75,90,20,38,52,51,74,70,86,20,27,91,55,47,54,86,15,
            16,74,32,68,27,19,54,13,22,34,74,76,50,74,97,87,42,58,95,17,93,39,33,22,87,96,90,71,22,48,46,37,18,17,65,54
        ,82,54,29,27,68,53,89,23,12,90,98,42,87,91,23,72,35,14,58,62,79,30,67,44,48]
    nums = [5,0,3,8,6]
    S = Solution()
    print(S.partitionDisjoint(nums))
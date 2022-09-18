from typing import List


class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        i = len(nums) - 2
        while i >0 and nums[i] >= nums[i+1]:
            i-=1

        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j-=1
            nums[i], nums[j] = nums[j], nums[i]

        l, r = i+1, len(nums)-1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
        print(nums)

        # return nums

if __name__ == '__main__':
    nums = [5,4,7,5,3,2]
    S = Solution()
    print(S.nextPermutation(nums))
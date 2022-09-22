from typing import List
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        elif n == 0:
            return 0
        dp = [0]*n
        dp[0] = nums[0]
        # nums = nums[::-1]
        for i in range(1,n):
            dp[i] = max(nums[i],dp[i-1]+nums[i])
        return max(dp)

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        # dp = [0]*n
        mid = nums[0]
        res = max(nums[0],-10**4)
        # dp[0] = nums[0]
        # nums = nums[::-1]
        for i in range(1,n):
            mid = max(nums[i], mid+nums[i])
            if mid > res:
                res = mid
        return res

if __name__ == '__main__':
    nums = [-1,-2]
    S = Solution()
    print(S.maxSubArray(nums))

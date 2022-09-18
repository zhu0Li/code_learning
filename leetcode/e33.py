from typing import List
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # mid = n//2
        l,r = 0,n-1
        while l <= r:
            mid = (l + r + 1) // 2
            if nums[mid] == target:
                return mid
            if nums[l]<nums[mid]:
                if nums[l] <= target < nums[mid]:
                    r = mid -1
                else:
                    l = mid + 1
            else:
                if nums[mid]<target<=nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1

if __name__ == '__main__':
    nums = [3,1]
    target = 1
    S = Solution()
    print(S.search(nums, target))
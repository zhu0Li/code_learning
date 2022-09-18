from typing import List
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        l,r = 0,len(nums) - 1
        while l <= r:
            if nums[l] == target and nums[r] == target:
                return [l,r]
            if nums[l] != target: l+=1
            if nums[r] != target: r-=1
        return [-1,-1]
## 二分
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binarySearch(nums: List[int], target : int) -> int:
            l,r = 0,len(nums)-1
            while l <= r:
                mid = (l + r ) // 2
                # if nums[mid] == target:
                #     return mid
                if nums[mid] < target: l = mid + 1
                else: r = mid - 1
            return l
        a = binarySearch(nums, target)
        b = binarySearch(nums, target+1)
        if a == len(nums) or nums[a] != target:
            return [-1, -1]
        else:
            return [a, b - 1]


if __name__ == '__main__':
    nums = [5,7,7,8,8,10]
    target =8
    S = Solution()
    print(S.searchRange(nums, target))

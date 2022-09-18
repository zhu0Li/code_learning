class Solution:
    def threeSumClosest(self, nums, target: int) -> int:
        if len(nums) == 3:
            return sum(nums)
        res = float("inf")
        nums.sort()
        for i in range(0, len(nums)):
            if i>0 and nums[i] == nums[i-1]:
                continue
            l,r = i+1,len(nums) - 1
            while l < r:
                cur = nums[i] + nums[l] + nums[r]
                if cur == target: return target
                if abs(cur - target) < abs(res - target):
                    res = cur
                if cur > target:
                    r-=1
                else:
                    l+=1
        return res


if __name__ == '__main__':
    nums = [1,1,1,0]
    target = -100
    S = Solution()
    print(S.threeSumClosest(nums, target))
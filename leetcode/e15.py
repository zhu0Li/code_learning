class Solution:
    def threeSum(self, nums):
        if len(nums) < 3:
            return []
        res = []
        nums_sort = self.quicksort(nums)
        for i in range(len(nums_sort)):
            l, r = i+1, len(nums_sort)-1
            if (i > 0 and nums_sort[i] == nums_sort[i - 1]):
                continue
            if nums_sort[i] >0:
                return res
            while l < r:
                if nums_sort[i] + nums_sort[l] + nums_sort[r] == 0:
                    res.append([nums_sort[i], nums_sort[l], nums_sort[r]])
                    while (l < r and nums_sort[l] == nums_sort[l + 1]):
                        l = l + 1
                    while (l < r and nums_sort[r] == nums_sort[r - 1]):
                        r = r - 1
                    l = l + 1
                    r = r - 1
                elif nums_sort[i] + nums_sort[l] + nums_sort[r] < 0:
                    l+=1
                else:
                    r-=1
        return res

    def quicksort(self, nums):
        if len(nums) < 2:
            return nums
        l, r = 0, len(nums)-1
        mid = nums[r//2]
        left, right = [], []
        nums.remove(mid)
        for n in nums:
            if n >=mid:
                right.append(n)
            else:
                left.append(n)
        return self.quicksort(left) + [mid] + self.quicksort(right)

if __name__ == '__main__':
    nums = [-1, 0, 1, 2, -1, -4]
    S = Solution()
    # print(sum(nums))
    print(S.threeSum(nums))
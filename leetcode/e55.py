from typing import List
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1 or nums[0]==n-1: return True
        if nums[0]==0: return False
        for i in range(n-2,-1,-1):
            if nums[i] == 0:
                l,r=0,i-1
                while l <= r:
                    if nums[l] > i-l: break
                    if nums[r] > i-r: break
                    # if l==r and nums[l] <= i-l: return False
                    if l==r or r-l==1:
                        if nums[l] <= i-l:
                            return False
                    l+=1
                    r-=1
        return True

##

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n, rightmost = len(nums), 0
        for i in range(n):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False
if __name__ == '__main__':
    nums = [2,3,1,1,4]
    S = Solution()
    print(S.canJump(nums))
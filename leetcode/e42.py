from typing import List
# from math import min
## 动态规划
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        max_left = [0]*n
        max_right = [0]*n
        res = 0
        for i in range(1,n-1):
            max_left[i] = max(max_left[i - 1], height[i - 1])
        for j in range(n-2,-1,-1):
            max_right[j] = max(max_right[j + 1], height[j + 1])
        for i in range(1,n-1):
            min_ = min(max_left[i], max_right[i])
            if min_ > height[i]:
                res += min_-height[i]
        return res

## 双指针
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        left, right = 0,n-1
        leftMax, rightMax =0,0
        res = 0
        while left<right:
            leftMax = max(leftMax, height[left])
            rightMax = max(rightMax, height[right])
            if height[left]<height[right]:
                res += leftMax - height[left]
                left += 1
            else:
                res += rightMax - height[right]
                right -= 1
        return res

## 栈
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        stack = list()
        n = len(height)

        for i, h in enumerate(height):
            while stack and h > height[stack[-1]]:
                top = stack.pop()
                if not stack:
                    break
                left = stack[-1]
                currWidth = i - left - 1
                currHeight = min(height[left], height[i]) - height[top]
                ans += currWidth * currHeight
            stack.append(i)

        return ans

if __name__ == '__main__':
    height = [0,1,0,2,1,0,1,3,2,1,2,1]
    top = height.pop()
    # print(top)
    # print(height)
    S = Solution()
    print(S.trap(height))
class Solution:
    def maxArea(self, height) -> int:
        i = 0
        j = len(height)
        res = 0
        while i < j:
            S = min(height[i], height[j-1])*(j-i-1)
            if S >= res:
                res = S
            if height[i] < height[j-1]:
                i += 1
            else:
                j -= 1
        return res
if __name__ == "__main__":
    height = [1,8,6,2,5,4,8,3,7]
    S = Solution()
    print(S.maxArea(height))
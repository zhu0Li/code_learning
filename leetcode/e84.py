from typing import List

#! 单调栈
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        size = len(heights)
        res = 0

        stack = []

        for i in range(size):
            while len(stack) > 0 and heights[i] < heights[stack[-1]]:
                cur_height = heights[stack.pop()]

                while len(stack) > 0 and cur_height == heights[stack[-1]]:
                    stack.pop()

                if len(stack) > 0:
                    cur_width = i - stack[-1] - 1
                else:
                    cur_width = i

                res = max(res, cur_height * cur_width)
            stack.append(i)

        while len(stack) > 0 is not None:
            cur_height = heights[stack.pop()]
            while len(stack) > 0 and cur_height == heights[stack[-1]]:
                stack.pop()

            if len(stack) > 0:
                cur_width = size - stack[-1] - 1
            else:
                cur_width = size
            res = max(res, cur_height * cur_width)

        return res

#! 单调栈+哨兵
from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        size = len(heights)
        res = 0
        heights = [0] + heights + [0]
        # 先放入哨兵结点，在循环中就不用做非空判断
        stack = [0]
        size += 2

        for i in range(1, size):
            while heights[i] < heights[stack[-1]]:
                cur_height = heights[stack.pop()]
                cur_width = i - stack[-1] - 1
                res = max(res, cur_height * cur_width)
            stack.append(i)
        return res

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        heights = [0] + heights + [0]
        size = len(heights)
        res = 0
        stack = [0]
        for i in range(1,size):
            while heights[stack[-1]]>heights[i]:
                cur_height = heights[stack.pop()]
                cur_width = i - stack[-1] - 1
                res = max(res, cur_height * cur_width)
            stack.append(i)
        return res

if __name__ == '__main__':
    heights = [2,1,5,6,2,3]
    S = Solution()
    print(S.largestRectangleArea(heights))

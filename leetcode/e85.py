from typing import List
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        res_nums = [0]*len(matrix[0])
        max_s = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if int(matrix[i][j]) == 0:
                    res_nums[j] = 0
                res_nums[j] += int(matrix[i][j])
            max_s = max(max_s, self.get_max_s(res_nums) )   # 求出当前形成柱形的面积，与之前比较取最大的面积
        return max_s

    def get_max_s(self, nums: List[int]):                   # 求柱形最大的面积，利用上题思路
        nums.append(-1)
        stack_index = []
        res = 0
        for i in range(len(nums)):
            while stack_index != [] and nums[i] < nums[stack_index[-1]]:
                h_index = stack_index.pop()
                left = -1 if stack_index == [] else stack_index[-1]
                res = max(res, (i - left -1)*nums[h_index])
            stack_index.append(i)
        return res

class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        res = 0
        array = [0 for _ in range(len(matrix[0]))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if int(matrix[i][j]) == 0:
                    array[j] = 0
                array[j] += int(matrix[i][j])
            res = max(res, self.get_max(array))
        return res
    def get_max(self, nums):
        nums = [0] + nums +[0]
        size = len(nums)
        stack = [0]
        res = 0
        for i in range(1,size):
            while nums[stack[-1]] > nums[i]:
                cur_height = nums[stack.pop()]
                cur_width = i - stack[-1] - 1
                res = max(res, cur_height * cur_width)
            stack.append(i)
        return res




    def get_max_s(self, nums: List[int]):  # 求柱形最大的面积，利用上题思路
        nums.append(-1)
        stack_index = []
        res = 0
        for i in range(len(nums)):
            while stack_index != [] and nums[i] < nums[stack_index[-1]]:
                h_index = stack_index.pop()
                left = -1 if stack_index == [] else stack_index[-1]
                res = max(res, (i - left - 1) * nums[h_index])
            stack_index.append(i)
        return res
if __name__ == '__main__':
    matrix = [["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"],
              ["1", "0", "0", "1", "0"]]
    S = Solution()
    print(S.maximalRectangle(matrix))
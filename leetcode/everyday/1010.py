from math import inf
from typing import List
# 动态规划
class Solution:
    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # f[i][0/1] 表示让 nums1 和 nums2 的前 i 个数严格递增所需操作的最小次数
        # 其中 f[i][0] 不交换 nums1[i] 和 nums2[i]，f[i][1] 交换 nums1[i] 和 nums2[i]
        f = [[inf, inf] for _ in range(n)]
        f[0] = [0, 1]
        for i in range(1, n):
            if nums1[i - 1] < nums1[i] and nums2[i - 1] < nums2[i]:
                f[i][0] = f[i - 1][0]
                f[i][1] = f[i - 1][1] + 1
            if nums2[i - 1] < nums1[i] and nums1[i - 1] < nums2[i]:
                f[i][0] = min(f[i][0], f[i - 1][1])
                f[i][1] = min(f[i][1], f[i - 1][0] + 1)
        return min(f[-1])

# 简化
# class Solution:
#     def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
#         n = len(nums1)
#         a, b = 0, 1
#         for i in range(1, n):
#             at, bt = a, b
#             a = b = n
#             if nums1[i] > nums1[i - 1] and nums2[i] > nums2[i - 1]:
#                 a = min(a, at)
#                 b = min(b, bt + 1)
#             if nums1[i] > nums2[i - 1] and nums2[i] > nums1[i - 1]:
#                 a = min(a, bt)
#                 b = min(b, at + 1)
#         return min(a, b)


if __name__ == '__main__':
    nums1 = [1,3,5,4]
    nums2 = [1,2,3,7]
    S = Solution()
    print(S.minSwap(nums1, nums2))

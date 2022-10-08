from typing import List
class Solution:
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        l1,l2 = len(nums1),len(nums2)
        l,r = 0,l2-1
        res = [0] * l1
        nums1.sort()
        nums2.sort()
        for i in range(0,l1):
            if nums1[i] > nums2[l]:
                res[l] = nums1[i]
                l += 1
            else:
                res[r] = nums1[i]
                r -= 1
        return res


class Solution:
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        n = len(nums1)
        idx1, idx2 = list(range(n)), list(range(n))
        idx1.sort(key=lambda x: nums1[x])
        idx2.sort(key=lambda x: nums2[x])

        ans = [0] * n
        left, right = 0, n - 1
        for i in range(n):
            if nums1[idx1[i]] > nums2[idx2[left]]:
                ans[idx2[left]] = nums1[idx1[i]]
                left += 1
            else:
                ans[idx2[right]] = nums1[idx1[i]]
                right -= 1

        return ans

if __name__ == "__main__":
    nums1 = [12,24,8,32]
    nums2 = [13,25,32,11]
    S = Solution()
    print(S.advantageCount(nums1, nums2))
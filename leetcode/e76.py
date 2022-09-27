##滑动窗口
import collections
from collections import defaultdict
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        '''
        如果hs哈希表中包含ht哈希表中的所有字符，并且对应的个数都不小于ht哈希表中各个字符的个数，那么说明当前的窗口是可行的，可行中的长度最短的滑动窗口就是答案。
        '''
        if len(s)<len(t):
            return ""
        #优化s
        # s = s.lstrip(t)
        #创建哈希表
        hs, ht = defaultdict(int), defaultdict(int)#初始化新加入key的value为0
        for char in t:
            ht[char] += 1
        res = ""
        left, right = 0, 0  # 滑动窗口
        cnt = 0  # 当前窗口中满足ht的字符个数
        while right < len(s):
            hs[s[right]] += 1
            if hs[s[right]] <= ht[s[right]]:  # 必须加入的元素
                cnt += 1  # 遇到了一个新的字符先加进了hs，所以相等的情况cnt也+1
            while left <= right and hs[s[left]] > ht[s[left]]:  # 窗口内元素都符合，开始压缩窗口
                hs[s[left]] -= 1
                left += 1
            if cnt == len(t):
                if not res or right - left + 1 < len(res):  # res为空或者遇到了更短的长度
                    res = s[left:right + 1]
            right += 1
        return res

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        hs, ht=defaultdict(int), defaultdict(int)
        for char in t:
            ht[char] += 1
        res = ""
        l, r = 0, 0
        cnt = 0
        while r<len(s):
            hs[s[r]] += 1
            if hs[s[r]] <= ht[s[r]]:
                cnt+=1
            while l<=r and hs[s[l]] > ht[s[l]]:
                hs[s[l]]-=1
                l+=1
            if cnt == len(t):
                if not res or r-l+1<len(res):
                    res = s[l:r+1]
            r+=1
        return res

## 简化
# class Solution:
#     def minWindow(self, s: str, t: str) -> str:
#         need, missing = collections.Counter(t), len(t)
#         i = start = end = 0
#         for j, c in enumerate(s, 1):
#             missing -= need[c] > 0
#             need[c] -= 1
#             if not missing:
#                 while need[s[i]] < 0:
#                     need[s[i]] += 1
#                     i += 1
#                 if not end or j - i < end - start:
#                     start, end = i, j
#         return s[start:end]

if __name__ == '__main__':
    s = "ADOBECODEBANC"
    t = "ABC"
    S = Solution()
    print(S.minWindow(s, t))
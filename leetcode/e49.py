import collections
from typing import List
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        def dfs(combination, combinations, size, index, words, used):
            if index == size:
                return combinations.append(''.join(combination[:]))
            else:
                for i in range(size):
                    if used[i] == False:
                        used[i] = True
                        combination+=words[i]
                        dfs(combination, combinations, size, index+1, words, used)
                        used[i] = False
                        combination.pop()
        n, size = len(strs),len(strs)
        i = 0
        if n <= 1:
            return [strs]
        # elif n == 2:
        #     if strs[0] == strs[1]:
        #         return [strs]
        #     else:
        #         return [[strs[0]], [strs[1]]]
        res, mid = [],[]
        while i < n :
            mid.append([strs[i]])
            combination = []
            combinations = []
            dfs(combination, combinations,len(strs[i]), 0, strs[i], [False for _ in range(len(strs[i]))])
            j = i + 1
            while j < n:
            # for j in range(i+1,n):
                if strs[j] in combinations:
                    mid[i].append(strs[j])
                    strs = strs[0:j] + strs[j+1:size]
                    n -= 1
                    j-=1
                j += 1
            # res.append(mid)
            i+=1
        return mid


## 排序
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)
        for st in strs:
            key = ''.join(sorted(st))
            mp[key].append(st)
        return list(mp.values())

## 计数
# class Solution:
#     def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
#         mp = collections.defaultdict(list)
#
#         for st in strs:
#             counts = [0] * 26
#             for ch in st:
#                 counts[ord(ch) - ord("a")] += 1
#             # 需要将 list 转换成 tuple 才能进行哈希
#             mp[tuple(counts)].append(st)
#
#         return list(mp.values())
if __name__ == '__main__':
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    S = Solution()
    print(S.groupAnagrams(strs))
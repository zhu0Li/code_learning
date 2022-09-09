## 遍历
# class Solution:
#     def longestCommonPrefix(self, strs) -> str:
#         if len(strs) == 1:
#             return strs[0]
#         l = 200
#         res = ''
#         for i in strs:
#             if len(i) < l:
#                 l = len(i)
#         for i in range(1,l+1):
#             mid = strs[0][0:i]
#             for j in range(1, len(strs)):
#                 if mid != strs[j][0:i]:
#                     break
#                 if j == len(strs) - 1:
#                     res = mid
#         return res

## 横向扫描
class Solution:
    def longestCommonPrefix(self, strs) -> str:
        if len(strs) == 1:
            return strs[0]
        res = self.com_(strs[0], strs[1])
        for i in range(1,len(strs)-1):
            res = self.com_(res,strs[i+1])
            if res == '':
                return res
        return res
    def com_(self,x,y):
        l = min(len(x),len(y))
        res = ''
        for i in range(1,l+1):
            if x[0:i] == y[0:i]:
                res = x[0:i]
            else:
                break
        return res


class Solution:
    def longestCommonPrefix(self, strs) -> str:
        if not strs:
            return ""

        prefix, count = strs[0], len(strs)
        for i in range(1, count):
            prefix = self.lcp(prefix, strs[i])
            if not prefix:
                break

        return prefix

    def lcp(self, str1, str2):
        length, index = min(len(str1), len(str2)), 0
        while index < length and str1[index] == str2[index]:
            index += 1
        return str1[:index]

if __name__ == '__main__':
    strs = ["flower","flow","flight"]
    # print(strs[0][0:1])
    S = Solution()
    print(S.longestCommonPrefix(strs))
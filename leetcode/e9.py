## 基本解法
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        mid = str(x)
        mid = mid[::-1]
        if int(mid) == x:
            return True
        else:
            return False

## 反转一半数字
# class Solution:
#     def isPalindrome(self, x: int) -> bool:
#         if x < 0 or (x > 0 and x % 10 == 0):
#             return False
#         elif x==0:
#             return True
#         res = 0
#         while x >= res:
#             mid = x % 10
#             res = res*10 + mid
#             if x == res:
#                 return True
#             x = x // 10
#             if x == res:
#                 return True
#         return False
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        revertedNumber = 0
        while x > revertedNumber:
            revertedNumber = revertedNumber * 10 + x % 10
            x //= 10
        return x == revertedNumber or x == revertedNumber // 10

if __name__ == '__main__':
    s = 131
    S = Solution()
    print(S.isPalindrome(s))
    # a = 101.
    # print(100%10, 101%10, 100.//10, a/10)
class Solution1:
    def reverse(self, x: int) -> int:
        if x == 0 or x<-2**31 or x>=2**31:
            return 0
        mid = str(x)
        if x >0:
            res = int(mid[::-1])
            if res<-2**31 or res>=2**31:
                return 0
            else:
                return res
        else:
            mid = mid[1:len(mid)]
            res = int('-'+mid[::-1])
            if res<-2**31 or res>=2**31:
                return 0
            else:
                return res
        # return int(res)


class Solution2:
    def reverse(self, x: int) -> int:
        y, res = abs(x), 0
        # 则其数值范围为 [−2^31,  2^31 − 1]
        # boundry = (1 << 31) - 1 if x > 0 else 1 << 31
        while y != 0:
            res = res * 10 + y % 10
            if res<-2**31 or res>=2**31:
                return 0
            y //= 10
        return res if x > 0 else -res

if __name__ == '__main__':
    x = 123
    s = Solution2()
    # print(2e31)
    print(s.reverse(x))
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if len(s) == 1 or numRows == 1:
            return s
        res =["" for i in range(numRows)]
        a = 0
        flag=-1
        for n in s:
            # print(a)
            # print(n)
            res[a]+=n
            if a == 0 or a == numRows - 1: flag = -flag
            a += flag
        return "".join(res)
        # print(res )
if __name__ == "__main__":
    s = "PAYPALISHIRING"
    numRows = 4
    ssss = Solution()
    print(ssss.convert(s, numRows))
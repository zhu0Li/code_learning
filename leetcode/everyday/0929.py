class Solution:
    def isFlipedString(self, s1: str, s2: str) -> bool:
        return len(s1) == len(s2) and s2 in s1 + s1



if __name__ == '__main__':
    s1 = "waterbottle"
    s2 = "erbottlewat"
    S = Solution()
    print(S.isFlipedString(s1, s2))
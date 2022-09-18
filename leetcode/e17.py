from typing import List

# class Solution:
#     def letterCombinations(self, digits: str) -> List[str]:
#         if not digits:
#             return list()
#
#         phoneMap = {
#             "2": "abc",
#             "3": "def",
#             "4": "ghi",
#             "5": "jkl",
#             "6": "mno",
#             "7": "pqrs",
#             "8": "tuv",
#             "9": "wxyz",
#         }
#
#         def backtrack(index: int):
#             if index == len(digits):
#                 combinations.append("".join(combination))
#             else:
#                 digit = digits[index]
#                 for letter in phoneMap[digit]:
#                     combination.append(letter)
#                     backtrack(index + 1)
#                     combination.pop()
#
#         combination = list()
#         combinations = list()
#         backtrack(0)
#         return combinations

class Solution():
    phoneMap = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        combinations = []
        combination = []
        self.backtrack(combination, combinations, 0, len(digits))
        return combinations

    def backtrack(self, combination, combinations, index: int, max_len: int):
        if index == max_len:
            return combinations.append(''.join(combination))
        else:
            for n in self.phoneMap[digits[index]]:
                combination.append(n)
                self.backtrack(combination, combinations, index+1, max_len)
                combination.pop()


if __name__ == '__main__':
    digits = "2398"
    S = Solution()
    # c = list()
    # d = ['a', 'd']
    # c.append("".join(d))
    # print(c)
    print(S.letterCombinations(digits))
    # print
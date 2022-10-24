from typing import List
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res=0
        min_index = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[min_index]:
                res = max((prices[i] - prices[min_index]),res)
            else: min_index = i
        return res

if __name__ == '__main__':
    prices = [7,6,4,3,1]
    S = Solution()
    print(S.maxProfit(prices))
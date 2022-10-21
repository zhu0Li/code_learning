from math import inf

class StockSpanner:

    def __init__(self):
        self.spans = []
        self.index = -1
    def next(self, price: int) -> int:
        res=0
        self.spans.append(price)
        self.index += 1
        i=self.index
        while self.spans[i]<=price and i>=0:
            res+=1
            i-=1
        # for i in range(self.index,-1,-1):
        #     if self.spans[i] <= price:
        #         res += 1
        #     else:
        #         break
        return res
# class StockSpanner:
#     def __init__(self):
#         self.stack = [(-1, inf)]
#         self.idx = -1
#
#     def next(self, price: int) -> int:
#         self.idx += 1
#         while price >= self.stack[-1][1]:
#             self.stack.pop()
#         self.stack.append((self.idx, price))
#         return self.idx - self.stack[-2][0]
if __name__ == "__main__":
    S = StockSpanner()
    print(S.next(100))
    print(S.next(80))
    print(S.next(60))
    print(S.next(70))
    print(S.next(60))
    print(S.next(75))
    print(S.next(85))


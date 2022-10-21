from typing import List
from collections import defaultdict, Counter
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        numbers = []
        res = 0
        mid = 0
        map = defaultdict(int)
        for i in range(len(fruits)):
            if fruits[i] not in numbers:
                numbers.append(fruits[i])
                map[fruits[i]] = i
                if len(numbers) <=2:
                    mid += 1
                else:
                    mid = abs(map[numbers[1]] - map[numbers[0]]) + 1
                    numbers = [fruits[i-1],fruits[i]]
            else:
                map[fruits[i]] = i
                mid += 1
            if mid > res:res = mid
        return res

#滑动窗口
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        cnt = Counter()

        left = ans = 0
        for right, x in enumerate(fruits):
            cnt[x] += 1
            while len(cnt) > 2:
                cnt[fruits[left]] -= 1
                if cnt[fruits[left]] == 0:
                    cnt.pop(fruits[left])
                left += 1
            ans = max(ans, right - left + 1)

        return ans
if __name__ == '__main__':
    fruits = [3,3,3,1,2,1,1,2,3,3,4]
    S = Solution()
    print(S.totalFruit(fruits))
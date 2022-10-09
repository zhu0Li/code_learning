class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        st = [0]
        for c in s:
            if c == '(':
                st.append(0)
            else:
                v = st.pop()
                st[-1] += max(2 * v, 1)
        return st[-1]

class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        st = [0]
        for c in s:
            if c == '(':
                st.append(0)
            else:
                v = st.pop()
                st[-1] += max(2 * v, 1)

        return st[-1]
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        ans = d = 0
        for i, c in enumerate(s):
            if c == '(':
                d += 1
            else:
                d -= 1
                if s[i - 1] == '(':
                    ans += 1 << d
        return ans
if __name__ == '__main__':
    x = "(((())))"
    S = Solution()
    print(S.scoreOfParentheses(x))
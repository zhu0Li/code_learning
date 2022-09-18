class Solution:
    kuohao = {
        '(':')',
        '[':']',
        '{':'}',
        '?':''
    }
    def isValid(self, s: str) -> bool:
        if len(s) % 2 != 0 or len(s) < 2:
            return False
        stack = ['?']
        for l in s:
            if l in self.kuohao:
                stack.append(l)
            else:
                if self.kuohao[stack[-1]] == l:
                    stack.pop(-1)
                else:
                    return False
        if len(stack) > 1:
            return False
        return True


        # while i<n:
        #     if s[i] in self.kuohao and s[i+1] == self.kuohao[s[i]]:
        #         i += 2
        #         continue
        #     elif s[i] in self.kuohao and s[n-1-i] == self.kuohao[s[i]]:
        #         i += 1
        #         if i > n//2:
        #             break
        #         continue
        #     else:
        #         return False
        # return True



if __name__ == '__main__':
    s =  "){"
    S = Solution()
    # a = ['a', 'b', 'c', 'd', 'e', 'f', 'a']
    # print(a.pop())
    print(S.isValid(s))
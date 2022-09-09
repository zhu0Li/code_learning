# def lengthOfLongestSubstring(s: str) -> int:
#     if not s: return 0
#     left = 0
#     lookup = set()
#     n = len(s)
#     max_len = 0
#     cur_len = 0
#     for i in range(n):
#         cur_len += 1
#         while s[i] in lookup:
#             lookup.remove(s[left])
#             left += 1
#             cur_len -= 1
#         if cur_len > max_len: max_len = cur_len
#         lookup.add(s[i])
#     return max_len


def lengthOfLongestSubstring(s: str) -> int:
    k, res= -1, 0
    c_dict = {}  #链表
    for i, c in enumerate(s):
        if c in c_dict and c_dict[c] > k:  # 字符c在字典中 且 上次出现的下标大于当前长度的起始下标
            k = c_dict[c]
            c_dict[c] = i
        else:
            c_dict[c] = i
            res = max(res, i-k)
        print(c_dict)
    return res

def maxlen(s):
    res = -1
    a=-1
    if not s:
        res = 0
    mid = {}
    for i, n in enumerate(s):
        print(n)
        if n in mid and mid[n] > a:
            a = mid[n]
            mid[n] = i
            # a=i
            print(a)
        else:
            mid[n] = i
            res = max(i-a,res)
        print(mid)
    return res

def aaa(s:str):
        res = -1
        mid = {}
        a = -1
        if len(s) == 1:
            return 1
        if not s:
            return 0
        for i,n in enumerate(s):
            if n in mid and mid[n] > a:
                a = mid[n]
                mid[n]=i
            # elif i==len(s)-1:
            #     res = i-a
            else:
                res = max(res,i-a)
                mid[n] = i
            print(res)
        return res
if __name__ == '__main__':
    # a = "tmmzuxt"
    # # for i,n in enumerate(a):
    # #     print(i,n)
    # print(aaa(a))
    a = None or 2
    print(a)
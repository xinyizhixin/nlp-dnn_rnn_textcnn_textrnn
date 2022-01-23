# 剑指 Offer 12. 矩阵中的路径
class Solution:
    def exist(self, board, word) -> bool:
        row_len,col_len,word_len=len(board),len(board[0]),len(word)
        def dfs(r,c,k):#以当前元素为起点，是否有满足条件的路径
            #r,c表示数组board的行列索引，k表示字符串word的索引
            #如果行列索引超出了边界或者r,c处的元素不匹配/已经访问过，返回False
            if not 0<=r<=row_len-1 or not 0<=c<=col_len-1 or board[r][c]!=word[k]:
                return False
            if k==word_len-1:#元素匹配的情况下，如果word字符串全部匹配，返回True
                return True
            board[r][c]=''#已经访问过的元素设置为空字符串
            res=dfs(r+1,c,k+1) or dfs(r-1,c,k+1) or dfs(r,c-1,k+1) or dfs(r,c+1,k+1)#
            board[r][c]=word[k]#所有可能路径中包含i,j的路径全部访问过以后,i,j位置的元素还原
            return res
        for r in range(row_len):#遍历mn个元素寻找满足条件的路径
            for c in range(col_len):
                if dfs(r,c,0):
                    return True
        return False
if __name__ == '__main__':
    A=[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    word="ABCCED"
    s=Solution()
    s.exist(A,word)

# 剑指 Offer 13. 机器人的运动范围
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        #可达解分析：
        # 根据数位和增量公式得知，数位和每逢 进位 突变一次。根据此特点，矩阵中满足数位和的解构成的几何形状形如
        # 多个等腰直角三角形 ，每个三角形的直角顶点位于 0, 10, 20, ...0,10,20,... 等数位和突变的矩阵索引处 。
        # 三角形内的解虽然都满足数位和要求，但由于机器人每步只能走一个单元格，而三角形间不一定是连通的，
        # 因此机器人不一定能到达，称之为 不可达解 ；同理，可到达的解称为 可达解 （本题求此解） 。
        def dfs(r,c,s_r,s_c):#r,c分别为行列索引,s_r,s_c为行列索引的数位之和
            if r>=m or c>=n or s_r+s_c>k or (r,c) in visited:#超边界，不满足条件，已经访问过，返回0
                return 0
            visited.add((r,c))
            #未访问过，上下左右探索路径
            return 1+dfs(r+1,c,s_r+1 if (r+1)%10 else s_r-8,s_c)+dfs(r,c+1,s_r,s_c+1 if (c+1)%10 else s_c-8)
            #若行列索引是10的倍数，行列索引数位之和会-8，否则正常+1
        visited=set()
        return dfs(0,0,0,0)
if __name__ == '__main__':
    m,n,k=2,3,1
    s = Solution()
    s.movingCount(m,n,k)


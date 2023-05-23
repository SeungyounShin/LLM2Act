import heapq as hq
import heapq

def min_cost(cost, m, n):
    dp = [[99999 for _ in range(n+1)] for _ in range(m+1)]
    dp[0][0] = cost[0][0]
    for i in range(0, m+1):
        for j in range(0, n+1):
            if i==0 and j==0:
                continue
            #print(i,j, dp[i][j-1], dp[i-1][j], dp[i-1][j-1] , cost[i][j])
            dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + cost[i][j]
    return dp[m][n]

assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8
assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12
assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16
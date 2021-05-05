# This file is to test the usage of some functions in Python
class Solution(object):
    def shipWithinDays(self, weights, D):
        """
        :type weights: List[int]
        :type D: int
        :rtype: int
        """
        n = len(weights)
        if D == 1:
            return sum(weights)
        if D == n:
            return max(weights)
        loads = []
        for i in range(n-D+2):
            left = sum(weights[:i])
            right = self.shipWithinDays(weights[i:], D-1)
            print(weights[:i], weights[i:])
            print(D-1)
            load = max(left, right)
            print(load)
            loads.append(load)
        return min(loads)


solution = Solution()
weights = [i+1 for i in range(10)]
D = 5
res = solution.shipWithinDays(weights, D)
print(res)
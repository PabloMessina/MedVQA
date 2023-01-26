class UnionFind:
    def __init__(self, n):
        self.setSize = [1] * n
        self.rank =[0] * n
        self.p = [i for i in range(n)]
        self.numSets = n
    def findSet(self, i):
        p = self.p[i]
        if p == i: return i
        p = self.p[i] = self.findSet(p)
        return p
    def isSameSet(self, i, j):
        return self.findSet(i) == self.findSet(j)
    def unionSet(self, i, j):
        if not self.isSameSet(i, j):
            self.numSets -= 1 
            x, y = self.findSet(i), self.findSet(j)            
            if self.rank[x] > self.rank[y]:
                self.p[y] = x
                self.setSize[x] += self.setSize[y]
            else:
                self.p[x] = y
                self.setSize[y] += self.setSize[x]
                if self.rank[x] == self.rank[y]: self.rank[y] += 1
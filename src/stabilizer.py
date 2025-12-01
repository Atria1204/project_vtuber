import math

class PointStabilizer:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev_point = None

    def update(self, point):
        if point is None: return self.prev_point
        
        if self.prev_point is None:
            self.prev_point = point
            return point
        
        # Smoothing
        x = int(self.alpha * point[0] + (1 - self.alpha) * self.prev_point[0])
        y = int(self.alpha * point[1] + (1 - self.alpha) * self.prev_point[1])
        
        self.prev_point = (x, y)
        return (x, y)

class StabilizerManager:
    """
    Mengelola banyak stabilizer sekaligus berdasarkan nama kunci (key).
    """
    def __init__(self, alpha=0.3):
        self.stabs = {}
        self.alpha = alpha

    def get_stable(self, key, new_point):
        if key not in self.stabs:
            self.stabs[key] = PointStabilizer(self.alpha)
        return self.stabs[key].update(new_point)
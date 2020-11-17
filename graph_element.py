class GraphElement:
    def __init__(self, idx, distance):
        self.idx = idx
        self.distance = distance

    def __lt__(self, other):
        if self.distance < other.distance:
            return True
        else:
            return False

    def __le__(self, other):
        if self.distance <= other.distance:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.distance > other.distance:
            return True
        else:
            return False

    def __eq__(self, other):
        if self.distance == other.distance:
            return True
        else:
            return False

    def __ne__(self, other):
        if self.distance != other.distance:
            return True
        else:
            return False

    def __str__(self):
        return "idx: " + str(self.idx) + ",distance: " + str(self.distance)

    # def __eq__(self, other):
    #     return self.distance == other.distance and self.idx == other.idx

class Inf:
    def __eq__(self, other):
        return isinstance(other, Inf)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return self == other

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return self == other


inf = Inf()

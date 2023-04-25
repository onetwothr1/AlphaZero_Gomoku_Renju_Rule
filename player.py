import enum

__all__ = [
    'Player',
]


class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white
    
    def __str__(self):
        return 'Black' if self==Player.black else 'White'
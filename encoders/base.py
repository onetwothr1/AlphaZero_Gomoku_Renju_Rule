class Encoder:
    def encode_board(self, game_state):
        raise NotImplementedError()
        
    def encode_point(self, point):
        raise NotImplementedError()
        
    def decode_point_index(self, index):
        raise NotImplementedError()
        
    def num_points(self):
        raise NotImplementedError()
        
    def shape(self):
        raise NotImplementedError()
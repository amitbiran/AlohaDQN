"""a class that represent a channel in the network"""
class Channel(object):
    def __init__(self,band_width):
        self.band_width = band_width

    def get_band_width(self):
        return self.band_width

class Agent(object):

    def __init__(self,in_channel,out_channel,take_action,id):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.take_action = take_action
        self.id = id
    def get_in_channel(self):
        return self.in_channel

    def get_out_channel(self):
        return self.out_channel


class Strategy:

    def execute(self, vec, json_data_noise):
        raise NotImplementedError


class StrategyMax(Strategy):

    def execute(self, vec, json_data_noise):
        return vec.max() * json_data_noise['LD'] / 3


class StrategyStd(Strategy):

    def execute(self, vec, json_data_noise):
        return vec.std() * json_data_noise['SNR']


class StrategyNone(Strategy):

    def execute(self, vec, json_data_noise):
        return json_data_noise['NONE']


# Strategy selection
strategies = {"LD": StrategyMax, "SNR": StrategyStd, "NONE": StrategyNone}

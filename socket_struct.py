from config import num_workers_total_global
class PushBatch():
    def __init__(self):
        self.batch_si = [1.32134]*(1*42*42*20)
        self.batch_a = [[1.0]*8]*20
        self.batch_adv = [1.0]*20
        self.batch_r = [1.0]*20
        self.batch_features_0 = [[1.0]*256]*20
        self.batch_features_1 = [[1.0]*256]*20
        self.batch_size = 0
        self.id = 0

class ReturnBatch():
    def __init__(self):
        self.batch_si = [[[[1.0]*1]*42]*42]*(20*num_workers_total_global+1)
        self.batch_a = [[1.0]*8]*(20*num_workers_total_global+1)
        self.batch_adv = [1.0]*(20*num_workers_total_global+1)
        self.batch_r = [1.0]*(20*num_workers_total_global+1)
        self.batch_features_0 = [[1.0]*256]*(20*num_workers_total_global+1)
        self.batch_features_1 = [[1.0]*256]*(20*num_workers_total_global+1)
        self.batch_size = 0

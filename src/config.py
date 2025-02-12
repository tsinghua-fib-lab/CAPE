
class Config(object):
    def __init__(self):
        self.output_size = 1
        self.feature_size = 8             
        self.sentence_len = 40     
        self.message_len = 10          
        self.embedding_size = 200
        self.dim_model = 200
        self.hidden_size_GRU = 64
        self.hidden_size = 64
        self.hidden_size_message = 64
        self.item_size = 16
        self.price_size = 5
        self.item_onehot_size = 33
        self.cate_size = 10
        self.path = '/data/mas/yuanyuan/persuasion/data/train_data_enhance_heat.csv'
        self.vector_path = '/data/mas/yuanyuan/persuasion/word_vector/Tencent_AI_Lab.json'
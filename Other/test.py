import torch

# from pytorch_pretrained_bert.modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert
#from modeling_convert import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert, BertModel

# load weight from local file
# download from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
def test1():
    config = BertConfig.from_json_file("./weights/bert-base-uncased-config.json")
    print("Building PyTorch model from configuration: {}".format(str(config)))

    model = BertForPreTraining(config)

    # Load weights from pytorch checkpoint
    model.load_state_dict(torch.load("./weights/bert-base-uncased-pytorch_model.bin"), strict=False)
    # print(model)


# load weight from remote server
def test2():
    model = BertModel.from_pretrained("bert-base-uncased", cache_dir='./cache')
    # print(model)


if __name__ == "__main__":
    test1()

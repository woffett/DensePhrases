from transformers import RobertaTokenizerFast, RobertaModel
#from transformers import AutoTokenizer, AutoConfig
from model_encoder import DPHEncoder


def test():
    print("Testing customized model encoder")
 
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    config =  model.config

    test_string = "Who is Jeff Dean?"
    
    encoder = DPHEncoder(config, tokenizer, model)
    
    inp = tokenizer.encode_plus(test_string, return_token_type_ids=True, return_tensors="pt")
    
    print(inp['input_ids'].size())

    output = encoder.embed_phrase(inp['input_ids'], inp['attention_mask'], inp['token_type_ids'])
    
    print(output.size())

if __name__ == '__main__':
    test()

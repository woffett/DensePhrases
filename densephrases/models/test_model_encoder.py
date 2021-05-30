from transformers import RobertaTokenizerFast, RobertaModel
#from transformers import AutoTokenizer, AutoConfig
from model_encoder import DPHEncoder


def test():
    print("Testing customized model encoder")
 
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    config =  model.config

    test_strings = ["Who is Jeff Dean?", "What predictions does the theory of General Reletivity make?"]
    
    encoder = DPHEncoder(config, tokenizer, model)
    
    inp = tokenizer.encode_plus(test_strings, return_token_type_ids=True, return_tensors="pt")
    
    print(inp['input_ids'][0].size())
    start_vec, end_vec = encoder.embed_phrase(inp['input_ids'], inp['attention_mask'], inp['token_type_ids'])
    query_vec = encoder.embed_query(inp['input_ids'], inp['attention_mask'], inp['token_type_ids'])
    
    #print(start_vec.size())

if __name__ == '__main__':
    test()

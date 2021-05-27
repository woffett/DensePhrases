import json
import argparse
import torch
import os
import random
import numpy as np
import requests
import logging
import math
import copy
import wandb
import string

from time import time
from tqdm import tqdm

from encoder import DensePhrases

class DPHEncoder(DensePhrases):
    def __init__(self, *args, **kwargs):
        super(DPHEncoder, self).__init__(*args, **kwargs)
        self.query_encoder = copy.deepcopy(self.phrase_encoder)

    def embed_phrase(self, input_ids, attention_mask, token_type_ids):
        """ Get phrase embeddings (token-wise) """
        
        outputs = self.phrase_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        """
        sequence_output_s = outputs_s[0]
        sequence_output_e = outputs_s[0]
        start = sequence_output_s[:,:,:]
        end = sequence_output_e[:,:,:]

        return start, end
        """
        return outputs[0]

    def embed_query(self, input_ids_, attention_mask_, token_type_ids_):
        """ Get query start/end embeddings """

        output = self.query_encoder(
            input_ids_,
            attention_mask=attention_mask_,
            token_type_ids=token_type_ids_,
            )
        #print("query encoder output", output)

        return output[0]

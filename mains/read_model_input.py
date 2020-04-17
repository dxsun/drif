# Made this file to read "model_input.pickle" to produce sequence of training images
import train.train_supervised
import pickle
import learning
import sys
sys.path.insert(0, "/storage/dxsun/drif")
from data_io.instructions import get_all_instructions, get_word_to_token_map


inp = pickle.load(open("model_input2.pickle", 'rb'))

# _, _, _, corpus = get_all_instructions()

# token2term, word2token = get_word_to_token_map(corpus)

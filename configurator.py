import argparse

" comment line configuration for model"

# cml argument parser
parser =  argparse.ArgumentParser()

# optional cml arguments
parser.add_argument("-v", "--verbose", action= argparse.BooleanOptionalAction, help= "wanna see updated model arguments")
parser.add_argument("--n_embd", type= int, help = "Token embedding dimention")
parser.add_argument("--block_size", type= int, help= 'Sequence/Context length')
parser.add_argument("--n_heads", type = int, help = "Number of Heads in Attention")
parser.add_argument("--n_layers", type= int, help= "Number of layer in transformer")
parser.add_argument("--score_func", type = str, help= "Score fucnction used to get affinity gate scores")  
parser.add_argument("--inter_dim", type= int, help= "Hidden state dim for MLP") 
parser.add_argument("--experts_dim", type= int, help= "Hidden state dim for MOE") 
parser.add_argument("--n_dense_layers", type= int, help= "Number of dense layer in a model")
parser.add_argument("-b", "--bias", action= argparse.BooleanOptionalAction, help= "Want to include bias term")


# parse the cml argument and remove None values
args =  parser.parse_args()
cml_updated_args =  {k:v for k, v in vars(args).items() if v is not None}

    
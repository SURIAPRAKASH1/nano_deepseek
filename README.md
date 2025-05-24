# _NaNo DeepSeek_:

_This repo contains nano version of DeepSeek LLM's that are implemented from scartch. For now only Deepseek-v3's implementation is available. so now You're thinking like what you mean NaNo. So it just captures underlying Algorithms (more precisely self learning Alg 🧠) details that are introduced (or updated MHA -> MLA with some extra stuff) without scaling to B-parameters.Am trying to make sure all the important features of Deepseek-v3 are implemented in way easy to go through._

  ## *What DeepSeek did different:*
  *There is so much we can go on and on about. Some of the Features that are implemented this repo*
  
  *1. MultiHead Latent Attention (MLA).*
  
  *2. Misture of Experts (MoE (not standard one) they prefer to call DeepseekMoE. cause it's combination of shared isolated-Experts and fine-grained Experts segmentation).*
  
  *3. Multi-Token Prediction (MTP).*
  
  *4. Complementary Seq-Wise auxiliary Loss*
  
  *5. Rotary Positional Embedding (RoPE in a decoupled way)*

  *Still there is so much left. Here is the Link to the paper. Actually it 53 page BOOK 🙄 but still lot of the MOE stuffs are in DeepseekMoE paper. They talked lot about traning and MiX datatype traning (eg:int8, bf16, f16)*
  
  **V3**: *https://arxiv.org/pdf/2412.19437*
    
  **DeepseekMoE**: *https://arxiv.org/pdf/2401.06066*

# *Install*:
    pip install torch transformers
*Not much needed..... So it only requires if you wanna use script in your local machine. BUT if use notebook version then it's different story **colab** will take care every thing*.

# *Model Preview*:
*Wanna checkout quickly How model doing it's job. Train only on GPU with these setup*
  
    python train.py -v --n_embd=144 --block_size=16 --n_heads=3 --n_layers=4 --inter_dim=576 --experts_dim=144

*It will build model with.*

    total parameters 49.54 M
    active parameters 39.09 M
    
*And then train about 15M and it will sample from your trained model. that looks something like this*

    yo his pawms are swetty
    I had to a little controversy?
    
    I just heard you're on me
    You know
    If you hear the fuck is like a c'm not even
    And I'm the fuck on everyone
    And there's why I'm so much that I'm a damn big
    I've done got the same rap, I'm a closet
    I told you'd never meant to me
    I'm a fucking body
    I'mma play my face
    You can be a thousand times
    It's the party from these rappers
    You were gonna take our heart to me
    Here, I'm going to a shit
    And I'm not an ass, I'm on
    I'mma get so bad, 'cause you can't get to make you
    
*not so bad for 15M training 🤐*


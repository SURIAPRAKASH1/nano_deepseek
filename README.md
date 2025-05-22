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
  
  **V3: https://arxiv.org/pdf/2412.19437**
    
  **DeepseekMoE: https://arxiv.org/pdf/2401.06066**
    

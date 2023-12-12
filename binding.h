#ifdef __cplusplus
// #include <vector>
// #include <string>
//TODO: can we get ride of the ifdef for cplusplus? doesnt seem to work in the bindgen builder
extern "C" {
#endif

//TODO: includes should be done in builder compiler. otherwise the linking is broken and worse, polyglot.
#include <stdbool.h>
#include <stdint.h>
    extern unsigned char tokenCallback(void *, char *);

    int load_state(void *ctx, char *statefile, char *modes);

    int eval(void *params_ptr, void *ctx, char *text);

    void save_state(void *ctx, char *dst, char *modes);

    void *load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, bool embeddings, bool mmap, bool low_vram, bool vocab_only, int n_gpu, int n_batch, const char *maingpu, const char *tensorsplit, bool numa);

    int get_embeddings(void *params_ptr, void *state_pr, float *res_embeddings);

    int get_token_embeddings(void *params_ptr, void *state_pr, int *tokens, int tokenSize, float *res_embeddings);

    void *llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                                int top_k, float top_p, float temp, float repeat_penalty,
                                int repeat_last_n, bool ignore_eos, bool memory_f16,
                                int n_batch, int n_keep, const char **antiprompt, int antiprompt_count,
                                float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *logit_bias, const char *session_file, bool prompt_cache_all, bool mlock, bool mmap, const char *maingpu, const char *tensorsplit, bool prompt_cache_ro);

    void llama_free_params(void *params_ptr);

    void llama_binding_free_model(void *state);

    int llama_predict(void *params_ptr, void *state_pr, char *result, bool debug);

    //FINETUNE//
    //Structs
//     struct my_llama_hparams
//    {
//         uint32_t n_vocab    ;
//         uint32_t n_ctx      ;
//         uint32_t n_embd     ;
//         uint32_t n_ff       ;
//         uint32_t n_head     ;
//         uint32_t n_head_kv  ;
//         uint32_t n_layer    ;

//         float f_norm_eps     ;
//         float f_norm_rms_eps ;

//         float rope_freq_base  ;
//         float rope_freq_scale ;
//     };

// struct my_llama_layer {
//     // normalization
//     struct ggml_tensor * attention_norm;

//     // attention
//     struct ggml_tensor * wq;
//     struct ggml_tensor * wk;
//     struct ggml_tensor * wv;
//     struct ggml_tensor * wo;

//     // normalization
//     struct ggml_tensor * ffn_norm;

//     // ff
//     struct ggml_tensor * w1;
//     struct ggml_tensor * w2;
//     struct ggml_tensor * w3;
// };

// struct my_llama_model {
//     struct my_llama_hparams hparams;

//     struct ggml_tensor * tok_embeddings;

//     struct ggml_tensor * norm;
//     struct ggml_tensor * output;

//     //TODO: need to handle vectors
//     //std::vector<my_llama_layer> layers;
//     //same as above but for c code
//     struct my_llama_layer * layers;
// };

// struct my_llama_lora_hparams {
//     uint32_t lora_r ;
//     uint32_t lora_alpha ;
//     uint32_t n_rank_attention_norm ;
//     uint32_t n_rank_wq ;
//     uint32_t n_rank_wk ;
//     uint32_t n_rank_wv ;
//     uint32_t n_rank_wo ;
//     uint32_t n_rank_ffn_norm ;
//     uint32_t n_rank_w1 ;
//     uint32_t n_rank_w2 ;
//     uint32_t n_rank_w3 ;
//     uint32_t n_rank_tok_embeddings ;
//     uint32_t n_rank_norm ;
//     uint32_t n_rank_output ;

//     //TODO:
//     // bool operator!=(const my_llama_lora_hparams& other) const {
//     //     return memcmp(this, &other, sizeof(other));
//     // }
// };


// struct my_llama_lora_layer {
//     // normalization
//     struct ggml_tensor * attention_norm_a;
//     struct ggml_tensor * attention_norm_b;

//     // attention
//     struct ggml_tensor * wq_a;
//     struct ggml_tensor * wq_b;
//     struct ggml_tensor * wk_a;
//     struct ggml_tensor * wk_b;
//     struct ggml_tensor * wv_a;
//     struct ggml_tensor * wv_b;
//     struct ggml_tensor * wo_a;
//     struct ggml_tensor * wo_b;

//     // normalization
//     struct ggml_tensor * ffn_norm_a;
//     struct ggml_tensor * ffn_norm_b;

//     // ff
//     struct ggml_tensor * w1_a;
//     struct ggml_tensor * w1_b;
//     struct ggml_tensor * w2_a;
//     struct ggml_tensor * w2_b;
//     struct ggml_tensor * w3_a;
//     struct ggml_tensor * w3_b;
// };

// struct my_llama_lora {
//     struct ggml_context * ctx;
//     //TODO: might error here
//     // std::vector<uint8_t> data;
//     //as an array
//     uint8_t * data;
//     uint32_t data_size;


//     struct my_llama_lora_hparams hparams;

//     struct ggml_tensor * tok_embeddings_a;
//     struct ggml_tensor * tok_embeddings_b;

//     struct ggml_tensor * norm_a;
//     struct ggml_tensor * norm_b;
//     struct ggml_tensor * output_a;
//     struct ggml_tensor * output_b;

//     //TODO: might error here
//     // std::vector<my_llama_lora_layer> layers;
//     struct my_llama_lora_layer * layers;
//     uint32_t n_layers;
// };

// // gguf constants
//  const char * LLM_KV_TRAINING_TYPE_FINETUNE_LORA   = "finetune_lora";
//  const char * LLM_KV_TRAINING_TYPE                 = "training.type";

//  const char * LLM_KV_TRAINING_LORA_RANK_TOKEN_EMBD  = "training.lora.rank.token_embd";
//  const char * LLM_KV_TRAINING_LORA_RANK_OUTPUT_NORM = "training.lora.rank.output_norm";
//  const char * LLM_KV_TRAINING_LORA_RANK_OUTPUT      = "training.lora.rank.output";
//  const char * LLM_KV_TRAINING_LORA_RANK_ATTN_NORM   = "training.lora.rank.attn_norm";
//  const char * LLM_KV_TRAINING_LORA_RANK_ATTN_Q      = "training.lora.rank.attn_q";
//  const char * LLM_KV_TRAINING_LORA_RANK_ATTN_K      = "training.lora.rank.attn_k";
//  const char * LLM_KV_TRAINING_LORA_RANK_ATTN_V      = "training.lora.rank.attn_v";
//  const char * LLM_KV_TRAINING_LORA_RANK_ATTN_OUT    = "training.lora.rank.attn_output";
//  const char * LLM_KV_TRAINING_LORA_RANK_FFN_NORM    = "training.lora.rank.ffn_norm";
//  const char * LLM_KV_TRAINING_LORA_RANK_FFN_GATE    = "training.lora.rank.ffn_gate";
//  const char * LLM_KV_TRAINING_LORA_RANK_FFN_DOWN    = "training.lora.rank.ffn_down";
//  const char * LLM_KV_TRAINING_LORA_RANK_FFN_UP      = "training.lora.rank.ffn_up";

// // gguf constants (sync with gguf.py)

//  const char * LLM_KV_GENERAL_ARCHITECTURE        = "general.architecture";
//  const char * LLM_KV_GENERAL_FILE_TYPE           = "general.file_type";

//  const char * LLM_KV_CONTEXT_LENGTH              = "%s.context_length";
//  const char * LLM_KV_EMBEDDING_LENGTH            = "%s.embedding_length";
//  const char * LLM_KV_BLOCK_COUNT                 = "%s.block_count";
//  const char * LLM_KV_FEED_FORWARD_LENGTH         = "%s.feed_forward_length";
//  const char * LLM_KV_ATTENTION_HEAD_COUNT        = "%s.attention.head_count";
//  const char * LLM_KV_ATTENTION_HEAD_COUNT_KV     = "%s.attention.head_count_kv";
//  const char * LLM_KV_ATTENTION_LAYERNORM_RMS_EPS = "%s.attention.layer_norm_rms_epsilon";
//  const char * LLM_KV_ROPE_DIMENSION_COUNT        = "%s.rope.dimension_count";
//  const char * LLM_KV_ROPE_FREQ_BASE              = "%s.rope.freq_base"; // TODO load in llama.cpp
//  const char * LLM_KV_ROPE_SCALE_LINEAR           = "%s.rope.scale_linear";

//  const char * LLM_TENSOR_TOKEN_EMBD    = "token_embd";
//  const char * LLM_TENSOR_OUTPUT_NORM   = "output_norm";
//  const char * LLM_TENSOR_OUTPUT        = "output";
//  const char * LLM_TENSOR_ATTN_NORM     = "blk.%d.attn_norm";
//  const char * LLM_TENSOR_ATTN_Q        = "blk.%d.attn_q";
//  const char * LLM_TENSOR_ATTN_K        = "blk.%d.attn_k";
//  const char * LLM_TENSOR_ATTN_V        = "blk.%d.attn_v";
//  const char * LLM_TENSOR_ATTN_OUT      = "blk.%d.attn_output";
//  const char * LLM_TENSOR_FFN_NORM      = "blk.%d.ffn_norm";
//  const char * LLM_TENSOR_FFN_GATE      = "blk.%d.ffn_gate";
//  const char * LLM_TENSOR_FFN_DOWN      = "blk.%d.ffn_down";
//  const char * LLM_TENSOR_FFN_UP        = "blk.%d.ffn_up";


// struct llama_model_params llama_model_default_params(void);

// //Functions
// void print_params(struct my_llama_hparams * params);
// void print_lora_params(struct my_llama_lora_hparams * params);
// //TODO: above these are void* pointers to solve the unknown size+pad of struct
// void load_model_hparams_gguf(struct gguf_context * ctx, struct my_llama_hparams * hparams, const char * expected_arch);
// void init_model(struct llama_model * input, struct my_llama_model * model, const char * fn_model, uint32_t n_ctx);

#ifdef __cplusplus
}
// std::vector<std::string> create_vector(const char **strings, int count);
// void delete_vector(std::vector<std::string> *vec);
#endif
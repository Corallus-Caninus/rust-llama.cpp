
//int main(int argc, char ** argv) {
//    struct train_params params = get_default_train_params();
//
//    if (!train_params_parse(argc, argv, &params)) {
//        return 1;
//    }
//
//    if (params.common.seed == LLAMA_DEFAULT_SEED) {
//        params.common.seed = time(NULL);
//    }
//    printf("%s: seed: %u\n", __func__, params.common.seed);
//    srand(params.common.seed);
//
//    struct llama_model_params llama_mparams = llama_model_default_params();
//    llama_mparams.vocab_only = false;
//
//    printf("%s: model base = '%s'\n", __func__, params.fn_model_base);
//    struct llama_model * lmodel = llama_load_model_from_file(params.fn_model_base, llama_mparams);
//
//    struct llama_context_params llama_cparams = llama_context_default_params();
//    struct llama_context * lctx = llama_new_context_with_model(lmodel, llama_cparams);
//
//    struct my_llama_model model;
//    init_model(lmodel, &model, params.fn_model_base, params.common.n_ctx);
//
//    struct my_llama_lora lora;
//
//    struct train_state      * train = init_train_state();
//    struct ggml_opt_context * opt   = train->opt;
//
//    // set params from command line
//    if (params.custom_f_norm_rms_eps) {
//        model.hparams.f_norm_rms_eps  = params.f_norm_rms_eps;
//    }
//    if (params.custom_rope_freq_base) {
//        model.hparams.rope_freq_base  = params.rope_freq_base;
//    }
//    if (params.custom_rope_freq_scale) {
//        model.hparams.rope_freq_scale = params.rope_freq_scale;
//    }
//    lora.hparams.lora_r                = params.lora_r;
//    lora.hparams.lora_alpha            = params.custom_lora_alpha            ? params.lora_alpha            : params.lora_r;
//    uint32_t n_rank_attention_norm     = params.custom_n_rank_attention_norm ? params.n_rank_attention_norm : 1;
//    uint32_t n_rank_wq                 = params.custom_n_rank_wq             ? params.n_rank_wq             : params.lora_r;
//    uint32_t n_rank_wk                 = params.custom_n_rank_wk             ? params.n_rank_wk             : params.lora_r;
//    uint32_t n_rank_wv                 = params.custom_n_rank_wv             ? params.n_rank_wv             : params.lora_r;
//    uint32_t n_rank_wo                 = params.custom_n_rank_wo             ? params.n_rank_wo             : params.lora_r;
//    uint32_t n_rank_ffn_norm           = params.custom_n_rank_ffn_norm       ? params.n_rank_ffn_norm       : 1;
//    uint32_t n_rank_w1                 = params.custom_n_rank_w1             ? params.n_rank_w1             : params.lora_r;
//    uint32_t n_rank_w2                 = params.custom_n_rank_w2             ? params.n_rank_w2             : params.lora_r;
//    uint32_t n_rank_w3                 = params.custom_n_rank_w3             ? params.n_rank_w3             : params.lora_r;
//    uint32_t n_rank_tok_embeddings     = params.custom_n_rank_tok_embeddings ? params.n_rank_tok_embeddings : params.lora_r;
//    uint32_t n_rank_norm               = params.custom_n_rank_norm           ? params.n_rank_norm           : 1;
//    uint32_t n_rank_output             = params.custom_n_rank_output         ? params.n_rank_output         : params.lora_r;
//    lora.hparams.n_rank_attention_norm = n_rank_attention_norm;
//    lora.hparams.n_rank_wq             = n_rank_wq;
//    lora.hparams.n_rank_wk             = n_rank_wk;
//    lora.hparams.n_rank_wv             = n_rank_wv;
//    lora.hparams.n_rank_wo             = n_rank_wo;
//    lora.hparams.n_rank_ffn_norm       = n_rank_ffn_norm;
//    lora.hparams.n_rank_w1             = n_rank_w1;
//    lora.hparams.n_rank_w2             = n_rank_w2;
//    lora.hparams.n_rank_w3             = n_rank_w3;
//    lora.hparams.n_rank_tok_embeddings = n_rank_tok_embeddings;
//    lora.hparams.n_rank_norm           = n_rank_norm;
//    lora.hparams.n_rank_output         = n_rank_output;
//
//    // set opt params from command line
//    opt->params = ggml_opt_default_params(GGML_OPT_ADAM);
//    opt->params.print_forward_graph     = false;
//    opt->params.print_backward_graph    = false;
//    opt->params.n_threads               = params.common.n_threads;
//    opt->params.past                    = params.common.opt_past;
//    opt->params.delta                   = params.common.opt_delta;
//    opt->params.max_no_improvement      = params.common.opt_max_no_improvement;
//    opt->params.n_gradient_accumulation = params.common.n_gradient_accumulation;
//    opt->params.adam.n_iter             = params.common.adam_n_iter;
//    opt->params.adam.sched              = 1.0f;
//    opt->params.adam.alpha              = params.common.adam_alpha;
//    opt->params.adam.decay              = params.common.adam_decay;
//    opt->params.adam.decay_min_ndim     = params.common.adam_decay_min_ndim;
//    opt->params.adam.beta1              = params.common.adam_beta1;
//    opt->params.adam.beta2              = params.common.adam_beta2;
//    opt->params.adam.gclip              = params.common.adam_gclip;
//    opt->params.adam.eps_f              = params.common.adam_eps_f;
//
//    ggml_allocr * alloc = NULL;
//
//    printf("%s: init model\n", __func__);
//    bool existed = load_checkpoint_lora_file(params.common.fn_checkpoint_in, &model, &lora, train);
//
//    if (existed) {
//        // overwrite last n_ctx with user provided n_ctx
//        if (params.common.custom_n_ctx) {
//            model.hparams.n_ctx = params.common.n_ctx;
//        }
//
//        const bool opt_param_count_changed = (
//           (lora.hparams.n_rank_attention_norm != n_rank_attention_norm)
//        || (lora.hparams.n_rank_wq             != n_rank_wq)
//        || (lora.hparams.n_rank_wk             != n_rank_wk)
//        || (lora.hparams.n_rank_wv             != n_rank_wv)
//        || (lora.hparams.n_rank_wo             != n_rank_wo)
//        || (lora.hparams.n_rank_ffn_norm       != n_rank_ffn_norm)
//        || (lora.hparams.n_rank_w1             != n_rank_w1)
//        || (lora.hparams.n_rank_w2             != n_rank_w2)
//        || (lora.hparams.n_rank_w3             != n_rank_w3)
//        || (lora.hparams.n_rank_tok_embeddings != n_rank_tok_embeddings)
//        || (lora.hparams.n_rank_norm           != n_rank_norm)
//        || (lora.hparams.n_rank_output         != n_rank_output)
//        );
//
//        const bool opt_past_changed = opt->params.past != params.common.opt_past;
//
//        if (opt_param_count_changed) {
//            print_lora_params(&lora.hparams);
//            die("Provided rank differs from checkpoint file. To use different rank start finetune from scratch with empty input checkpoint, e.g --checkpoint-in ''. Aborting.");
//            // need to discard previous optimizer gradient statistics and opt_init with new shapes
//            // TODO
//        }
//        if (opt_past_changed) {
//            die("Optimizer parameter '--opt-past N' differs from checkpoint file. To use different value finetune from scratch with empty input checkpoint, e.g --checkpoint-in ''. Aborting");
//            // need to discard previous optimizer past function value statistics and opt_init with new shapes
//            // TODO
//        }
//    } else { // existed == false
//        init_lora(&model, &lora);
//        randomize_lora(&lora, params.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
//        if (!params.only_write_lora) {
//            ggml_opt_init(opt->ctx, opt, opt->params, get_parameter_count(&lora));
//        }
//    }
//    opt->iter = train->train_its;
//
//    print_params(&model.hparams);
//    print_lora_params(&lora.hparams);
//    printf("%s: total train_iterations %llu\n", __func__, (long long unsigned) train->train_its);
//    printf("%s: seen train_samples     %llu\n", __func__, (long long unsigned) train->train_samples);
//    printf("%s: seen train_tokens      %llu\n", __func__, (long long unsigned) train->train_tokens);
//    printf("%s: completed train_epochs %llu\n", __func__, (long long unsigned) train->train_epochs);
//    printf("%s: lora_size = %zu bytes (%.1f MB)\n", __func__, (ggml_used_mem(lora.ctx) + lora.data.size()), (float) (ggml_used_mem(lora.ctx) + lora.data.size()) / (1024.0f*1024.0f));
//
//    if (params.only_write_lora) {
//        save_train_files_data save_data;
//        save_data.fn_checkpoint_out = "";
//        save_data.fn_lora_out       = params.fn_lora_out;
//        save_data.pattern_fn_it     = params.common.pattern_fn_it;
//        save_data.fn_latest         = params.common.fn_latest;
//        save_data.model             = &model;
//        save_data.lora              = &lora;
//
//        save_train_files(&save_data, train);
//
//        free_train_state(train);
//        ggml_free(lora.ctx);
//        llama_free(lctx);
//        llama_free_model(lmodel);
//        return 0;
//    }
//
//    printf("%s: opt_size  = %zu bytes (%.1f MB)\n", __func__, ggml_get_mem_size(opt->ctx), (float) ggml_get_mem_size(opt->ctx) / (1024.0f*1024.0f));
//    printf("%s: opt iter %d\n", __func__, opt->iter);
//
//    int n_tokens = model.hparams.n_ctx;
//    int n_vocab  = model.hparams.n_vocab;
//    int n_batch  = params.common.n_batch;
//
//
//    std::vector<uint8_t> mem_input_data;
//    std::vector<uint8_t> mem_compute_data;
//
//    // context for input tensors without their data
//    struct ggml_init_params ctx_input_params = {
//        ggml_tensor_overhead() * 2, // mem_size
//        NULL,                       // mem_buffer
//        true,                       // no_alloc
//    };
//    struct ggml_context * ctx_input = ggml_init(ctx_input_params);
//
//    // the input tensors
//    struct ggml_tensor * tokens_input  = ggml_new_tensor_2d(ctx_input, GGML_TYPE_I32, n_tokens, n_batch);
//    struct ggml_tensor * target_probs  = ggml_new_tensor_3d(ctx_input, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
//
//    // measure required memory for input tensors
//    alloc = ggml_allocr_new_measure(tensor_alignment);
//    ggml_allocr_alloc(alloc, tokens_input);
//    ggml_allocr_alloc(alloc, target_probs);
//    size_t max_input_size = ggml_allocr_max_size(alloc) + tensor_alignment;
//    ggml_allocr_free(alloc);
//    printf("%s: input_size = %zu bytes (%.1f MB)\n", __func__, max_input_size, (float) max_input_size / (1024.0f*1024.0f));
//
//    // allocate input tensors
//    mem_input_data.resize(max_input_size);
//    alloc = ggml_allocr_new(mem_input_data.data(), mem_input_data.size(), tensor_alignment);
//    ggml_allocr_alloc(alloc, tokens_input);
//    ggml_allocr_alloc(alloc, target_probs);
//    ggml_allocr_free(alloc);
//
//    // context for compute tensors without their data
//    size_t estimated_compute_size_wo_data = (
//        ggml_tensor_overhead()*GGML_MAX_NODES*2
//      + (GGML_OBJECT_SIZE+GGML_GRAPH_SIZE)*(
//            params.common.use_checkpointing ? 3 : 2
//        )
//    );
//    struct ggml_init_params ctx_compute_params = {
//        estimated_compute_size_wo_data, // mem_size
//        NULL,                           // mem_buffer
//        true,                           // no_alloc
//    };
//    struct ggml_context * ctx_compute = NULL;
//
//    struct ggml_tensor * loss   = NULL;
//    struct ggml_tensor * logits = NULL;
//
//    struct ggml_cgraph * gf     = NULL;
//    struct ggml_cgraph * gb     = NULL;
//    struct ggml_cgraph * gb_tmp = NULL;
//
//    // measure required memory for compute tensors
//    size_t best_compute_size = SIZE_MAX;
//    enum ggml_cgraph_eval_order best_order = GGML_CGRAPH_EVAL_ORDER_COUNT;
//    // find best evaluation order
//    for (unsigned order = 0; order < (unsigned) GGML_CGRAPH_EVAL_ORDER_COUNT; ++order) {
//        ctx_compute = ggml_init(ctx_compute_params);
//        alloc = ggml_allocr_new_measure(tensor_alignment);
//        gf = ggml_new_graph(ctx_compute);
//        gf->order = (enum ggml_cgraph_eval_order) order;
//        gb = ggml_new_graph(ctx_compute);
//        gb_tmp = params.common.use_checkpointing
//            ? ggml_new_graph(ctx_compute)
//            : NULL;
//        loss = llama_build_lora_finetune_graphs(
//            &model, &lora, alloc, ctx_compute,
//            gf, gb, gb_tmp,
//            &logits, tokens_input, target_probs,
//            n_tokens, n_batch,
//            params.common.use_flash,
//            params.common.use_checkpointing
//        );
//        size_t max_compute_size = ggml_allocr_max_size(alloc) + tensor_alignment;
//        if (max_compute_size < best_compute_size) {
//            best_compute_size = max_compute_size;
//            best_order = gf->order;
//        }
//        ggml_allocr_free(alloc);
//        ggml_free(ctx_compute);
//    }
//    size_t max_compute_size = best_compute_size;
//    printf("%s: compute_size = %zu bytes (%.1f MB)\n", __func__, max_compute_size, (float) max_compute_size / (1024.0f*1024.0f));
//    printf("%s: evaluation order = %s\n", __func__,
//        (best_order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? "LEFT_TO_RIGHT" :
//        (best_order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? "RIGHT_TO_LEFT" :
//        "invalid");
//
//    // allocate compute tensors
//    mem_compute_data.resize(max_compute_size);
//    ctx_compute = ggml_init(ctx_compute_params);
//    alloc = ggml_allocr_new(mem_compute_data.data(), mem_compute_data.size(), tensor_alignment);
//    gf = ggml_new_graph(ctx_compute);
//    gf->order = best_order;
//    gb = ggml_new_graph(ctx_compute);
//    gb_tmp = params.common.use_checkpointing
//        ? ggml_new_graph(ctx_compute)
//        : NULL;
//    loss = llama_build_lora_finetune_graphs(
//        &model, &lora, alloc, ctx_compute,
//        gf, gb, gb_tmp,
//        &logits, tokens_input, target_probs,
//        n_tokens, n_batch,
//        params.common.use_flash,
//        params.common.use_checkpointing
//    );
//    ggml_allocr_free(alloc);
//
//    // tokenize data
//    std::vector<llama_token> train_tokens;
//    std::vector<size_t> train_samples_begin;
//    std::vector<size_t> train_samples_size;
//    printf("%s: tokenize training data\n", __func__);
//    tokenize_file(lctx,
//            params.common.fn_train_data,
//            params.common.sample_start,
//            params.common.include_sample_start,
//            params.common.overlapping_samples,
//            n_tokens,
//            train_tokens,
//            train_samples_begin,
//            train_samples_size);
//    GGML_ASSERT(train_samples_begin.size() == train_samples_size.size());
//
//    printf("%s: number of training tokens: %zu\n", __func__, train_tokens.size());
//
//    std::vector<size_t> token_noccurs;
//    token_noccurs.resize(model.hparams.n_vocab, 0);
//    for (unsigned int i = 0; i < train_tokens.size(); ++i) {
//        ++token_noccurs[train_tokens[i]];
//    }
//    int n_unique_tokens = 0;
//    for (unsigned int i = 0; i < token_noccurs.size(); ++i) {
//        if (token_noccurs[i] == 0) continue;
//        ++n_unique_tokens;
//    }
//    printf("%s: number of unique tokens: %d\n", __func__, n_unique_tokens);
//
//    size_t shuffle_samples_hash = compute_samples_hash(params.common.fn_train_data, train_samples_begin.data(), train_samples_size.data(), train_samples_size.size());
//    const bool changed_train_data = (shuffle_samples_hash != train->shuffle_samples_hash) || (train->shuffle_sample_count != train_samples_size.size());
//    if (changed_train_data) {
//        printf("%s: train data seems to have changed. restarting shuffled epoch.\n", __func__);
//    }
//    if (params.common.force_reshuffle) {
//        printf("%s: forced reshuffling of data. restarting with newly shuffled epoch.\n", __func__);
//    }
//    if ((train->shuffle_rng_state_current == "") || changed_train_data || params.common.force_reshuffle) {
//        train->shuffle_rng_state_current = mt19937_seed_to_state(params.common.seed);
//        train->shuffle_sample_count = train_samples_size.size();
//        train->shuffle_next_sample = 0;
//        train->shuffle_samples_hash = shuffle_samples_hash;
//    }
//    std::vector<size_t> train_shuffled_samples_offs;
//    std::vector<size_t> train_shuffled_samples_begin;
//    std::vector<size_t> train_shuffled_samples_size;
//    train_shuffled_samples_offs.resize(train_samples_begin.size());
//    train_shuffled_samples_begin.resize(train_samples_begin.size());
//    train_shuffled_samples_size.resize(train_samples_size.size());
//    train->shuffle_rng_state_next = shuffle_samples(
//        train->shuffle_rng_state_current,
//        train_shuffled_samples_offs.data(),
//        train_shuffled_samples_begin.data(),
//        train_shuffled_samples_size.data(),
//        train_samples_begin.data(),
//        train_samples_size.data(),
//        train_samples_size.size());
//
//    printf("%s: begin training\n", __func__);
//
//    save_train_files_data save_data;
//    save_data.fn_checkpoint_out = params.common.fn_checkpoint_out;
//    save_data.fn_lora_out       = params.fn_lora_out;
//    save_data.pattern_fn_it     = params.common.pattern_fn_it;
//    save_data.fn_latest         = params.common.fn_latest;
//    save_data.model             = &model;
//    save_data.lora              = &lora;
//
//    struct train_opt_callback_data opt_cb_data;
//    opt_cb_data.params                 = &params.common;
//    opt_cb_data.train                  = train;
//    opt_cb_data.save_cb                = &save_train_files;
//    opt_cb_data.save_data              = &save_data;
//    opt_cb_data.lctx                   = lctx;
//    opt_cb_data.last_save_iter         = opt->iter;
//    opt_cb_data.tokens_data            = train_tokens.data();
//    opt_cb_data.tokens_size            = train_tokens.size();
//    opt_cb_data.samples_begin          = train_samples_begin.data();
//    opt_cb_data.samples_size           = train_samples_size.data();
//    opt_cb_data.shuffled_samples_offs  = train_shuffled_samples_offs.data();
//    opt_cb_data.shuffled_samples_begin = train_shuffled_samples_begin.data();
//    opt_cb_data.shuffled_samples_size  = train_shuffled_samples_size.data();
//    opt_cb_data.samples_count          = train_samples_size.size();
//    opt_cb_data.tokens_input           = tokens_input;
//    opt_cb_data.target_probs           = target_probs;
//    opt_cb_data.first_iter             = opt->iter;
//    opt_cb_data.first_epoch            = train->train_epochs;
//    opt_cb_data.iter_at_last_epoch     = -1;
//    opt_cb_data.last_time              = ggml_time_ms();
//    opt_cb_data.millis_per_iter        = 0.0;
//
//    // measure required memory for work buffer
//    size_t max_work_size = ggml_graph_plan(gb, params.common.n_threads).work_size + GGML_OBJECT_SIZE;
//    printf("%s: work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float) max_work_size / (1024.0f*1024.0f));
//
//    // context for work buffer
//    struct ggml_init_params ctx_work_params = {
//        max_work_size, // mem_size
//        NULL,          // mem_buffer
//        false,         // no_alloc
//    };
//    struct ggml_context * ctx_work = ggml_init(ctx_work_params);
//
//    int64_t t0 = ggml_time_ms();
//
//    ggml_opt_resume_g(ctx_work, opt, loss, gf, gb, &train_opt_callback, (void *) &opt_cb_data);
//
//    ggml_free(ctx_work);
//    ggml_free(ctx_compute);
//    ggml_free(ctx_input);
//
//    int64_t t1 = ggml_time_ms();
//    printf("%s: total training time: ", __func__);
//    print_duration((double) (t1 - t0));
//    printf("\n");
//
//    int new_iters = opt->iter - opt_cb_data.last_save_iter;
//    if (new_iters > 0) {
//        train->train_its     += new_iters;
//        train->train_tokens  += new_iters * opt->params.n_gradient_accumulation * n_batch * n_tokens;
//
//        save_train_files(&save_data, train);
//        opt_cb_data.last_save_iter = opt->iter;
//    }
//
//    ggml_free(opt->ctx);
//    free_train_state(train);
//    ggml_free(lora.ctx);
//    llama_free(lctx);
//    llama_free_model(lmodel);
//    return 0;
//}
//
use llama_cpp_rs::*;
//we have a bindgen to every function mentioned above, write the equivalent main function in rust
//we will import them later
fn main() {
    //basic linker and bindgen test:
    let mut my_hparams = my_llama_hparams{
        n_vocab: 0,
        n_ctx: 0,
        n_embd: 0,
        n_ff: 0,
        n_head: 0,
        n_head_kv: 0,
        n_layer: 0,
        f_norm_rms_eps: 0.0,
        rope_freq_base: 0.0,
        rope_freq_scale: 0.0,
    };
    rs_print_params(&mut my_hparams);

    //TODO: something here likely needs to be manually dropped since the framework 
    //      is dropping. valgrind reports ~70bytes leaked but that could be during drop routines in rust post-segfault

    //recreate lora trainer in finetune example
    let train_state = rs_init_train_state();

    let mut train_params = rs_get_default_train_params();
    let mut llama_model_params = rs_llama_model_default_params();
    
    llama_model_params.vocab_only = false;
    train_params.fn_model_base = std::ffi::CString::new("../../models/ggml-model-f16-mistral-instruct.gguf").unwrap().into_raw();
    train_params.common.seed = 0;
    //print the entire struct of train_params
    println!("train_params: {:?}", train_params);

    let mut llama_model = rs_llama_load_model_from_file(train_params.fn_model_base, llama_model_params);
    let mut llama_cparams = rs_llama_context_default_params(); 

    let mut llama_context = rs_llama_new_context_with_model(llama_model, llama_cparams);

    println!("loaded model from file..");
    let llama_context_params = rs_llama_context_default_params();
    let llama_context = rs_llama_new_context_with_model(llama_model, llama_context_params);
    let mut my_llama_model = rs_new_my_llama_model();
    println!("my_llama_model construction complete..");
    let fn_model_base = train_params.fn_model_base;
    rs_init_model(llama_model, my_llama_model, fn_model_base, train_params.common.n_ctx as u32);
    println!("model initialized..");

    //TODO: segfault at end, we are leaking a bit from c/c++, likely a Drop call causes segfault on a c++ 
    // type, either a string or a vector. maybe a double free due to calling abstractions in c++ with native allocation.

}
    
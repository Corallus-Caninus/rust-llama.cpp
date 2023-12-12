#include "common.h"
#include "llama.h"
#include "train.h"
// #include "ggml.h"

#include "binding.h"
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <regex>
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#endif

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
void sigint_handler(int signo)
{
    if (signo == SIGINT)
    {
        _exit(130);
    }
}
#endif

static std::string llama_token_to_str(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}


int get_embeddings(void *params_ptr, void *state_pr, float *res_embeddings)
{
    gpt_params *params_p = (gpt_params *)params_ptr;
    llama_context *ctx = (llama_context *)state_pr;
    gpt_params params = *params_p;

    if (params_p->seed <= 0)
    {
        params_p->seed = time(NULL);
    }

    std::mt19937 rng(params_p->seed);

    llama_backend_init(params_p->numa);

    int n_past = 0;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params_p->prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, params_p->prompt, true);

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    if (embd_inp.size() > 0)
    {
        if (llama_eval(ctx, embd_inp.data(), embd_inp.size(), n_past))
        {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }
    }

    const llama_model *model = llama_get_model(ctx);

    const int n_embd = llama_n_embd(model);

    const auto embeddings = llama_get_embeddings(ctx);

    for (int i = 0; i < n_embd; i++)
    {
        res_embeddings[i] = embeddings[i];
    }

    return 0;
}

int get_token_embeddings(void *params_ptr, void *state_pr, int *tokens, int tokenSize, float *res_embeddings)
{
    gpt_params *params_p = (gpt_params *)params_ptr;
    llama_context *ctx = (llama_context *)state_pr;
    gpt_params params = *params_p;

    for (int i = 0; i < tokenSize; i++)
    {
        auto token_str = llama_token_to_str(ctx, tokens[i]);
        if (token_str.empty())
        {
            continue;
        }
        params_p->prompt += token_str;
    }

    return get_embeddings(params_ptr, state_pr, res_embeddings);
}

int eval(void *params_ptr, void *state_pr, char *text)
{
    gpt_params *params_p = (gpt_params *)params_ptr;
    llama_context *ctx = (llama_context *)state_pr;

    auto n_past = 0;
    auto last_n_tokens_data = std::vector<llama_token>(params_p->repeat_last_n, 0);

    auto tokens = std::vector<llama_token>(params_p->n_ctx);
    auto n_prompt_tokens = llama_tokenize(llama_get_model(ctx), text, strlen(text), tokens.data(), tokens.size(), true);

    if (n_prompt_tokens < 1)
    {
        fprintf(stderr, "%s : failed to tokenize prompt\n", __func__);
        return 1;
    }

    // evaluate prompt
    return llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past);
}

int llama_predict(void *params_ptr, void *state_pr, char *result, bool debug)
{
    gpt_params *params_p = (gpt_params *)params_ptr;
    llama_context *ctx = (llama_context *)state_pr;
    
    llama_set_n_threads(ctx, params_p->n_threads, params_p->n_threads_batch);

    const int n_ctx = llama_n_ctx(ctx);

    if (params_p->seed <= 0)
    {
        params_p->seed = time(NULL);
    }

    std::mt19937 rng(params_p->seed);

    // print input
    if (debug)
    {
        fprintf(stderr, "%s: input: %s\n", __func__, params_p->prompt.c_str());
    }

    std::string path_session = params_p->path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty())
    {
        if (debug)
        {
            fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        }
        // fopen to check for existing session
        FILE *fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL)
        {
            std::fclose(fp);

            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out))
            {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            llama_set_rng_seed(ctx, params_p->seed);
            if (debug)
            {
                fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
            }
        }
        else
        {
            if (debug)
            {
                fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
            }
        }
    }

    std::vector<llama_token> embd_inp;
    if (!params_p->prompt.empty() || session_tokens.empty())
    {
        // Add a space in front of the first character to match OG llama tokenizer behavior
        params_p->prompt.insert(0, 1, ' ');

        embd_inp = ::llama_tokenize(ctx, params_p->prompt, true);
    }
    else
    {
        embd_inp = session_tokens;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size())
    {
        for (llama_token id : session_tokens)
        {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens])
            {
                break;
            }
            n_matching_session_tokens++;
        }
        if (debug)
        {
            if (params_p->prompt.empty() && n_matching_session_tokens == embd_inp.size())
            {
                fprintf(stderr, "%s: using full prompt from session file\n", __func__);
            }
            else if (n_matching_session_tokens >= embd_inp.size())
            {
                fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
            }
            else if (n_matching_session_tokens < (embd_inp.size() / 2))
            {
                fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                        __func__, n_matching_session_tokens, embd_inp.size());
            }
            else
            {
                fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                        __func__, n_matching_session_tokens, embd_inp.size());
            }
        }
    }
    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() &&
        session_tokens.size() > embd_inp.size())
    {
        session_tokens.resize(embd_inp.size() - 1);
    }
    // number of tokens to keep when resetting context
    if (params_p->n_keep < 0 || params_p->n_keep > (int)embd_inp.size())
    {
        params_p->n_keep = (int)embd_inp.size();
    }

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();
    int n_past = 0;
    int n_remain = params_p->n_predict;
    int n_consumed = 0;
    int n_session_consumed = 0;

    std::vector<llama_token> embd;
    std::string res = "";

    // do one empty run to warm up the model
    {
        llama_token tmp[1] = {
            llama_token_bos(ctx),
        };
        llama_eval(ctx, tmp, 1, 0);
        llama_reset_timings(ctx);
    }

    while (n_remain != 0)
    {
        // predict
        if (embd.size() > 0)
        {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int)embd.size() > n_ctx)
            {
                const int n_left = n_past - params_p->n_keep;

                // always keep the first token - BOS
                n_past = std::max(1, params_p->n_keep);

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(), last_n_tokens.end() - embd.size());

                // stop saving session if we run out of context
                path_session.clear();

                // printf("\n---\n");
                // printf("resetting: '");
                // for (int i = 0; i < (int) embd.size(); i++) {
                //     printf("%s", llama_token_to_str(ctx, embd[i]));
                // }
                // printf("'\n");
                // printf("\n---\n");
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int)session_tokens.size())
            {
                size_t i = 0;
                for (; i < embd.size(); i++)
                {
                    if (embd[i] != session_tokens[n_session_consumed])
                    {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int)session_tokens.size())
                    {
                        ++i;
                        break;
                    }
                }
                if (i > 0)
                {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int)embd.size(); i += params_p->n_batch)
            {
                int n_eval = (int)embd.size() - i;
                if (n_eval > params_p->n_batch)
                {
                    n_eval = params_p->n_batch;
                }
                if (llama_eval(ctx, &embd[i], n_eval, n_past))
                {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    return 1;
                }
                n_past += n_eval;
            }

            if (embd.size() > 0 && !path_session.empty())
            {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int)embd_inp.size() <= n_consumed)
        {
            // out of user input, sample next token
            const float temp = params_p->temp;
            const int32_t top_k = params_p->top_k <= 0 ? llama_n_vocab(llama_get_model(ctx)) : params_p->top_k;
            const float top_p = params_p->top_p;
            const float tfs_z = params_p->tfs_z;
            const float typical_p = params_p->typical_p;
            const int32_t repeat_last_n = params_p->repeat_last_n < 0 ? n_ctx : params_p->repeat_last_n;
            const float repeat_penalty = params_p->repeat_penalty;
            const float alpha_presence = params_p->presence_penalty;
            const float alpha_frequency = params_p->frequency_penalty;
            const int mirostat = params_p->mirostat;
            const float mirostat_tau = params_p->mirostat_tau;
            const float mirostat_eta = params_p->mirostat_eta;
            const bool penalize_nl = params_p->penalize_nl;

            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params_p->prompt_cache_ro)
            {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);
                auto n_vocab = llama_n_vocab(llama_get_model(ctx));

                // Apply params_p->logit_bias map
                for (auto it = params_p->logit_bias.begin(); it != params_p->logit_bias.end(); it++)
                {
                    logits[it->first] += it->second;
                }

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++)
                {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

                // Apply penalties
                float nl_logit = logits[llama_token_nl(ctx)];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &candidates_p,
                                                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                              last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                              last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl)
                {
                    logits[llama_token_nl(ctx)] = nl_logit;
                }

                if (temp <= 0)
                {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &candidates_p);
                }
                else
                {
                    if (mirostat == 1)
                    {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    }
                    else if (mirostat == 2)
                    {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    }
                    else
                    {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                        llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }
                // printf("`%d`", candidates_p.size);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;

            // call the token callback, no need to check if one is actually registered, that will
            // be handled on the Go side.
            auto token_str = llama_token_to_str(ctx, id);
            if (!tokenCallback(state_pr, (char*)token_str.c_str()))
            {
                break;
            }
        }
        else
        {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int)embd_inp.size() > n_consumed)
            {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int)embd.size() >= params_p->n_batch)
                {
                    break;
                }
            }
        }

        for (auto id : embd)
        {
            res += llama_token_to_str(ctx, id);
        }

        // check for stop prompt
        if (params_p->antiprompt.size())
        {
            std::string last_output;
            for (auto id : last_n_tokens)
            {
                last_output += llama_token_to_str(ctx, id);
            }
            // Check if each of the reverse prompts appears at the end of the output.
            for (std::string &antiprompt : params_p->antiprompt)
            {
                // size_t extra_padding = params_p->interactive ? 0 : 2;
                size_t extra_padding = 2;
                size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                                              ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                                              : 0;

                if (last_output.find(antiprompt.c_str(), search_start_pos) != std::string::npos)
                {
                    goto end;
                }
            }
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos(ctx))
        {
            break;
        }
    }

    if (!path_session.empty() && params_p->prompt_cache_all && !params_p->prompt_cache_ro)
    {
        if (debug)
        {
            fprintf(stderr, "\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        }
        llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

end:
#if defined(_WIN32)
    signal(SIGINT, SIG_DFL);
#endif

    if (debug)
    {
        llama_print_timings(ctx);
        llama_reset_timings(ctx);
    }

    strcpy(result, res.c_str());
    return 0;
}

void llama_binding_free_model(void *state_ptr)
{
    llama_context *ctx = (llama_context *)state_ptr;
    llama_free(ctx);
}

void llama_free_params(void *params_ptr)
{
    gpt_params *params = (gpt_params *)params_ptr;
    delete params;
}

std::vector<std::string> create_vector(const char **strings, int count)
{
    std::vector<std::string> *vec = new std::vector<std::string>;
    for (int i = 0; i < count; i++)
    {
        vec->push_back(std::string(strings[i]));
    }
    return *vec;
}

void delete_vector(std::vector<std::string> *vec)
{
    delete vec;
}

int load_state(void *ctx, char *statefile, char *modes)
{
    llama_context *state = (llama_context *)ctx;
    const llama_context *constState = static_cast<const llama_context *>(state);
    const size_t state_size = llama_get_state_size(state);
    uint8_t *state_mem = new uint8_t[state_size];

    {
        FILE *fp_read = fopen(statefile, modes);
        if (state_size != llama_get_state_size(constState))
        {
            fprintf(stderr, "\n%s : failed to validate state size\n", __func__);
            return 1;
        }

        const size_t ret = fread(state_mem, 1, state_size, fp_read);
        if (ret != state_size)
        {
            fprintf(stderr, "\n%s : failed to read state\n", __func__);
            return 1;
        }

        llama_set_state_data(state, state_mem); // could also read directly from memory mapped file
        fclose(fp_read);
    }

    return 0;
}

void save_state(void *ctx, char *dst, char *modes)
{
    llama_context *state = (llama_context *)ctx;

    const size_t state_size = llama_get_state_size(state);
    uint8_t *state_mem = new uint8_t[state_size];

    // Save state (rng, logits, embedding and kv_cache) to file
    {
        FILE *fp_write = fopen(dst, modes);
        llama_copy_state_data(state, state_mem); // could also copy directly to memory mapped file
        fwrite(state_mem, 1, state_size, fp_write);
        fclose(fp_write);
    }
}

void *llama_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
                            float top_p, float temp, float repeat_penalty, int repeat_last_n, bool ignore_eos, bool memory_f16, int n_batch, int n_keep, const char **antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *logit_bias, const char *session_file, bool prompt_cache_all, bool mlock, bool mmap,
                            const char *maingpu, const char *tensorsplit, bool prompt_cache_ro)
{
    gpt_params *params = new gpt_params;
    params->seed = seed;
    params->n_threads = threads;
    params->n_threads_batch = threads;
    params->n_predict = tokens;
    params->repeat_last_n = repeat_last_n;
    params->prompt_cache_ro = prompt_cache_ro;
    params->top_k = top_k;
    params->top_p = top_p;
    params->memory_f16 = memory_f16;
    params->temp = temp;
    params->use_mmap = mmap;
    params->use_mlock = mlock;
    params->repeat_penalty = repeat_penalty;
    params->n_batch = n_batch;
    params->n_keep = n_keep;
    if (maingpu[0] != '\0')
    {
        params->main_gpu = std::stoi(maingpu);
    }

    if (tensorsplit[0] != '\0')
    {
        std::string arg_next = tensorsplit;
        // split string by , and /
        const std::regex regex{R"([,/]+)"};
        std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
        std::vector<std::string> split_arg{it, {}};
        GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

        for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i)
        {
            if (i < split_arg.size())
            {
                params->tensor_split[i] = std::stof(split_arg[i]);
            }
            else
            {
                params->tensor_split[i] = 0.0f;
            }
        }
    }

    params->prompt_cache_all = prompt_cache_all;
    params->path_prompt_cache = session_file;

    /* if (ignore_eos) // TODO: Cannot be set before context is allocated (llama_token_eos requires context access)
    {
        params->logit_bias[llama_token_eos()] = -INFINITY;
	}
    */
    if (antiprompt_count > 0)
    {
        params->antiprompt = create_vector(antiprompt, antiprompt_count);
    }
    params->tfs_z = tfs_z;
    params->typical_p = typical_p;
    params->presence_penalty = presence_penalty;
    params->mirostat = mirostat;
    params->mirostat_eta = mirostat_eta;
    params->mirostat_tau = mirostat_tau;
    params->penalize_nl = penalize_nl;
    std::stringstream ss(logit_bias);
    llama_token key;
    char sign;
    std::string value_str;
    if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-'))
    {
        params->logit_bias[key] = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
    }
    params->frequency_penalty = frequency_penalty;
    params->prompt = prompt;

    return params;
}

void *load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, bool embeddings, bool mmap, bool low_vram, bool vocab_only, int n_gpu_layers, int n_batch, const char *maingpu, const char *tensorsplit, bool numa)
{
    // load the model
    auto lparams = llama_context_default_params();
    auto mparams = llama_model_default_params();

    lparams.n_ctx = n_ctx;
    lparams.seed = n_seed;
    lparams.f16_kv = memory_f16;
    lparams.embedding = embeddings;
    mparams.use_mlock = mlock;
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.use_mmap = mmap;
    // mparams.low_vram = low_vram; LOW_VRAM not a thing anymore in the API? verify
    mparams.vocab_only = vocab_only;

    if (maingpu[0] != '\0')
    {
        mparams.main_gpu = std::stoi(maingpu);
    }

    if (tensorsplit[0] != '\0')
    {
        std::string arg_next = tensorsplit;
        // split string by , and /
        const std::regex regex{R"([,/]+)"};
        std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
        std::vector<std::string> split_arg{it, {}};
        GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

	float *tsplit = (float*)malloc(sizeof(float) * LLAMA_MAX_DEVICES);

        for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i)
        {
            if (i < split_arg.size())
            {
                tsplit[i] = std::stof(split_arg[i]);
            }
            else
            {
                tsplit[i] = 0.0f;
            }
        }
	mparams.tensor_split = tsplit;
    }

    if (n_batch > 0)
        lparams.n_batch = n_batch;

    llama_backend_init(numa);
    void *res = nullptr;
    try
    {
        auto model = llama_load_model_from_file(fname, mparams);
	res = llama_new_context_with_model(model, lparams);
    }
    catch (std::runtime_error &e)
    {
        fprintf(stderr, "failed %s", e.what());
        return res;
    }

    return res;
}

// //FINETUNE//
// //TODO: we can still extract this for codegen automation by using and include in this file
// void print_params(struct my_llama_hparams * params) {
//     printf("%s: n_vocab:   %u\n", __func__, params->n_vocab);
//     printf("%s: n_ctx:     %u\n", __func__, params->n_ctx);
//     printf("%s: n_embd:    %u\n", __func__, params->n_embd);
//     printf("%s: n_ff:      %u\n", __func__, params->n_ff);
//     printf("%s: n_head:    %u\n", __func__, params->n_head);
//     printf("%s: n_head_kv: %u\n", __func__, params->n_head_kv);
//     printf("%s: n_layer:   %u\n", __func__, params->n_layer);
//     printf("%s: norm_rms_eps          : %f\n", __func__, params->f_norm_rms_eps);
//     printf("%s: rope_freq_base        : %f\n", __func__, params->rope_freq_base);
//     printf("%s: rope_freq_scale       : %f\n", __func__, params->rope_freq_scale);
// }


//  void print_lora_params(struct my_llama_lora_hparams * params) {
//     printf("%s: n_rank_attention_norm : %u\n", __func__, params->n_rank_attention_norm);
//     printf("%s: n_rank_wq             : %u\n", __func__, params->n_rank_wq);
//     printf("%s: n_rank_wk             : %u\n", __func__, params->n_rank_wk);
//     printf("%s: n_rank_wv             : %u\n", __func__, params->n_rank_wv);
//     printf("%s: n_rank_wo             : %u\n", __func__, params->n_rank_wo);
//     printf("%s: n_rank_ffn_norm       : %u\n", __func__, params->n_rank_ffn_norm);
//     printf("%s: n_rank_w1             : %u\n", __func__, params->n_rank_w1);
//     printf("%s: n_rank_w2             : %u\n", __func__, params->n_rank_w2);
//     printf("%s: n_rank_w3             : %u\n", __func__, params->n_rank_w3);
//     printf("%s: n_rank_tok_embeddings : %u\n", __func__, params->n_rank_tok_embeddings);
//     printf("%s: n_rank_norm           : %u\n", __func__, params->n_rank_norm);
//     printf("%s: n_rank_output         : %u\n", __func__, params->n_rank_output);
// }


// #define GGUF_GET_KEY(ctx, dst, func, type, req, key) \
// { \
//     const std::string skey(key); \
//     const int kid = gguf_find_key(ctx, skey.c_str()); \
//     if (kid >= 0) { \
//         enum gguf_type ktype = gguf_get_kv_type(ctx, kid); \
//         if (ktype != (type)) { \
//             die_fmt("key %s has wrong type: %s", skey.c_str(), gguf_type_name(ktype)); \
//         } \
//         (dst) = func(ctx, kid); \
//     } else if (req) { \
//         die_fmt("key not found in model: %s", skey.c_str()); \
//     } \
// }

// //  void load_model_hparams_gguf(struct gguf_context * ctx, struct my_llama_hparams * hparams, const char * expected_arch) {
//  void load_model_hparams_gguf(gguf_context * ctx, struct my_llama_hparams * hparams, const char * expected_arch) {
//     std::string arch;

//     GGUF_GET_KEY(ctx, arch, gguf_get_val_str, GGUF_TYPE_STRING, true, LLM_KV_GENERAL_ARCHITECTURE);
//     if (expected_arch != NULL) {
//         if (arch != expected_arch) {
//             printf("%s: arch=%s expected_arch=%s\n", __func__, arch.c_str(), expected_arch);
//         }
//         GGML_ASSERT(arch == expected_arch);
//     }

//     std::vector<char> keybuf;
//     keybuf.resize(512);
//     auto kv = [&arch, &keybuf](const char * key) -> const char * {
//         snprintf(keybuf.data(), keybuf.size(), key, arch.c_str());
//         return keybuf.data();
//     };

//     GGUF_GET_KEY(ctx, hparams->n_embd,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_EMBEDDING_LENGTH));
//     GGUF_GET_KEY(ctx, hparams->n_ctx,          gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_CONTEXT_LENGTH));
//     GGUF_GET_KEY(ctx, hparams->n_ff,           gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_FEED_FORWARD_LENGTH));
//     GGUF_GET_KEY(ctx, hparams->n_head,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_ATTENTION_HEAD_COUNT));
//     GGUF_GET_KEY(ctx, hparams->n_layer,        gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_BLOCK_COUNT));

//     // n_head_kv is optional, default to n_head
//     hparams->n_head_kv = hparams->n_head;
//     GGUF_GET_KEY(ctx, hparams->n_head_kv,      gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_ATTENTION_HEAD_COUNT_KV));

//     float rope_freq_scale = 1.0f;
//     GGUF_GET_KEY(ctx, hparams->f_norm_rms_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS));
//     GGUF_GET_KEY(ctx, hparams->rope_freq_base, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_FREQ_BASE));
//     GGUF_GET_KEY(ctx, rope_freq_scale, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_SCALE_LINEAR));
//     if (rope_freq_scale != 1.0f) {
//         hparams->rope_freq_scale = 1.0f / rope_freq_scale;
//     }
// }

// //CPP struct methods to C function conversions:
// uint32_t llama_hparam_n_gqa(struct my_llama_hparams * hparams) {
//     return hparams->n_head/hparams->n_head_kv;
// }
// uint32_t llama_hparam_n_embd_head(struct my_llama_hparams * hparams) {
//     return hparams->n_embd/hparams->n_head;
// }
// uint32_t llama_hparam_n_embd_gqa(struct my_llama_hparams * hparams) {
//     return hparams->n_embd/llama_hparam_n_gqa(hparams);
// }
// bool llama_hparam_neq(struct my_llama_hparams * hparams, struct my_llama_hparams * other) {
//     return memcmp(hparams, other, sizeof(*other));
// }
// //End of CPP struct methods to C function conversions

// void init_model(struct llama_model * input, struct my_llama_model * model, const char * fn_model, uint32_t n_ctx) {
//    auto & hparams = model->hparams;

//    std::vector<char> tn_buf;
//    tn_buf.resize(GGML_MAX_NAME);
//    auto tn = [&tn_buf](const char * key) -> const char * {
//        snprintf(tn_buf.data(), tn_buf.size(), "%s.weight", key);
//        return tn_buf.data();
//    };
//    auto tni = [&tn_buf](const char * key, int bid) -> const char * {
//        snprintf(tn_buf.data(), tn_buf.size(), key, bid);
//        std::string s = tn_buf.data();
//        snprintf(tn_buf.data(), tn_buf.size(), "%s.weight", s.c_str());
//        return tn_buf.data();
//    };


//    // get parameters directly from gguf file
//    {
//        struct gguf_init_params params = {
//            /*.no_alloc = */ false,
//            /*.ctx      = */ NULL,
//        };
//        struct gguf_context * mctx = gguf_init_from_file(fn_model, params);

//        load_model_hparams_gguf(mctx, &hparams, "llama");

//        gguf_free(mctx);
//    }
//    hparams.n_vocab = llama_n_vocab(input);
//    hparams.n_ctx = n_ctx;

//    // get tensors from llama_model (possibly mmapped)
//    model->tok_embeddings = llama_get_model_tensor(input, tn(LLM_TENSOR_TOKEN_EMBD));
//    model->norm           = llama_get_model_tensor(input, tn(LLM_TENSOR_OUTPUT_NORM));
//    model->output         = llama_get_model_tensor(input, tn(LLM_TENSOR_OUTPUT));

//    assert_shape_2d(model->tok_embeddings, hparams.n_embd, hparams.n_vocab);
//    assert_shape_1d(model->norm,           hparams.n_embd);
//    assert_shape_2d(model->output,         hparams.n_embd, hparams.n_vocab);

//    //model->layers.resize(hparams.n_layer);
//    //instead of using a vector we will use a pointer to an array
//    if (model->layers == NULL){
//         model->layers = (struct my_llama_layer *)malloc(sizeof(struct my_llama_layer) * hparams.n_layer);
//    }else {
//         free(model->layers);
//         model->layers = (struct my_llama_layer *)malloc(sizeof(struct my_llama_layer) * hparams.n_layer);
//     }
//    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
//        //TODO: since this vector is addressed as a strided offset lets see if we can spoof it as an array
//        auto & layer = model->layers[i];

//        layer.attention_norm = llama_get_model_tensor(input, tni(LLM_TENSOR_ATTN_NORM, i));
//        layer.wq             = llama_get_model_tensor(input, tni(LLM_TENSOR_ATTN_Q, i));
//        layer.wk             = llama_get_model_tensor(input, tni(LLM_TENSOR_ATTN_K, i));
//        layer.wv             = llama_get_model_tensor(input, tni(LLM_TENSOR_ATTN_V, i));
//        layer.wo             = llama_get_model_tensor(input, tni(LLM_TENSOR_ATTN_OUT, i));
//        layer.ffn_norm       = llama_get_model_tensor(input, tni(LLM_TENSOR_FFN_NORM, i));
//        layer.w1             = llama_get_model_tensor(input, tni(LLM_TENSOR_FFN_GATE, i));
//        layer.w2             = llama_get_model_tensor(input, tni(LLM_TENSOR_FFN_DOWN, i));
//        layer.w3             = llama_get_model_tensor(input, tni(LLM_TENSOR_FFN_UP, i));

//        assert_shape_1d(layer.attention_norm, hparams.n_embd);
//        assert_shape_2d(layer.wq,             hparams.n_embd, hparams.n_embd);
//        //TODO: need to rework to C style.. its like someone wrote good C code then fucked it up for the pedantic claim of cpp
//        assert_shape_2d(layer.wk,             hparams.n_embd, llama_hparam_n_embd_gqa(&hparams));
//        assert_shape_2d(layer.wv,             hparams.n_embd, llama_hparam_n_embd_gqa(&hparams));
//        assert_shape_2d(layer.wo,             hparams.n_embd, hparams.n_embd);
//        assert_shape_1d(layer.ffn_norm,       hparams.n_embd);
//        assert_shape_2d(layer.w1,             hparams.n_embd, hparams.n_ff);
//        assert_shape_2d(layer.w2,             hparams.n_ff,   hparams.n_embd);
//         assert_shape_2d(layer.w3,             hparams.n_embd, hparams.n_ff);
//     }
// }

//////////////////////////TODO: big paste strong induction test//////////////////////////T


//TODO: try just creating a header for finetune.cpp instead of these bindings
//  void set_param_lora(struct my_llama_lora * lora) {
//     //const uint32_t n_layer = lora->layers.size();
//     //layers is an array so generate the equivalent code for an array instead of a vector
//     const uint32_t n_layer = sizeof

//     struct ggml_context* ctx = lora->ctx;

//     ggml_set_param(ctx, lora->tok_embeddings_a);
//     ggml_set_param(ctx, lora->tok_embeddings_b);
//     ggml_set_param(ctx, lora->norm_a);
//     ggml_set_param(ctx, lora->norm_b);
//     ggml_set_param(ctx, lora->output_a);
//     ggml_set_param(ctx, lora->output_b);

//     for (uint32_t i = 0; i < n_layer; ++i) {
//         auto & layer = lora->layers[i];

//         ggml_set_param(ctx, layer.attention_norm_a);
//         ggml_set_param(ctx, layer.attention_norm_b);
//         ggml_set_param(ctx, layer.wq_a);
//         ggml_set_param(ctx, layer.wq_b);
//         ggml_set_param(ctx, layer.wk_a);
//         ggml_set_param(ctx, layer.wk_b);
//         ggml_set_param(ctx, layer.wv_a);
//         ggml_set_param(ctx, layer.wv_b);
//         ggml_set_param(ctx, layer.wo_a);
//         ggml_set_param(ctx, layer.wo_b);
//         ggml_set_param(ctx, layer.ffn_norm_a);
//         ggml_set_param(ctx, layer.ffn_norm_b);
//         ggml_set_param(ctx, layer.w1_a);
//         ggml_set_param(ctx, layer.w1_b);
//         ggml_set_param(ctx, layer.w2_a);
//         ggml_set_param(ctx, layer.w2_b);
//         ggml_set_param(ctx, layer.w3_a);
//         ggml_set_param(ctx, layer.w3_b);
//     }
// }

//  void alloc_lora(struct ggml_allocr * alloc, struct my_llama_lora * lora) {
//     ggml_allocr_alloc(alloc, lora->tok_embeddings_a);
//     ggml_allocr_alloc(alloc, lora->tok_embeddings_b);
//     ggml_allocr_alloc(alloc, lora->norm_a);
//     ggml_allocr_alloc(alloc, lora->norm_b);
//     ggml_allocr_alloc(alloc, lora->output_a);
//     ggml_allocr_alloc(alloc, lora->output_b);
//     for (uint32_t i = 0; i < lora->layers.size(); ++i) {
//         auto & layer = lora->layers[i];
//         ggml_allocr_alloc(alloc, layer.attention_norm_a);
//         ggml_allocr_alloc(alloc, layer.attention_norm_b);
//         ggml_allocr_alloc(alloc, layer.wq_a);
//         ggml_allocr_alloc(alloc, layer.wq_b);
//         ggml_allocr_alloc(alloc, layer.wk_a);
//         ggml_allocr_alloc(alloc, layer.wk_b);
//         ggml_allocr_alloc(alloc, layer.wv_a);
//         ggml_allocr_alloc(alloc, layer.wv_b);
//         ggml_allocr_alloc(alloc, layer.wo_a);
//         ggml_allocr_alloc(alloc, layer.wo_b);
//         ggml_allocr_alloc(alloc, layer.ffn_norm_a);
//         ggml_allocr_alloc(alloc, layer.ffn_norm_b);
//         ggml_allocr_alloc(alloc, layer.w1_a);
//         ggml_allocr_alloc(alloc, layer.w1_b);
//         ggml_allocr_alloc(alloc, layer.w2_a);
//         ggml_allocr_alloc(alloc, layer.w2_b);
//         ggml_allocr_alloc(alloc, layer.w3_a);
//         ggml_allocr_alloc(alloc, layer.w3_b);
//     }
//     ggml_allocr_alloc(alloc, lora->tok_embeddings_a->grad);
//     ggml_allocr_alloc(alloc, lora->tok_embeddings_b->grad);
//     ggml_allocr_alloc(alloc, lora->norm_a->grad);
//     ggml_allocr_alloc(alloc, lora->norm_b->grad);
//     ggml_allocr_alloc(alloc, lora->output_a->grad);
//     ggml_allocr_alloc(alloc, lora->output_b->grad);
//     for (uint32_t i = 0; i < lora->layers.size(); ++i) {
//         auto & layer = lora->layers[i];
//         ggml_allocr_alloc(alloc, layer.attention_norm_a->grad);
//         ggml_allocr_alloc(alloc, layer.attention_norm_b->grad);
//         ggml_allocr_alloc(alloc, layer.wq_a->grad);
//         ggml_allocr_alloc(alloc, layer.wq_b->grad);
//         ggml_allocr_alloc(alloc, layer.wk_a->grad);
//         ggml_allocr_alloc(alloc, layer.wk_b->grad);
//         ggml_allocr_alloc(alloc, layer.wv_a->grad);
//         ggml_allocr_alloc(alloc, layer.wv_b->grad);
//         ggml_allocr_alloc(alloc, layer.wo_a->grad);
//         ggml_allocr_alloc(alloc, layer.wo_b->grad);
//         ggml_allocr_alloc(alloc, layer.ffn_norm_a->grad);
//         ggml_allocr_alloc(alloc, layer.ffn_norm_b->grad);
//         ggml_allocr_alloc(alloc, layer.w1_a->grad);
//         ggml_allocr_alloc(alloc, layer.w1_b->grad);
//         ggml_allocr_alloc(alloc, layer.w2_a->grad);
//         ggml_allocr_alloc(alloc, layer.w2_b->grad);
//         ggml_allocr_alloc(alloc, layer.w3_a->grad);
//         ggml_allocr_alloc(alloc, layer.w3_b->grad);
//     }
// }

//  void init_lora(const struct my_llama_model * model, struct my_llama_lora * lora) {
//     const auto & lparams = lora->hparams;

//     const uint32_t n_embd     = model->hparams.n_embd;
//     const uint32_t n_embd_gqa = model->hparams.n_embd_gqa();
//     const uint32_t n_layer    = model->hparams.n_layer;
//     const uint32_t n_vocab    = model->hparams.n_vocab;
//     const uint32_t n_ff       = model->hparams.n_ff;

//     std::vector<char> tn_buf;
//     tn_buf.resize(GGML_MAX_NAME);
//     auto tn = [&tn_buf](const char * key, const char * suffix) -> const char * {
//         snprintf(tn_buf.data(), tn_buf.size(), "%s%s", key, suffix);
//         return tn_buf.data();
//     };
//     auto tni = [&tn_buf](const char * key, const char * suffix, int bid) -> const char * {
//         snprintf(tn_buf.data(), tn_buf.size(), key, bid);
//         std::string s = tn_buf.data();
//         snprintf(tn_buf.data(), tn_buf.size(), "%s%s", s.c_str(), suffix);
//         return tn_buf.data();
//     };

//     // context for lora tensors without their data
//     struct ggml_init_params ctx_lora_params;
//     ctx_lora_params.mem_size   = ggml_tensor_overhead()*2*(6 + n_layer*18);
//     ctx_lora_params.mem_buffer = NULL;
//     ctx_lora_params.no_alloc   = true;

//     struct ggml_context * ctx = ggml_init(ctx_lora_params);
//     lora->ctx = ctx;

//     lora->tok_embeddings_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_tok_embeddings, n_embd);
//     lora->tok_embeddings_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_tok_embeddings, n_vocab);
//     lora->norm_a           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_norm, n_embd);
//     lora->norm_b           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_norm, 1);
//     lora->output_a         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_output, n_embd);
//     lora->output_b         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_output, n_vocab);

//     ggml_set_name(lora->tok_embeddings_a, tn(LLM_TENSOR_TOKEN_EMBD,  ".weight.lora_a"));
//     ggml_set_name(lora->tok_embeddings_b, tn(LLM_TENSOR_TOKEN_EMBD,  ".weight.lora_b"));
//     ggml_set_name(lora->norm_a,           tn(LLM_TENSOR_OUTPUT_NORM, ".weight.lora_a"));
//     ggml_set_name(lora->norm_b,           tn(LLM_TENSOR_OUTPUT_NORM, ".weight.lora_b"));
//     ggml_set_name(lora->output_a,         tn(LLM_TENSOR_OUTPUT,      ".weight.lora_a"));
//     ggml_set_name(lora->output_b,         tn(LLM_TENSOR_OUTPUT,      ".weight.lora_b"));

//     lora->layers.resize(n_layer);
//     for (uint32_t i = 0; i < n_layer; ++i) {
//         auto & layer = lora->layers[i];

//         layer.attention_norm_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_attention_norm, n_embd);
//         layer.attention_norm_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_attention_norm, 1);

//         layer.wq_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wq, n_embd);
//         layer.wq_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wq, n_embd);
//         layer.wk_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wk, n_embd);
//         layer.wk_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wk, n_embd_gqa);
//         layer.wv_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wv, n_embd);
//         layer.wv_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wv, n_embd_gqa);
//         layer.wo_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wo, n_embd);
//         layer.wo_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wo, n_embd);

//         layer.ffn_norm_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_ffn_norm, n_embd);
//         layer.ffn_norm_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_ffn_norm, 1);

//         layer.w1_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w1, n_embd);
//         layer.w1_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w1, n_ff);
//         layer.w2_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w2, n_ff);
//         layer.w2_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w2, n_embd);
//         layer.w3_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w3, n_embd);
//         layer.w3_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w3, n_ff);

//         ggml_set_name(layer.attention_norm_a, tni(LLM_TENSOR_ATTN_NORM, ".weight.lora_a", i));
//         ggml_set_name(layer.attention_norm_b, tni(LLM_TENSOR_ATTN_NORM, ".weight.lora_b", i));
//         ggml_set_name(layer.wq_a,             tni(LLM_TENSOR_ATTN_Q,    ".weight.lora_a", i));
//         ggml_set_name(layer.wq_b,             tni(LLM_TENSOR_ATTN_Q,    ".weight.lora_b", i));
//         ggml_set_name(layer.wk_a,             tni(LLM_TENSOR_ATTN_K,    ".weight.lora_a", i));
//         ggml_set_name(layer.wk_b,             tni(LLM_TENSOR_ATTN_K,    ".weight.lora_b", i));
//         ggml_set_name(layer.wv_a,             tni(LLM_TENSOR_ATTN_V,    ".weight.lora_a", i));
//         ggml_set_name(layer.wv_b,             tni(LLM_TENSOR_ATTN_V,    ".weight.lora_b", i));
//         ggml_set_name(layer.wo_a,             tni(LLM_TENSOR_ATTN_OUT,  ".weight.lora_a", i));
//         ggml_set_name(layer.wo_b,             tni(LLM_TENSOR_ATTN_OUT,  ".weight.lora_b", i));
//         ggml_set_name(layer.ffn_norm_a,       tni(LLM_TENSOR_FFN_NORM,  ".weight.lora_a", i));
//         ggml_set_name(layer.ffn_norm_b,       tni(LLM_TENSOR_FFN_NORM,  ".weight.lora_b", i));
//         ggml_set_name(layer.w1_a,             tni(LLM_TENSOR_FFN_GATE,  ".weight.lora_a", i));
//         ggml_set_name(layer.w1_b,             tni(LLM_TENSOR_FFN_GATE,  ".weight.lora_b", i));
//         ggml_set_name(layer.w2_a,             tni(LLM_TENSOR_FFN_DOWN,  ".weight.lora_a", i));
//         ggml_set_name(layer.w2_b,             tni(LLM_TENSOR_FFN_DOWN,  ".weight.lora_b", i));
//         ggml_set_name(layer.w3_a,             tni(LLM_TENSOR_FFN_UP,    ".weight.lora_a", i));
//         ggml_set_name(layer.w3_b,             tni(LLM_TENSOR_FFN_UP,    ".weight.lora_b", i));
//     }

//     set_param_lora(lora);

//     // measure data size
//     struct ggml_allocr * alloc = NULL;
//     alloc = ggml_allocr_new_measure(tensor_alignment);
//     alloc_lora(alloc, lora);

//     // allocate data
//     lora->data.resize(ggml_allocr_max_size(alloc) + tensor_alignment);
//     ggml_allocr_free(alloc);
//     alloc = ggml_allocr_new(lora->data.data(), lora->data.size(), tensor_alignment);
//     alloc_lora(alloc, lora);
//     ggml_allocr_free(alloc);
// }

//  void randomize_lora(struct my_llama_lora * lora, int seed, float mean, float std, float min, float max) {
//     const uint32_t n_layer = lora->layers.size();

//     struct random_normal_distribution * rnd = init_random_normal_distribution(seed, mean, std, min, max);

//     randomize_tensor_normal(lora->tok_embeddings_a, rnd);
//     randomize_tensor_normal(lora->tok_embeddings_b, rnd);
//     randomize_tensor_normal(lora->norm_a,           rnd);
//     randomize_tensor_normal(lora->norm_b,           rnd);
//     randomize_tensor_normal(lora->output_a,         rnd);
//     randomize_tensor_normal(lora->output_b,         rnd);

//     for (uint32_t i = 0; i < n_layer; ++i) {
//         auto & layer = lora->layers[i];
//         randomize_tensor_normal(layer.attention_norm_a, rnd);
//         randomize_tensor_normal(layer.attention_norm_b, rnd);

//         randomize_tensor_normal(layer.wq_a, rnd);
//         randomize_tensor_normal(layer.wq_b, rnd);
//         randomize_tensor_normal(layer.wk_a, rnd);
//         randomize_tensor_normal(layer.wk_b, rnd);
//         randomize_tensor_normal(layer.wv_a, rnd);
//         randomize_tensor_normal(layer.wv_b, rnd);
//         randomize_tensor_normal(layer.wo_a, rnd);
//         randomize_tensor_normal(layer.wo_b, rnd);

//         randomize_tensor_normal(layer.ffn_norm_a, rnd);
//         randomize_tensor_normal(layer.ffn_norm_b, rnd);

//         randomize_tensor_normal(layer.w1_a, rnd);
//         randomize_tensor_normal(layer.w1_b, rnd);
//         randomize_tensor_normal(layer.w2_a, rnd);
//         randomize_tensor_normal(layer.w2_b, rnd);
//         randomize_tensor_normal(layer.w3_a, rnd);
//         randomize_tensor_normal(layer.w3_b, rnd);
//     }

//     free_random_normal_distribution(rnd);
// }

//  struct ggml_tensor * llama_build_lora_finetune_graphs(
//         struct my_llama_model * model,
//         struct my_llama_lora  * lora,
//         struct ggml_allocr    * alloc,
//         struct ggml_context   * ctx,
//         struct ggml_cgraph    * gf,
//         struct ggml_cgraph    * gb,
//         struct ggml_cgraph    * gb_tmp,
//         struct ggml_tensor  * * logits,
//         struct ggml_tensor    * tokens_input,
//         struct ggml_tensor    * targets,
//         const  int              n_tokens,
//         const  int              n_batch,
//         const  bool             enable_flash_attn,
//         const  bool             enable_checkpointing) {

//     ggml_set_scratch(ctx, { 0, 0, nullptr, });
//     const int n_past = 0;
//     const int N = n_tokens;
//     const auto & hparams  = model->hparams;
//     const int n_ctx       = hparams.n_ctx;
//     const int n_vocab     = hparams.n_vocab;
//     const int n_embd      = hparams.n_embd;
//     const int n_layer     = hparams.n_layer;
//     const int n_head      = hparams.n_head;
//     const int n_head_kv   = hparams.n_head_kv;
//     const int n_ff        = hparams.n_ff;
//     //TODO: this may need a wrapper extraction of methods to make c compliant bindings
//     const int n_rot       = hparams.n_embd_head();
//     const int n_embd_head = hparams.n_embd_head();
//     const int n_embd_gqa  = hparams.n_embd_gqa();
//     const float rms_norm_eps    = hparams.f_norm_rms_eps;
//     const float rope_freq_base  = hparams.rope_freq_base;
//     const float rope_freq_scale = hparams.rope_freq_scale;

//     GGML_ASSERT((size_t) n_layer == lora->layers.size());

//     auto set_name = [](struct ggml_tensor * t, const char * n) {
//         ggml_set_name(t, n);
//         if (t->grad) {
//             ggml_format_name(t->grad, "%s->grad", n);
//         }
//     };

//     // KQ_pos - contains the positions
//     struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
//     ggml_allocr_alloc(alloc, KQ_pos);
//     if (!ggml_allocr_is_measure(alloc)) {
//         int * data = (int *) KQ_pos->data;
//         for (int i = 0; i < N; ++i) {
//             data[i] = n_past + i;
//         }
//     }

//     // rope has so much parameters that we make a custom function for it
//     auto rope = [ctx, KQ_pos, n_rot, n_ctx, rope_freq_base, rope_freq_scale]
//                 (struct ggml_tensor * t) -> struct ggml_tensor * {
//         // not capturing these, to silcence warnings
//         const int rope_mode = 0;

//         return ggml_rope_custom(ctx,
//             t, KQ_pos, n_rot, rope_mode, n_ctx,
//             rope_freq_base, rope_freq_scale);
//     };

//     set_name(tokens_input, "tokens_input");
//     set_name(targets,      "targets");

//     GGML_ASSERT(tokens_input->type == GGML_TYPE_I32);

//     auto add_to_f32 = [] (struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
//         if (ggml_is_quantized(a->type)) {
//             return ggml_add_cast(ctx, a, b, GGML_TYPE_F32);
//         } else if (a->type == GGML_TYPE_F32) {
//             return ggml_add(ctx, a, b);
//         } else {
//             die_fmt("%s: Finetuning on tensors with type '%s' is not yet supported.\n",
//                 __func__, ggml_type_name(a->type));
//         }
//     };

//     struct ggml_tensor * tok_embeddings = add_to_f32(ctx, model->tok_embeddings, ggml_mul_mat(ctx, lora->tok_embeddings_a, lora->tok_embeddings_b));
//     struct ggml_tensor * norm           = add_to_f32(ctx, model->norm, ggml_mul_mat(ctx, lora->norm_a, lora->norm_b));
//     struct ggml_tensor * output         = add_to_f32(ctx, model->output, ggml_mul_mat(ctx, lora->output_a, lora->output_b));

//     struct ggml_tensor * t00 = ggml_reshape_1d(ctx, tokens_input, N*n_batch);  set_name(t00, "t00"); assert_shape_1d(t00, N*n_batch);
//     struct ggml_tensor * t01 = ggml_get_rows(ctx, tok_embeddings, t00);        set_name(t01, "t01"); assert_shape_2d(t01, n_embd, N*n_batch);

//     struct ggml_tensor * cur = t01;

//     std::vector<struct ggml_tensor *> checkpoints;
//     if (enable_checkpointing) {
//         checkpoints.push_back(tokens_input);
//         checkpoints.push_back(targets);
//         checkpoints.push_back(t00);
//         checkpoints.push_back(t01);
//     }

//     struct ggml_tensor * kv_scale = NULL;
//     if (!enable_flash_attn) {
//         kv_scale = ggml_new_f32(ctx, 1.0f/sqrtf(float(n_embd)/n_head));
//     }

//     for (int il = 0; il < n_layer; ++il) {
//         struct my_llama_layer & layer = model->layers[il];
//         struct my_llama_lora_layer & llayer = lora->layers[il];

//         struct ggml_tensor * attention_norm = add_to_f32(ctx, layer.attention_norm, ggml_mul_mat(ctx, llayer.attention_norm_a, llayer.attention_norm_b));
//         struct ggml_tensor * ffn_norm = add_to_f32(ctx, layer.ffn_norm, ggml_mul_mat(ctx, llayer.ffn_norm_a, llayer.ffn_norm_b));
//         struct ggml_tensor * wq = add_to_f32(ctx, layer.wq, ggml_mul_mat(ctx, llayer.wq_a, llayer.wq_b));
//         struct ggml_tensor * wk = add_to_f32(ctx, layer.wk, ggml_mul_mat(ctx, llayer.wk_a, llayer.wk_b));
//         struct ggml_tensor * wv = add_to_f32(ctx, layer.wv, ggml_mul_mat(ctx, llayer.wv_a, llayer.wv_b));
//         struct ggml_tensor * wo = add_to_f32(ctx, layer.wo, ggml_mul_mat(ctx, llayer.wo_a, llayer.wo_b));
//         struct ggml_tensor * w1 = add_to_f32(ctx, layer.w1, ggml_mul_mat(ctx, llayer.w1_a, llayer.w1_b));
//         struct ggml_tensor * w2 = add_to_f32(ctx, layer.w2, ggml_mul_mat(ctx, llayer.w2_a, llayer.w2_b));
//         struct ggml_tensor * w3 = add_to_f32(ctx, layer.w3, ggml_mul_mat(ctx, llayer.w3_a, llayer.w3_b));

//         struct ggml_tensor * t02 = ggml_rms_norm     (ctx, cur, rms_norm_eps);                       set_name(t02, "t02");     assert_shape_2d(t02, n_embd, N*n_batch);
//         struct ggml_tensor * t03 = ggml_repeat       (ctx, attention_norm, t02);                     set_name(t03, "t03");     assert_shape_2d(t03, n_embd, N*n_batch);
//         struct ggml_tensor * t04 = ggml_mul          (ctx, t03, t02);                                set_name(t04, "t04");     assert_shape_2d(t04, n_embd, N*n_batch);
//         struct ggml_tensor * t05 = ggml_mul_mat      (ctx, wq, t04);                                 set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
//         struct ggml_tensor * t06 = ggml_reshape_4d   (ctx, t05, n_embd_head, n_head, N, n_batch);    set_name(t06, "t06");     assert_shape_4d(t06, n_embd_head, n_head, N, n_batch);
//         struct ggml_tensor * t07 = rope              (t06);                                          set_name(t07, "t07");     assert_shape_4d(t07, n_embd_head, n_head, N, n_batch);
//         struct ggml_tensor * t08 = ggml_mul_mat      (ctx, wk, t04);                                 set_name(t08, "t08");     assert_shape_2d(t08, n_embd_gqa, N*n_batch);
//         struct ggml_tensor * t09 = ggml_reshape_4d   (ctx, t08, n_embd_head, n_head_kv, N, n_batch); set_name(t09, "t09");     assert_shape_4d(t09, n_embd_head, n_head_kv, N, n_batch);
//         struct ggml_tensor * t10 = rope              (t09);                                          set_name(t10, "t10");     assert_shape_4d(t10, n_embd_head, n_head_kv, N, n_batch);

//         struct ggml_tensor * t11;
//         if (ggml_is_quantized(wv->type)) {
//             struct ggml_tensor * t11_1 = ggml_mul_mat  (ctx, wv, t04);                               set_name(t11_1, "t11_1"); assert_shape_2d(t11_1, n_embd_gqa, N*n_batch);
//             struct ggml_tensor * t11_2 = ggml_transpose(ctx, t11_1);                                 set_name(t11_2, "t11_2"); assert_shape_2d(t11_2, N*n_batch, n_embd_gqa);
//                                  t11   = ggml_cont     (ctx, t11_2);                                 set_name(t11, "t11");     assert_shape_2d(t11, N*n_batch, n_embd_gqa);
//         } else {
//                                  t11   = ggml_mul_mat  (ctx, t04, wv);                               set_name(t11, "t11");     assert_shape_2d(t11, N*n_batch, n_embd_gqa);
//         }

//         struct ggml_tensor * t12 = ggml_reshape_4d   (ctx, t11, N, n_batch, n_embd_head, n_head_kv); set_name(t12, "t12");     assert_shape_4d(t12, N, n_batch, n_embd_head, n_head_kv);
//         struct ggml_tensor * t13 = ggml_permute      (ctx, t07, 0, 2, 1, 3);                         set_name(t13, "t13");     assert_shape_4d(t13, n_embd_head, N, n_head, n_batch);
//         struct ggml_tensor * t14 = ggml_permute      (ctx, t10, 0, 2, 1, 3);                         set_name(t14, "t14");     assert_shape_4d(t14, n_embd_head, N, n_head_kv, n_batch);
//         struct ggml_tensor * t15 = ggml_permute      (ctx, t12, 0, 3, 1, 2);                         set_name(t15, "t15");     assert_shape_4d(t15, N, n_embd_head, n_head_kv, n_batch);
//         struct ggml_tensor * t16;
//         if (enable_flash_attn) {
//             t16 = ggml_flash_attn(ctx, t13, t14, t15, true);                                         set_name(t16, "t16");     assert_shape_4d(t16, n_embd_head, N, n_head, n_batch);
//         } else {
//             struct ggml_tensor * t16_0 = ggml_mul_mat              (ctx, t14, t13);                  set_name(t16_0, "t16_0"); assert_shape_4d(t16_0, N, N, n_head, n_batch);
//             struct ggml_tensor * t16_1 = ggml_scale_inplace        (ctx, t16_0, kv_scale);           set_name(t16_1, "t16_1"); assert_shape_4d(t16_1, N, N, n_head, n_batch);
//             struct ggml_tensor * t16_2 = ggml_diag_mask_inf_inplace(ctx, t16_1, n_past);             set_name(t16_2, "t16_2"); assert_shape_4d(t16_2, N, N, n_head, n_batch);
//             struct ggml_tensor * t16_3 = ggml_soft_max_inplace     (ctx, t16_2);                     set_name(t16_3, "t16_3"); assert_shape_4d(t16_3, N, N, n_head, n_batch);
//             t16 = ggml_mul_mat(ctx, t15, t16_3);                                                     set_name(t16, "t16");     assert_shape_4d(t16, n_embd_head, N, n_head, n_batch);
//         }
//         struct ggml_tensor * t17 = ggml_permute      (ctx, t16, 0, 2, 1, 3);                         set_name(t17, "t17");     assert_shape_4d(t17, n_embd_head, n_head, N, n_batch);
//         struct ggml_tensor * t18 = ggml_cont         (ctx, t17);                                     set_name(t18, "t18");     assert_shape_4d(t18, n_embd_head, n_head, N, n_batch);
//         struct ggml_tensor * t19 = ggml_reshape_2d   (ctx, t18, n_embd, N*n_batch);                  set_name(t19, "t19");     assert_shape_2d(t19, n_embd, N*n_batch);
//         struct ggml_tensor * t20 = ggml_mul_mat      (ctx, wo, t19);                                 set_name(t20, "t20");     assert_shape_2d(t20, n_embd, N*n_batch);
//         struct ggml_tensor * t21 = ggml_add          (ctx, t20, cur);                                set_name(t21, "t21");     assert_shape_2d(t21, n_embd, N*n_batch);
//         struct ggml_tensor * t22 = ggml_rms_norm     (ctx, t21, rms_norm_eps);                       set_name(t22, "t22");     assert_shape_2d(t22, n_embd, N*n_batch);
//         struct ggml_tensor * t23 = ggml_repeat       (ctx, ffn_norm, t22);                           set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
//         struct ggml_tensor * t24 = ggml_mul          (ctx, t23, t22);                                set_name(t24, "t24");     assert_shape_2d(t24, n_embd, N*n_batch);
//         struct ggml_tensor * t25 = ggml_mul_mat      (ctx, w3, t24);                                 set_name(t25, "t25");     assert_shape_2d(t25, n_ff, N*n_batch);
//         struct ggml_tensor * t26 = ggml_mul_mat      (ctx, w1, t24);                                 set_name(t26, "t26");     assert_shape_2d(t26, n_ff, N*n_batch);
//         struct ggml_tensor * t27 = ggml_silu         (ctx, t26);                                     set_name(t27, "t27");     assert_shape_2d(t27, n_ff, N*n_batch);
//         struct ggml_tensor * t28 = ggml_mul          (ctx, t27, t25);                                set_name(t28, "t28");     assert_shape_2d(t28, n_ff, N*n_batch);
//         struct ggml_tensor * t29 = ggml_mul_mat      (ctx, w2, t28);                                 set_name(t29, "t29");     assert_shape_2d(t29, n_embd, N*n_batch);
//         struct ggml_tensor * t30 = ggml_add          (ctx, t29, t21);                                set_name(t30, "t30");     assert_shape_2d(t30, n_embd, N*n_batch);
//         cur = t30;
//         if (enable_checkpointing) {
//             checkpoints.push_back(cur);
//         }
//     }
//     struct ggml_tensor * t31   = ggml_rms_norm          (ctx, cur, rms_norm_eps);                    set_name(t31, "t31");     assert_shape_2d(t31, n_embd, N*n_batch);
//     struct ggml_tensor * t32   = ggml_repeat            (ctx, norm, t31);                            set_name(t32, "t32");     assert_shape_2d(t32, n_embd, N*n_batch);
//     struct ggml_tensor * t33   = ggml_mul               (ctx, t32, t31);                             set_name(t33, "t33");     assert_shape_2d(t33, n_embd, N*n_batch);
//     struct ggml_tensor * t34   = ggml_mul_mat           (ctx, output, t33);                          set_name(t34, "t34");     assert_shape_2d(t34, n_vocab, N*n_batch);
//     struct ggml_tensor * t35   = ggml_reshape_3d        (ctx, t34, n_vocab, N, n_batch);             set_name(t35, "t35");     assert_shape_3d(t35, n_vocab, N, n_batch);
//     struct ggml_tensor * t36   = ggml_cross_entropy_loss(ctx, t35, targets);                         set_name(t36, "t36");     assert_shape_1d(t36, 1);

//     if (enable_checkpointing) {
//         checkpoints.push_back(t31);
//         checkpoints.push_back(t32);
//         checkpoints.push_back(t33);
//         checkpoints.push_back(t34);
//         checkpoints.push_back(t35);
//         checkpoints.push_back(t36);
//     }

//     ggml_build_forward_expand(gf, t36);

//     if (enable_checkpointing) {
//         ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints.data(), (int) checkpoints.size());
//     } else {
//         *gb = *gf;
//         ggml_build_backward_expand(ctx, gf, gb, true);
//     }

//     GGML_ASSERT(alloc != NULL);

//     // make sure some tensors are not reallocated by inserting new temporary nodes depending on them
//     int n_leafs_before = gb->n_leafs;
//     int n_nodes_before = gb->n_nodes;
//     struct ggml_tensor * one = ggml_new_f32(ctx, 1.0f);
//     // output tensors
//     ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t35, one));
//     ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36, one));
//     // input gradient
//     ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36->grad, one));
//     GGML_ASSERT(t36->grad->data == NULL && t36->grad->view_src == NULL);
//     ggml_allocr_alloc(alloc, t36->grad);
//     // KQ_pos
//     ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, KQ_pos, one));

//     // make sure base model tensors data cannot be used in viewable operations
//     ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, model->tok_embeddings, one));
//     ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, model->norm, one));
//     ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, model->output, one));
//     for (int il = 0; il < n_layer; ++il) {
//         struct my_llama_layer & layer = model->layers[il];
//         ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.attention_norm, one));
//         ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.ffn_norm, one));
//         ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wq, one));
//         ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wk, one));
//         ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wv, one));
//         ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wo, one));
//         ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.w1, one));
//         ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.w2, one));
//         ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.w3, one));
//     }

//     // allocating checkpoints in one block to reduce memory fragmentation
//     // note: they will be freed in reverse order
//     for (unsigned int i = 0; i < checkpoints.size(); ++i) {
//         if (checkpoints[i]->data == NULL && checkpoints[i]->view_src == NULL) {
//             ggml_allocr_alloc(alloc, checkpoints[i]);
//         }
//     }

//     ggml_allocr_alloc_graph(alloc, gb);

//     // remove the additional nodes and leafs
//     for (int i = n_leafs_before; i < gb->n_leafs; ++i) {
//         gb->leafs[i] = NULL;
//     }
//     for (int i = n_nodes_before; i < gb->n_nodes; ++i) {
//         gb->nodes[i] = NULL;
//     }
//     gb->n_leafs = n_leafs_before;
//     gb->n_nodes = n_nodes_before;

//     *logits = t35;
//     return t36;
// }

//  void load_llama_lora_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct my_llama_model * model, struct my_llama_lora * lora) {
//     // NOTE: gguf_context must be initialized with f_ggml_ctx and no_alloc=false, otherwise tensor data can not be read

//     std::string arch;

//     std::vector<char> keybuf;
//     keybuf.resize(512);

//     GGUF_GET_KEY(fctx, arch, gguf_get_val_str, GGUF_TYPE_STRING, true, LLM_KV_GENERAL_ARCHITECTURE);
//     GGML_ASSERT(arch == "llama");

//     uint32_t ftype_u;
//     GGUF_GET_KEY(fctx, ftype_u, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_GENERAL_FILE_TYPE);
//     GGML_ASSERT((enum llama_ftype) ftype_u == LLAMA_FTYPE_ALL_F32);

//     struct my_llama_hparams hparams;
//     load_model_hparams_gguf(fctx, &hparams, arch.c_str());

//     // parameters that define tensor shapes must match
//     GGML_ASSERT(hparams.n_embd    == model->hparams.n_embd);
//     GGML_ASSERT(hparams.n_ff      == model->hparams.n_ff);
//     GGML_ASSERT(hparams.n_head    == model->hparams.n_head);
//     GGML_ASSERT(hparams.n_head_kv == model->hparams.n_head_kv);
//     GGML_ASSERT(hparams.n_layer   == model->hparams.n_layer);

//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_tok_embeddings, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_TOKEN_EMBD);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_norm,           gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_OUTPUT_NORM);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_output,         gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_OUTPUT);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_attention_norm, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_ATTN_NORM);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_wq,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_ATTN_Q);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_wk,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_ATTN_K);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_wv,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_ATTN_V);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_wo,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_ATTN_OUT);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_ffn_norm,       gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_FFN_NORM);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_w1,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_FFN_GATE);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_w2,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_FFN_DOWN);
//     GGUF_GET_KEY(fctx, lora->hparams.n_rank_w3,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_FFN_UP);

//     init_lora(model, lora);

//     copy_tensor_by_name(lora->tok_embeddings_a, f_ggml_ctx, ggml_get_name(lora->tok_embeddings_a));
//     copy_tensor_by_name(lora->tok_embeddings_b, f_ggml_ctx, ggml_get_name(lora->tok_embeddings_b));
//     copy_tensor_by_name(lora->norm_a,           f_ggml_ctx, ggml_get_name(lora->norm_a));
//     copy_tensor_by_name(lora->norm_b,           f_ggml_ctx, ggml_get_name(lora->norm_b));
//     copy_tensor_by_name(lora->output_a,         f_ggml_ctx, ggml_get_name(lora->output_a));
//     copy_tensor_by_name(lora->output_b,         f_ggml_ctx, ggml_get_name(lora->output_b));

//     for (uint32_t i = 0; i < lora->layers.size(); ++i) {
//         auto & layer = lora->layers[i];
//         copy_tensor_by_name(layer.attention_norm_a, f_ggml_ctx, ggml_get_name(layer.attention_norm_a));
//         copy_tensor_by_name(layer.attention_norm_b, f_ggml_ctx, ggml_get_name(layer.attention_norm_b));
//         copy_tensor_by_name(layer.wq_a,             f_ggml_ctx, ggml_get_name(layer.wq_a));
//         copy_tensor_by_name(layer.wq_b,             f_ggml_ctx, ggml_get_name(layer.wq_b));
//         copy_tensor_by_name(layer.wk_a,             f_ggml_ctx, ggml_get_name(layer.wk_a));
//         copy_tensor_by_name(layer.wk_b,             f_ggml_ctx, ggml_get_name(layer.wk_b));
//         copy_tensor_by_name(layer.wv_a,             f_ggml_ctx, ggml_get_name(layer.wv_a));
//         copy_tensor_by_name(layer.wv_b,             f_ggml_ctx, ggml_get_name(layer.wv_b));
//         copy_tensor_by_name(layer.wo_a,             f_ggml_ctx, ggml_get_name(layer.wo_a));
//         copy_tensor_by_name(layer.wo_b,             f_ggml_ctx, ggml_get_name(layer.wo_b));
//         copy_tensor_by_name(layer.ffn_norm_a,       f_ggml_ctx, ggml_get_name(layer.ffn_norm_a));
//         copy_tensor_by_name(layer.ffn_norm_b,       f_ggml_ctx, ggml_get_name(layer.ffn_norm_b));
//         copy_tensor_by_name(layer.w1_a,             f_ggml_ctx, ggml_get_name(layer.w1_a));
//         copy_tensor_by_name(layer.w1_b,             f_ggml_ctx, ggml_get_name(layer.w1_b));
//         copy_tensor_by_name(layer.w2_a,             f_ggml_ctx, ggml_get_name(layer.w2_a));
//         copy_tensor_by_name(layer.w2_b,             f_ggml_ctx, ggml_get_name(layer.w2_b));
//         copy_tensor_by_name(layer.w3_a,             f_ggml_ctx, ggml_get_name(layer.w3_a));
//         copy_tensor_by_name(layer.w3_b,             f_ggml_ctx, ggml_get_name(layer.w3_b));
//     }
// }

//  void save_llama_lora_gguf(struct gguf_context * fctx, struct my_llama_model * model, struct my_llama_lora * lora) {
//     const char * arch = "llama";
//     enum llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

//     std::vector<char> keybuf;
//     keybuf.resize(512);
//     auto kv = [arch, &keybuf](const char * key) -> const char * {
//         snprintf(keybuf.data(), keybuf.size(), key, arch);
//         return keybuf.data();
//     };

//     gguf_set_val_str(fctx, LLM_KV_GENERAL_ARCHITECTURE, arch);
//     gguf_set_val_u32(fctx, LLM_KV_GENERAL_FILE_TYPE, ftype);

//     gguf_set_val_u32(fctx, kv(LLM_KV_CONTEXT_LENGTH),              model->hparams.n_ctx);
//     gguf_set_val_u32(fctx, kv(LLM_KV_EMBEDDING_LENGTH),            model->hparams.n_embd);
//     gguf_set_val_u32(fctx, kv(LLM_KV_FEED_FORWARD_LENGTH),         model->hparams.n_ff);
//     gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT),        model->hparams.n_head);
//     gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT_KV),     model->hparams.n_head_kv);
//     gguf_set_val_u32(fctx, kv(LLM_KV_BLOCK_COUNT),                 model->hparams.n_layer);
//     gguf_set_val_u32(fctx, kv(LLM_KV_ROPE_DIMENSION_COUNT),        model->hparams.n_embd_head());
//     gguf_set_val_f32(fctx, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS), model->hparams.f_norm_rms_eps);
//     gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_FREQ_BASE),              model->hparams.rope_freq_base);
//     gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_SCALE_LINEAR),           model->hparams.rope_freq_scale);

//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_TOKEN_EMBD,   lora->hparams.n_rank_tok_embeddings);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_OUTPUT_NORM,  lora->hparams.n_rank_norm);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_OUTPUT,       lora->hparams.n_rank_output);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_ATTN_NORM,    lora->hparams.n_rank_attention_norm);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_ATTN_Q,       lora->hparams.n_rank_wq);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_ATTN_K,       lora->hparams.n_rank_wk);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_ATTN_V,       lora->hparams.n_rank_wv);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_ATTN_OUT,     lora->hparams.n_rank_wo);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_FFN_NORM,     lora->hparams.n_rank_ffn_norm);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_FFN_GATE,     lora->hparams.n_rank_w1);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_FFN_DOWN,     lora->hparams.n_rank_w2);
//     gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_FFN_UP,       lora->hparams.n_rank_w3);

//     gguf_add_tensor(fctx, lora->tok_embeddings_a);
//     gguf_add_tensor(fctx, lora->tok_embeddings_b);
//     gguf_add_tensor(fctx, lora->norm_a);
//     gguf_add_tensor(fctx, lora->norm_b);
//     gguf_add_tensor(fctx, lora->output_a);
//     gguf_add_tensor(fctx, lora->output_b);

//     for (uint32_t i = 0; i < lora->layers.size(); ++i) {
//         auto & layer = lora->layers[i];

//         gguf_add_tensor(fctx, layer.attention_norm_a);
//         gguf_add_tensor(fctx, layer.attention_norm_b);
//         gguf_add_tensor(fctx, layer.wq_a);
//         gguf_add_tensor(fctx, layer.wq_b);
//         gguf_add_tensor(fctx, layer.wk_a);
//         gguf_add_tensor(fctx, layer.wk_b);
//         gguf_add_tensor(fctx, layer.wv_a);
//         gguf_add_tensor(fctx, layer.wv_b);
//         gguf_add_tensor(fctx, layer.wo_a);
//         gguf_add_tensor(fctx, layer.wo_b);
//         gguf_add_tensor(fctx, layer.ffn_norm_a);
//         gguf_add_tensor(fctx, layer.ffn_norm_b);
//         gguf_add_tensor(fctx, layer.w1_a);
//         gguf_add_tensor(fctx, layer.w1_b);
//         gguf_add_tensor(fctx, layer.w2_a);
//         gguf_add_tensor(fctx, layer.w2_b);
//         gguf_add_tensor(fctx, layer.w3_a);
//         gguf_add_tensor(fctx, layer.w3_b);
//     }
// }

//  void load_checkpoint_lora_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct my_llama_model * model, struct my_llama_lora * lora, struct train_state * train) {
//     std::string train_type = LLM_KV_TRAINING_TYPE_FINETUNE_LORA;
//     GGUF_GET_KEY(fctx, train_type, gguf_get_val_str, GGUF_TYPE_STRING, false, LLM_KV_TRAINING_TYPE);
//     GGML_ASSERT(train_type == LLM_KV_TRAINING_TYPE_FINETUNE_LORA);

//     load_train_state_gguf(fctx, f_ggml_ctx, train);
//     load_llama_lora_gguf(fctx, f_ggml_ctx, model, lora);
// }

//  void save_checkpoint_lora_gguf(struct gguf_context * fctx, struct my_llama_model * model, struct my_llama_lora * lora, struct train_state * train) {
//     gguf_set_val_str(fctx, LLM_KV_TRAINING_TYPE, LLM_KV_TRAINING_TYPE_FINETUNE_LORA);
//     save_llama_lora_gguf(fctx, model, lora);
//     save_train_state_gguf(fctx, train);
// }

//  bool load_checkpoint_lora_file(const char * filename, struct my_llama_model * model, struct my_llama_lora * lora, struct train_state * train) {
//     struct ggml_context * f_ggml_ctx;
//     struct gguf_init_params params;
//     params.no_alloc = false;
//     params.ctx = &f_ggml_ctx;
//     struct gguf_context * fctx = gguf_init_from_file(filename, params);
//     if (fctx == NULL) {
//         return false;
//     }

//     load_checkpoint_lora_gguf(fctx, f_ggml_ctx, model, lora, train);

//     gguf_free(fctx);
//     return true;
// }

//  void save_checkpoint_lora_file(const char * filename, struct my_llama_model * model, struct my_llama_lora * lora, struct train_state * train) {
//     printf("%s: saving to %s\n", __func__, filename);
//     struct gguf_context * fctx = gguf_init_empty();

//     save_checkpoint_lora_gguf(fctx, model, lora, train);

//     // write file
//     const bool only_meta = false;
//     gguf_write_to_file(fctx, filename, only_meta);
//     gguf_free(fctx);
// }

// struct llama_file {
//     // use FILE * so we don't have to re-open the file to mmap
//     FILE * fp;
//     size_t size;

//     llama_file(const char * fname, const char * mode) {
//         fp = std::fopen(fname, mode);
//         if (fp == NULL) {
//             size = 0;
//         } else {
//             seek(0, SEEK_END);
//             size = tell();
//             seek(0, SEEK_SET);
//         }
//     }

//     size_t tell() const {
// #ifdef _WIN32
//         __int64 ret = _ftelli64(fp);
// #else
//         long ret = std::ftell(fp);
// #endif
//         GGML_ASSERT(ret != -1); // this really shouldn't fail
//         return (size_t) ret;
//     }

//     void seek(size_t offset, int whence) {
// #ifdef _WIN32
//         int ret = _fseeki64(fp, (__int64) offset, whence);
// #else
//         int ret = std::fseek(fp, (long) offset, whence);
// #endif
//         GGML_ASSERT(ret == 0); // same
//     }

//     void read_raw(void * ptr, size_t size) {
//         if (size == 0) {
//             return;
//         }
//         errno = 0;
//         std::size_t ret = std::fread(ptr, size, 1, fp);
//         if (ferror(fp)) {
//             die_fmt("read error: %s", strerror(errno));
//         }
//         if (ret != 1) {
//             die("unexpectedly reached end of file");
//         }
//     }

//     std::uint32_t read_u32() {
//         std::uint32_t ret;
//         read_raw(&ret, sizeof(ret));
//         return ret;
//     }

//     std::string read_string(std::uint32_t len) {
//         std::vector<char> chars(len);
//         read_raw(chars.data(), len);
//         return std::string(chars.data(), len);
//     }

//     void write_raw(const void * ptr, size_t size) {
//         if (size == 0) {
//             return;
//         }
//         errno = 0;
//         size_t ret = std::fwrite(ptr, size, 1, fp);
//         if (ret != 1) {
//             die_fmt("write error: %s", strerror(errno));
//         }
//     }

//     void write_u32(std::uint32_t val) {
//         write_raw(&val, sizeof(val));
//     }

//     ~llama_file() {
//         if (fp) {
//             std::fclose(fp);
//         }
//     }
// };

//  void write_tensor(struct llama_file * file, struct ggml_tensor * tensor, const char * name) {
//     if (tensor == NULL) {
//         file->write_u32(0);
//         file->write_u32(0);
//         file->write_u32(GGML_TYPE_F32);
//         file->seek((0-file->tell()) & 31, SEEK_CUR);
//         return;
//     }
//     if (name == NULL) {
//         name = ggml_get_name(tensor);
//     }
//     uint32_t name_len = strlen(name);
//     uint32_t nd = tensor->n_dims;
//     uint32_t ne[4] = { (uint32_t)tensor->ne[0],
//                        (uint32_t)tensor->ne[1],
//                        (uint32_t)tensor->ne[2],
//                        (uint32_t)tensor->ne[3] };
//     file->write_u32(nd);
//     file->write_u32(name_len);
//     file->write_u32(tensor->type);
//     file->write_raw(ne, sizeof(ne[0]) * nd);
//     file->write_raw(name, name_len);
//     file->seek((0-file->tell()) & 31, SEEK_CUR);
//     file->write_raw(tensor->data, ggml_nbytes(tensor));
// }

//  void save_as_llama_lora(const char * filename, struct my_llama_lora * lora) {
//     printf("%s: saving to %s\n", __func__, filename);
//     struct llama_file file(filename, "wb");
//     if (file.fp == NULL) {
//         return;
//     }

//     std::vector<char> tn_buf;
//     tn_buf.resize(GGML_MAX_NAME);

//     auto tn = [&tn_buf](const char * key, const char * suffix) -> const char * {
//         snprintf(tn_buf.data(), tn_buf.size(), "%s%s", key, suffix);
//         return tn_buf.data();
//     };

//     auto tni = [&tn_buf](const char * key, int bid, const char * suffix) -> const char * {
//         snprintf(tn_buf.data(), tn_buf.size(), key, bid);
//         std::string s = tn_buf.data();
//         snprintf(tn_buf.data(), tn_buf.size(), "%s%s", s.c_str(), suffix);
//         return tn_buf.data();
//     };

//     uint32_t LLAMA_FILE_MAGIC_LORA = 0x67676C61; // 'ggla'
//     // write_magic
//     file.write_u32(LLAMA_FILE_MAGIC_LORA);   // magic
//     file.write_u32(1); // version
//     // write_hparams
//     file.write_u32(lora->hparams.lora_r);
//     file.write_u32(lora->hparams.lora_alpha);
//     // write tensors
//     write_tensor(&file, lora->tok_embeddings_a, tn(LLM_TENSOR_TOKEN_EMBD,  ".weight.loraA"));
//     write_tensor(&file, lora->tok_embeddings_b, tn(LLM_TENSOR_TOKEN_EMBD,  ".weight.loraB"));
//     write_tensor(&file, lora->norm_a,           tn(LLM_TENSOR_OUTPUT_NORM, ".weight.loraA"));
//     write_tensor(&file, lora->norm_b,           tn(LLM_TENSOR_OUTPUT_NORM, ".weight.loraB"));
//     write_tensor(&file, lora->output_a,         tn(LLM_TENSOR_OUTPUT,      ".weight.loraA"));
//     write_tensor(&file, lora->output_b,         tn(LLM_TENSOR_OUTPUT,      ".weight.loraB"));
//     for (uint32_t i = 0; i < lora->layers.size(); ++i) {
//         auto & layer = lora->layers[i];
//         write_tensor(&file, layer.attention_norm_a, tni(LLM_TENSOR_ATTN_NORM, i, ".weight.loraA"));
//         write_tensor(&file, layer.attention_norm_b, tni(LLM_TENSOR_ATTN_NORM, i, ".weight.loraB"));
//         write_tensor(&file, layer.wq_a,             tni(LLM_TENSOR_ATTN_Q,    i, ".weight.loraA"));
//         write_tensor(&file, layer.wq_b,             tni(LLM_TENSOR_ATTN_Q,    i, ".weight.loraB"));
//         write_tensor(&file, layer.wk_a,             tni(LLM_TENSOR_ATTN_K,    i, ".weight.loraA"));
//         write_tensor(&file, layer.wk_b,             tni(LLM_TENSOR_ATTN_K,    i, ".weight.loraB"));
//         write_tensor(&file, layer.wv_a,             tni(LLM_TENSOR_ATTN_V,    i, ".weight.loraA"));
//         write_tensor(&file, layer.wv_b,             tni(LLM_TENSOR_ATTN_V,    i, ".weight.loraB"));
//         write_tensor(&file, layer.wo_a,             tni(LLM_TENSOR_ATTN_OUT,  i, ".weight.loraA"));
//         write_tensor(&file, layer.wo_b,             tni(LLM_TENSOR_ATTN_OUT,  i, ".weight.loraB"));
//         write_tensor(&file, layer.ffn_norm_a,       tni(LLM_TENSOR_FFN_NORM,  i, ".weight.loraA"));
//         write_tensor(&file, layer.ffn_norm_b,       tni(LLM_TENSOR_FFN_NORM,  i, ".weight.loraB"));
//         write_tensor(&file, layer.w1_a,             tni(LLM_TENSOR_FFN_GATE,  i, ".weight.loraA"));
//         write_tensor(&file, layer.w1_b,             tni(LLM_TENSOR_FFN_GATE,  i, ".weight.loraB"));
//         write_tensor(&file, layer.w2_a,             tni(LLM_TENSOR_FFN_DOWN,  i, ".weight.loraA"));
//         write_tensor(&file, layer.w2_b,             tni(LLM_TENSOR_FFN_DOWN,  i, ".weight.loraB"));
//         write_tensor(&file, layer.w3_a,             tni(LLM_TENSOR_FFN_UP,    i, ".weight.loraA"));
//         write_tensor(&file, layer.w3_b,             tni(LLM_TENSOR_FFN_UP,    i, ".weight.loraB"));
//     }
// }

// struct train_params {
//     struct train_params_common common;

//     const char * fn_model_base;
//     const char * fn_lora_out;

//     bool only_write_lora;

//     float f_norm_rms_eps;
//     float rope_freq_base;
//     float rope_freq_scale;

//     bool custom_f_norm_rms_eps;
//     bool custom_rope_freq_base;
//     bool custom_rope_freq_scale;

//     int32_t lora_r;
//     int32_t lora_alpha;
//     bool custom_lora_alpha;

//     uint32_t n_rank_attention_norm;
//     uint32_t n_rank_wq;
//     uint32_t n_rank_wk;
//     uint32_t n_rank_wv;
//     uint32_t n_rank_wo;
//     uint32_t n_rank_ffn_norm;
//     uint32_t n_rank_w1;
//     uint32_t n_rank_w2;
//     uint32_t n_rank_w3;
//     uint32_t n_rank_tok_embeddings;
//     uint32_t n_rank_norm;
//     uint32_t n_rank_output;

//     bool custom_n_rank_attention_norm;
//     bool custom_n_rank_wq;
//     bool custom_n_rank_wk;
//     bool custom_n_rank_wv;
//     bool custom_n_rank_wo;
//     bool custom_n_rank_ffn_norm;
//     bool custom_n_rank_w1;
//     bool custom_n_rank_w2;
//     bool custom_n_rank_w3;
//     bool custom_n_rank_tok_embeddings;
//     bool custom_n_rank_norm;
//     bool custom_n_rank_output;
// };

//  struct train_params get_default_train_params() {
//     struct train_params params;
//     params.common = get_default_train_params_common();
//     params.fn_model_base     = "";
//     params.fn_lora_out       = "ggml-lora-ITERATION-f32.gguf";

//     params.only_write_lora = false;

//     params.f_norm_rms_eps  = 1e-5f;
//     params.rope_freq_base  = 10000.0f;
//     params.rope_freq_scale = 1.0f;

//     params.custom_f_norm_rms_eps  = false;
//     params.custom_rope_freq_base  = false;
//     params.custom_rope_freq_scale = false;

//     params.lora_r      = 4;
//     params.lora_alpha  = 4;
//     params.custom_lora_alpha = false;

//     params.n_rank_attention_norm = 1;
//     params.n_rank_wq             = 4;
//     params.n_rank_wk             = 4;
//     params.n_rank_wv             = 4;
//     params.n_rank_wo             = 4;
//     params.n_rank_ffn_norm       = 1;
//     params.n_rank_w1             = 4;
//     params.n_rank_w2             = 4;
//     params.n_rank_w3             = 4;
//     params.n_rank_tok_embeddings = 4;
//     params.n_rank_norm           = 1;
//     params.n_rank_output         = 4;

//     params.custom_n_rank_attention_norm = false;
//     params.custom_n_rank_wq             = false;
//     params.custom_n_rank_wk             = false;
//     params.custom_n_rank_wv             = false;
//     params.custom_n_rank_wo             = false;
//     params.custom_n_rank_ffn_norm       = false;
//     params.custom_n_rank_w1             = false;
//     params.custom_n_rank_w2             = false;
//     params.custom_n_rank_w3             = false;
//     params.custom_n_rank_tok_embeddings = false;
//     params.custom_n_rank_norm           = false;
//     params.custom_n_rank_output         = false;

//     return params;
// }

//  void train_print_usage(int argc, char ** argv, const struct train_params * params) {
//     fprintf(stderr, "usage: %s [options]\n", argv[0]);
//     fprintf(stderr, "\n");
//     fprintf(stderr, "options:\n");
//     fprintf(stderr, "  -h, --help                 show this help message and exit\n");

//     fprintf(stderr, "  --model-base FNAME         model path from which to load base model (default '%s')\n", params->fn_model_base);
//     fprintf(stderr, "  --lora-out FNAME           path to save llama lora (default '%s')\n", params->fn_lora_out);
//     fprintf(stderr, "  --only-write-lora          only save llama lora, don't do any training.  use this if you only want to convert a checkpoint to a lora adapter.\n");
//     fprintf(stderr, "  --norm-rms-eps F           RMS-Norm epsilon value (default %f)\n", params->f_norm_rms_eps);
//     fprintf(stderr, "  --rope-freq-base F         Frequency base for ROPE (default %f)\n", params->rope_freq_base);
//     fprintf(stderr, "  --rope-freq-scale F        Frequency scale for ROPE (default %f)\n", params->rope_freq_scale);
//     fprintf(stderr, "  --lora-alpha N             LORA alpha : resulting LORA scaling is alpha/r. (default %d)\n", params->lora_alpha);
//     fprintf(stderr, "  --lora-r N                 LORA r: default rank. Also specifies resulting scaling together with lora-alpha. (default %d)\n", params->lora_r);
//     fprintf(stderr, "  --rank-att-norm N          LORA rank for attention norm tensor, overrides default rank. Norm tensors should generally have rank 1.\n");
//     fprintf(stderr, "  --rank-ffn-norm N          LORA rank for feed-forward norm tensor, overrides default rank. Norm tensors should generally have rank 1.\n");
//     fprintf(stderr, "  --rank-out-norm N          LORA rank for output norm tensor, overrides default rank. Norm tensors should generally have rank 1.\n");
//     fprintf(stderr, "  --rank-tok-embd N          LORA rank for token embeddings tensor, overrides default rank.\n");
//     fprintf(stderr, "  --rank-out N               LORA rank for output tensor, overrides default rank.\n");
//     fprintf(stderr, "  --rank-wq N                LORA rank for wq tensor, overrides default rank.\n");
//     fprintf(stderr, "  --rank-wk N                LORA rank for wk tensor, overrides default rank.\n");
//     fprintf(stderr, "  --rank-wv N                LORA rank for wv tensor, overrides default rank.\n");
//     fprintf(stderr, "  --rank-wo N                LORA rank for wo tensor, overrides default rank.\n");
//     fprintf(stderr, "  --rank-w1 N                LORA rank for w1 tensor, overrides default rank.\n");
//     fprintf(stderr, "  --rank-w2 N                LORA rank for w2 tensor, overrides default rank.\n");
//     fprintf(stderr, "  --rank-w3 N                LORA rank for w3 tensor, overrides default rank.\n");

//     print_common_train_usage(argc, argv, &params->common);
// }

//  bool train_params_parse(int argc, char ** argv, struct train_params * params) {
//     bool invalid_param = false;
//     std::string arg;
//     struct train_params default_params = get_default_train_params();
//     const std::string arg_prefix = "--";

//     for (int i = 1; i < argc; i++) {
//         arg = argv[i];
//         if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
//             std::replace(arg.begin(), arg.end(), '_', '-');
//         }

//         if (consume_common_train_arg(argc, argv, &i, &params->common, &invalid_param)) {
//             if (invalid_param) {
//                 break;
//             } else if (params->common.print_usage) {
//                 train_print_usage(argc, argv, &default_params);
//                 exit(0);
//             }
//         } else if (arg == "--model-base") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->fn_model_base = argv[i];
//         } else if (arg == "--lora-out") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->fn_lora_out = argv[i];
//         } else if (arg == "--only-write-lora") {
//             params->only_write_lora = true;
//         } else if (arg == "--norm-rms-eps") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->f_norm_rms_eps = std::stof(argv[i]);
//             params->custom_f_norm_rms_eps = true;
//         } else if (arg == "--rope-freq-base") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->rope_freq_base = std::stof(argv[i]);
//             params->custom_rope_freq_base = true;
//         } else if (arg == "--rope-freq-scale") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->rope_freq_scale = std::stof(argv[i]);
//             params->custom_rope_freq_scale = true;
//         } else if (arg == "--lora-alpha") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->lora_alpha = std::stoi(argv[i]);
//             params->custom_lora_alpha = true;
//         } else if (arg == "--lora-r") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->lora_r = std::stoi(argv[i]);
//         } else if (arg == "--rank-att-norm") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_attention_norm = std::stoi(argv[i]);
//             params->custom_n_rank_attention_norm = true;
//         } else if (arg == "--rank-ffn-norm") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_ffn_norm = std::stoi(argv[i]);
//             params->custom_n_rank_ffn_norm = true;
//         } else if (arg == "--rank-out-norm") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_norm = std::stoi(argv[i]);
//             params->custom_n_rank_norm = true;
//         } else if (arg == "--rank-tok-embd") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_tok_embeddings = std::stoi(argv[i]);
//             params->custom_n_rank_tok_embeddings = true;
//         } else if (arg == "--rank-out") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_output = std::stoi(argv[i]);
//             params->custom_n_rank_output = true;
//         } else if (arg == "--rank-wq") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_wq = std::stoi(argv[i]);
//             params->custom_n_rank_wq = true;
//         } else if (arg == "--rank-wk") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_wk = std::stoi(argv[i]);
//             params->custom_n_rank_wk = true;
//         } else if (arg == "--rank-wv") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_wv = std::stoi(argv[i]);
//             params->custom_n_rank_wv = true;
//         } else if (arg == "--rank-wo") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_wo = std::stoi(argv[i]);
//             params->custom_n_rank_wo = true;
//         } else if (arg == "--rank-w1") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_w1 = std::stoi(argv[i]);
//             params->custom_n_rank_w1 = true;
//         } else if (arg == "--rank-w2") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_w2 = std::stoi(argv[i]);
//             params->custom_n_rank_w2 = true;
//         } else if (arg == "--rank-w3") {
//             if (++i >= argc) {
//                 invalid_param = true;
//                 break;
//             }
//             params->n_rank_w3 = std::stoi(argv[i]);
//             params->custom_n_rank_w3 = true;
//         } else {
//             fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
//             train_print_usage(argc, argv, &default_params);
//             exit(1);
//         }
//     }
//     if (invalid_param) {
//         fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
//         train_print_usage(argc, argv, &default_params);
//         exit(1);
//     }
//     finish_processing_train_args(&params->common);
//     return true;
// }

// struct save_train_files_data {
//     const char            * fn_checkpoint_out;
//     const char            * fn_lora_out;
//     const char            * pattern_fn_it;
//     const char            * fn_latest;
//     struct my_llama_model * model;
//     struct my_llama_lora  * lora;
// };

//  void save_train_files(void * vdata, struct train_state * train) {
//     struct save_train_files_data * data   = (struct save_train_files_data *) vdata;

//     int64_t iter = train->opt->iter;

//     if (strlen(data->fn_checkpoint_out) > 0) {
//         save_checkpoint_lora_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->model, data->lora, train);
//         save_checkpoint_lora_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->model, data->lora, train);
//     }
//     if (strlen(data->fn_lora_out) > 0) {
//         save_as_llama_lora(get_train_filename(data->fn_lora_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->lora);
//         save_as_llama_lora(get_train_filename(data->fn_lora_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->lora);
//     }
// }

//  int64_t get_parameter_count(struct my_llama_lora* lora) {
//     int64_t nx = 0;
//     nx += ggml_nelements(lora->tok_embeddings_a);
//     nx += ggml_nelements(lora->tok_embeddings_b);
//     nx += ggml_nelements(lora->norm_a);
//     nx += ggml_nelements(lora->norm_b);
//     nx += ggml_nelements(lora->output_a);
//     nx += ggml_nelements(lora->output_b);

//     for (uint32_t i = 0; i < lora->layers.size(); ++i) {
//         auto & layer = lora->layers[i];
//         nx += ggml_nelements(layer.attention_norm_a);
//         nx += ggml_nelements(layer.attention_norm_b);
//         nx += ggml_nelements(layer.wq_a);
//         nx += ggml_nelements(layer.wq_b);
//         nx += ggml_nelements(layer.wk_a);
//         nx += ggml_nelements(layer.wk_b);
//         nx += ggml_nelements(layer.wv_a);
//         nx += ggml_nelements(layer.wv_b);
//         nx += ggml_nelements(layer.wo_a);
//         nx += ggml_nelements(layer.wo_b);
//         nx += ggml_nelements(layer.ffn_norm_a);
//         nx += ggml_nelements(layer.ffn_norm_b);
//         nx += ggml_nelements(layer.w1_a);
//         nx += ggml_nelements(layer.w1_b);
//         nx += ggml_nelements(layer.w2_a);
//         nx += ggml_nelements(layer.w2_b);
//         nx += ggml_nelements(layer.w3_a);
//         nx += ggml_nelements(layer.w3_b);
//     }
//     return nx;
// }

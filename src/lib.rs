use std::{
    collections::HashMap,
    error::Error,
    ffi::{c_char, c_void, CStr, CString},
    mem::size_of,
    sync::Mutex,
};

use options::{ModelOptions, PredictOptions};

use lazy_static::lazy_static;
pub mod options;
//TODO: this should be programmatic, use syn on binding.rs and codegen or writer crate
//TODO: restore original bindings example so we can PR

include!(concat!(env!("OUT_DIR"), "/binding.rs"));
// include!("bindings.rs");
// include!(concat!(env!("OUT_DIR"), "/llama_bindings.rs"));

//TODO: going to need to wrap all the strings variables with a writer and codegen
//TODO: need to pass all argv into a builder instead, this is a framework not a cli app


//TODO: remove rs naming. once we get a builder this shouldnt be a problem.
//TODO: we are tracing the llama.cpp/examples/finetune/finetune.cpp main routine and extracting a sequence of functions that define each step
//NOTE: much of this is unecessary but we want to :
//      1. modularly debug each step  
//      2. ablate what shouldnt be encapsulated  
//      3. create a class builder wrapper

    // pub fn init_train_state() -> *mut train_state;
pub fn rs_init_train_state() -> *mut train_state {
    unsafe { init_train_state() }
}

pub fn rs_new_my_llama_model() -> *mut my_llama_model {
    println!("starting rs_new_my_llama_model");
    unsafe{
        let mut result = &mut init_my_llama_model();
        println!("finished rs_new_my_llama_model");
        return result;
    };
}
pub fn rs_init_model(input: *mut llama_model, model: *mut my_llama_model, fn_model: *const ::std::os::raw::c_char, n_ctx: u32) {
    println!("starting rs_init_model");
    unsafe { init_model(input, model, fn_model, n_ctx) };
    println!("finished rs_init_model");
}
//    struct train_params params = get_default_train_params();
pub fn rs_get_default_train_params() -> train_params {
    unsafe { get_default_train_params() }
}
pub fn set_seed(params: *mut train_params, seed: u32) {
    unsafe { (*params).common.seed = seed }
}

//    struct llama_model_params llama_mparams = llama_model_default_params();
pub fn rs_llama_model_default_params() -> llama_model_params {
    unsafe { llama_model_default_params() }
}

//    struct llama_model * lmodel = llama_load_model_from_file(params.fn_model_base, llama_mparams);
pub fn rs_llama_load_model_from_file(fn_model_base: *const c_char, llama_mparams: llama_model_params) -> *mut llama_model {
    unsafe { llama_load_model_from_file(fn_model_base, llama_mparams) }
}

//TODO: @DEPRECATED, keep unsafe wrapper low so we abstract clean to a proper builder 
////same as above but use rust string for fn_model_base and rs_llama_model_default_params() for llama_mparams
//pub fn rs_llama_load_model_from_file2(fn_model_base: String) -> *mut llama_model {
    //let c_str = CString::new(fn_model_base.clone()).unwrap();
    //let fn_model_base = c_str.as_ptr();
    //// let llama_mparams = rs_llama_model_default_params();
    //unsafe { 
    //let llama_mparams = llama_model_default_params();
        //llama_load_model_from_file(fn_model_base, llama_mparams) 
    //}
//}

//    struct llama_context_params llama_cparams = llama_context_default_params();
pub fn rs_llama_context_default_params() -> llama_context_params {
    unsafe { llama_context_default_params() }
}
//    struct llama_context * lctx = llama_new_context_with_model(lmodel, llama_cparams);
// pub fn rs_llama_new_context_with_model(lmodel: *mut llama_model, llama_cparams: llama_context_params) -> *mut llama_context {
pub fn rs_llama_new_context_with_model(lmodel: *mut llama_model, llama_cparams: llama_context_params) -> *mut llama_context {
    unsafe { llama_new_context_with_model(lmodel, llama_cparams) }
}

//initializer for defaults of my_llama_model (probably null since we always use a loader)
// pub fn rs_my_llama_model() -> my_llama_model {
// TODO
// }

//TODO: @DEPRECATED
//    init_model(lmodel, &model, params.fn_model_base, params.common.n_ctx);
// pub fn rs_init_model2(lmodel: *mut llama_model, model: *mut my_llama_model, fn_model_base: *mut c_char, n_ctx: u32) {
//     unsafe { init_model(lmodel, model, fn_model_base, n_ctx) }
// }
// pub fn print_params(params: *mut my_llama_hparams);
pub fn rs_print_params(params: &mut my_llama_hparams) {
    //USAGE: rs_print_params(my_llama_model.my_llama_hparams);
//    pub fn print_params(params: *mut my_llama_hparams);
unsafe { print_params(params) }
}

//TODO: TEST LOAD AND PRINT A GGUF


// lazy_static! {
//     static ref CALLBACKS: Mutex<HashMap<usize, Box<dyn Fn(String) -> bool + Send + 'static>>> =
//         Mutex::new(HashMap::new());
// }

// #[derive(Debug, Clone)]
// pub struct LLama {
//     state: *mut c_void,
//     embeddings: bool,
//     context_size: i32,
// }

// impl LLama {
//     pub fn new(model: String, opts: &ModelOptions) -> Result<Self, Box<dyn Error>> {
//         let model_path = CString::new(model).unwrap();

//         let main_gpu_cstr = CString::new(opts.main_gpu.clone()).unwrap();

//         let main_gpu = main_gpu_cstr.as_ptr();

//         let tensor_split_cstr = CString::new(opts.tensor_split.clone()).unwrap();

//         let tensor_split = tensor_split_cstr.as_ptr();

//         unsafe {
//             let result = load_model(
//                 model_path.as_ptr(),
//                 opts.context_size,
//                 opts.seed,
//                 opts.f16_memory,
//                 opts.m_lock,
//                 opts.embeddings,
//                 opts.m_map,
//                 opts.low_vram,
//                 opts.vocab_only,
//                 opts.n_gpu_layers,
//                 opts.n_batch,
//                 main_gpu,
//                 tensor_split,
//                 opts.numa,
//             );

//             if result == std::ptr::null_mut() {
//                 return Err("Failed to load model".into());
//             } else {
//                 Ok(Self {
//                     state: result,
//                     embeddings: opts.embeddings,
//                     context_size: opts.context_size,
//                 })
//             }
//         }
//     }

//     //TODO: other wrappers here for other functions.
//     pub fn free_model(&self) {
//         unsafe {
//             llama_binding_free_model(self.state);
//         }
//     }

//     pub fn load_state(&self, state: String) -> Result<(), Box<dyn Error>> {
//         let d = CString::new(state).unwrap().into_raw();
//         let w = CString::new("rb").unwrap().into_raw();

//         unsafe {
//             let result = load_state(self.state, d, w);

//             if result != 0 {
//                 return Err("Failed to load state".into());
//             } else {
//                 Ok(())
//             }
//         }
//     }

//     pub fn save_state(&self, dst: String) -> Result<(), Box<dyn Error>> {
//         let d = CString::new(dst.clone()).unwrap().into_raw();
//         let w = CString::new("wb").unwrap().into_raw();

//         unsafe {
//             save_state(self.state, d, w);
//         };

//         std::fs::metadata(dst).map_err(|_| "Failed to save state".to_string())?;

//         Ok(())
//     }

//     pub fn eval(&self, text: String, opts: &mut PredictOptions) -> Result<(), Box<dyn Error>> {
//         let c_str = CString::new(text.clone()).unwrap();

//         let input = c_str.as_ptr();

//         let input2 = c_str.into_raw();

//         if opts.tokens == 0 {
//             opts.tokens = 99999999;
//         }

//         let reverse_count = opts.stop_prompts.len();

//         let mut c_strings: Vec<CString> = Vec::new();

//         let mut reverse_prompt = Vec::with_capacity(reverse_count);

//         let mut pass: *mut *const c_char = std::ptr::null_mut();

//         for prompt in &opts.stop_prompts {
//             let c_string = CString::new(prompt.clone()).unwrap();
//             reverse_prompt.push(c_string.as_ptr());
//             c_strings.push(c_string);
//         }

//         if !reverse_prompt.is_empty() {
//             pass = reverse_prompt.as_mut_ptr();
//         }

//         let logit_bias_cstr = CString::new(opts.logit_bias.clone()).unwrap();

//         let logit_bias = logit_bias_cstr.as_ptr();

//         let path_prompt_cache_cstr = CString::new(opts.path_prompt_cache.clone()).unwrap();

//         let path_prompt_cache = path_prompt_cache_cstr.as_ptr();

//         let main_gpu_cstr = CString::new(opts.main_gpu.clone()).unwrap();

//         let main_gpu = main_gpu_cstr.as_ptr();

//         let tensor_split_cstr = CString::new(opts.tensor_split.clone()).unwrap();

//         let tensor_split = tensor_split_cstr.as_ptr();

//         unsafe {
//             let params = llama_allocate_params(
//                 input,
//                 opts.seed,
//                 opts.threads,
//                 opts.tokens,
//                 opts.top_k,
//                 opts.top_p,
//                 opts.temperature,
//                 opts.penalty,
//                 opts.repeat,
//                 opts.ignore_eos,
//                 opts.f16_kv,
//                 opts.batch,
//                 opts.n_keep,
//                 pass,
//                 reverse_count as i32,
//                 opts.tail_free_sampling_z,
//                 opts.typical_p,
//                 opts.frequency_penalty,
//                 opts.presence_penalty,
//                 opts.mirostat,
//                 opts.mirostat_eta,
//                 opts.mirostat_tau,
//                 opts.penalize_nl,
//                 logit_bias,
//                 path_prompt_cache,
//                 opts.prompt_cache_all,
//                 opts.m_lock,
//                 opts.m_map,
//                 main_gpu,
//                 tensor_split,
//                 opts.prompt_cache_ro,
//             );

//             let ret = eval(params, self.state, input2);

//             if ret != 0 {
//                 return Err("Failed to predict".into());
//             }

//             llama_free_params(params);
//         }

//         Ok(())
//     }

//     pub fn token_embeddings(
//         &self,
//         tokens: Vec<i32>,
//         opts: &mut PredictOptions,
//     ) -> Result<Vec<f32>, Box<dyn Error>> {
//         if !self.embeddings {
//             return Err("model loaded without embeddings".into());
//         }

//         if opts.tokens == 0 {
//             opts.tokens = 99999999;
//         }

//         let mut out = Vec::with_capacity(opts.tokens as usize);

//         let mut my_array: Vec<i32> = Vec::with_capacity(opts.tokens as usize * size_of::<i32>());

//         for (i, &v) in tokens.iter().enumerate() {
//             my_array[i] = v;
//         }

//         let logit_bias_cstr = CString::new(opts.logit_bias.clone()).unwrap();

//         let logit_bias = logit_bias_cstr.as_ptr();

//         let path_prompt_cache_cstr = CString::new(opts.path_prompt_cache.clone()).unwrap();

//         let path_prompt_cache = path_prompt_cache_cstr.as_ptr();

//         let main_gpu_cstr = CString::new(opts.main_gpu.clone()).unwrap();

//         let main_gpu = main_gpu_cstr.as_ptr();

//         let tensor_split_cstr = CString::new(opts.tensor_split.clone()).unwrap();

//         let tensor_split = tensor_split_cstr.as_ptr();

//         let input = CString::new("").unwrap();

//         unsafe {
//             let params = llama_allocate_params(
//                 input.as_ptr(),
//                 opts.seed,
//                 opts.threads,
//                 opts.tokens,
//                 opts.top_k,
//                 opts.top_p,
//                 opts.temperature,
//                 opts.penalty,
//                 opts.repeat,
//                 opts.ignore_eos,
//                 opts.f16_kv,
//                 opts.batch,
//                 opts.n_keep,
//                 std::ptr::null_mut(),
//                 0,
//                 opts.tail_free_sampling_z,
//                 opts.typical_p,
//                 opts.frequency_penalty,
//                 opts.presence_penalty,
//                 opts.mirostat,
//                 opts.mirostat_eta,
//                 opts.mirostat_tau,
//                 opts.penalize_nl,
//                 logit_bias,
//                 path_prompt_cache,
//                 opts.prompt_cache_all,
//                 opts.m_lock,
//                 opts.m_map,
//                 main_gpu,
//                 tensor_split,
//                 opts.prompt_cache_ro,
//             );

//             let ret = get_token_embeddings(
//                 params,
//                 self.state,
//                 my_array.as_mut_ptr(),
//                 my_array.len() as i32,
//                 out.as_mut_ptr(),
//             );

//             if ret != 0 {
//                 return Err("Embedding inference failed".into());
//             }

//             Ok(out)
//         }
//     }

//     pub fn embeddings(
//         &self,
//         text: String,
//         opts: &mut PredictOptions,
//     ) -> Result<Vec<f32>, Box<dyn Error>> {
//         if !self.embeddings {
//             return Err("model loaded without embeddings".into());
//         }

//         let c_str = CString::new(text.clone()).unwrap();

//         let input = c_str.as_ptr();

//         if opts.tokens == 0 {
//             opts.tokens = 99999999;
//         }

//         let reverse_count = opts.stop_prompts.len();

//         let mut c_strings: Vec<CString> = Vec::new();

//         let mut reverse_prompt = Vec::with_capacity(reverse_count);

//         let mut pass: *mut *const c_char = std::ptr::null_mut();

//         for prompt in &opts.stop_prompts {
//             let c_string = CString::new(prompt.clone()).unwrap();
//             reverse_prompt.push(c_string.as_ptr());
//             c_strings.push(c_string);
//         }

//         if !reverse_prompt.is_empty() {
//             pass = reverse_prompt.as_mut_ptr();
//         }

//         let mut out = Vec::with_capacity(opts.tokens as usize);

//         let logit_bias_cstr = CString::new(opts.logit_bias.clone()).unwrap();

//         let logit_bias = logit_bias_cstr.as_ptr();

//         let path_prompt_cache_cstr = CString::new(opts.path_prompt_cache.clone()).unwrap();

//         let path_prompt_cache = path_prompt_cache_cstr.as_ptr();

//         let main_gpu_cstr = CString::new(opts.main_gpu.clone()).unwrap();

//         let main_gpu = main_gpu_cstr.as_ptr();

//         let tensor_split_cstr = CString::new(opts.tensor_split.clone()).unwrap();

//         let tensor_split = tensor_split_cstr.as_ptr();

//         unsafe {
//             let params = llama_allocate_params(
//                 input,
//                 opts.seed,
//                 opts.threads,
//                 opts.tokens,
//                 opts.top_k,
//                 opts.top_p,
//                 opts.temperature,
//                 opts.penalty,
//                 opts.repeat,
//                 opts.ignore_eos,
//                 opts.f16_kv,
//                 opts.batch,
//                 opts.n_keep,
//                 pass,
//                 reverse_count as i32,
//                 opts.tail_free_sampling_z,
//                 opts.typical_p,
//                 opts.frequency_penalty,
//                 opts.presence_penalty,
//                 opts.mirostat,
//                 opts.mirostat_eta,
//                 opts.mirostat_tau,
//                 opts.penalize_nl,
//                 logit_bias,
//                 path_prompt_cache,
//                 opts.prompt_cache_all,
//                 opts.m_lock,
//                 opts.m_map,
//                 main_gpu,
//                 tensor_split,
//                 opts.prompt_cache_ro,
//             );

//             let ret = get_embeddings(params, self.state, out.as_mut_ptr());

//             if ret != 0 {
//                 return Err("Embedding inference failed".into());
//             }

//             Ok(out)
//         }
//     }

//     pub fn set_token_callback(
//         &self,
//         callback: Option<Box<dyn Fn(String) -> bool + Send + 'static>>,
//     ) {
//         set_callback(self.state, callback);
//     }

//     pub fn predict(&self, text: String, opts: PredictOptions) -> Result<String, Box<dyn Error>> {
//         let c_str = CString::new(text.clone()).unwrap();

//         let input = c_str.as_ptr();
//         let mut opts = opts;

//         if opts.tokens == 0 {
//             opts.tokens = 99999999;
//         }

//         if let Some(callback) = opts.token_callback {
//             set_callback(self.state, Some(callback));
//         }

//         let reverse_count = opts.stop_prompts.len();

//         let mut c_strings: Vec<CString> = Vec::new();

//         let mut reverse_prompt = Vec::with_capacity(reverse_count);

//         let mut pass: *mut *const c_char = std::ptr::null_mut();

//         for prompt in &opts.stop_prompts {
//             let c_string = CString::new(prompt.clone()).unwrap();
//             reverse_prompt.push(c_string.as_ptr());
//             c_strings.push(c_string);
//         }

//         if !reverse_prompt.is_empty() {
//             pass = reverse_prompt.as_mut_ptr();
//         }

//         println!("count {}", reverse_count);

//         let mut out = Vec::with_capacity(opts.tokens as usize);

//         let logit_bias_cstr = CString::new(opts.logit_bias.clone()).unwrap();

//         let logit_bias = logit_bias_cstr.as_ptr();

//         let path_prompt_cache_cstr = CString::new(opts.path_prompt_cache.clone()).unwrap();

//         let path_prompt_cache = path_prompt_cache_cstr.as_ptr();

//         let main_gpu_cstr = CString::new(opts.main_gpu.clone()).unwrap();

//         let main_gpu = main_gpu_cstr.as_ptr();

//         let tensor_split_cstr = CString::new(opts.tensor_split.clone()).unwrap();

//         let tensor_split = tensor_split_cstr.as_ptr();

//         unsafe {
//             let params = llama_allocate_params(
//                 input,
//                 opts.seed,
//                 opts.threads,
//                 opts.tokens,
//                 opts.top_k,
//                 opts.top_p,
//                 opts.temperature,
//                 opts.penalty,
//                 opts.repeat,
//                 opts.ignore_eos,
//                 opts.f16_kv,
//                 opts.batch,
//                 opts.n_keep,
//                 pass,
//                 reverse_count as i32,
//                 opts.tail_free_sampling_z,
//                 opts.typical_p,
//                 opts.frequency_penalty,
//                 opts.presence_penalty,
//                 opts.mirostat,
//                 opts.mirostat_eta,
//                 opts.mirostat_tau,
//                 opts.penalize_nl,
//                 logit_bias,
//                 path_prompt_cache,
//                 opts.prompt_cache_all,
//                 opts.m_lock,
//                 opts.m_map,
//                 main_gpu,
//                 tensor_split,
//                 opts.prompt_cache_ro,
//             );

//             let ret = llama_predict(params, self.state, out.as_mut_ptr(), opts.debug_mode);

//             if ret != 0 {
//                 return Err("Failed to predict".into());
//             }

//             llama_free_params(params);

//             let c_str: &CStr = CStr::from_ptr(out.as_mut_ptr());
//             let mut res: String = c_str.to_str().unwrap().to_owned();

//             res = res.trim_start().to_string();
//             res = res.trim_start_matches(&text).to_string();
//             res = res.trim_start_matches('\n').to_string();

//             for s in &opts.stop_prompts {
//                 res = res.trim_end_matches(s).to_string();
//             }

//             Ok(res)
//         }
//     }
// }

// impl Drop for LLama {
//     fn drop(&mut self) {
//         self.free_model();
//     }
// }

// fn set_callback(
//     state: *mut c_void,
//     callback: Option<Box<dyn Fn(String) -> bool + Send + 'static>>,
// ) {
//     let mut callbacks = CALLBACKS.lock().unwrap();

//     if let Some(callback) = callback {
//         callbacks.insert(state as usize, callback);
//     } else {
//         callbacks.remove(&(state as usize));
//     }
// }

// #[no_mangle]
// extern "C" fn tokenCallback(state: *mut c_void, token: *const c_char) -> bool {
//     let mut callbacks = CALLBACKS.lock().unwrap();

//     if let Some(callback) = callbacks.get_mut(&(state as usize)) {
//         let c_str: &CStr = unsafe { CStr::from_ptr(token) };
//         let str_slice: &str = c_str.to_str().unwrap();
//         let string: String = str_slice.to_owned();

//         return callback(string);
//     }

//     true
// }

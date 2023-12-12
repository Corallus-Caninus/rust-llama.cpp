use std::env;
use std::path::PathBuf;

use cc::Build;

use syn;
//import to_tokens for item
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::process;

//import standard hashmap
use proc_macro2::TokenStream;
use quote::ToTokens;
use std::collections::HashMap;

//TODO: bind up all the original goodies too. UPDATE: this just needs a verify
fn compile_bindings(out_src_path: &PathBuf) {
    let bindings = bindgen::Builder::default()
        //we start from this file only because its the highest in the abstraction tree.
        //TODO: name this bindgen.c/hpp?
        .header("./finetune_binding.hpp")
        .clang_arg("-stdlib=libc++")
        .clang_arg("-std=c++11")
        .clang_arg("-I./llama.cpp/common")
        .clang_arg("-I./llama.cpp/")
        .allowlist_file("./finetune_binding.hpp")
        .allowlist_file("./llama.cpp/common/train.h")
        .allowlist_file("./llama.cpp/llama.h")
        .allowlist_file("vector")
        //ablate string since the union type generated isnt supported in bindgen yet
        .opaque_type("std::string")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(&out_src_path.join("binding.rs"))
        .expect("Couldn't write bindings!");
}

fn compile_raw_bindgen(out_src_path: &PathBuf) {
    let ggml = bindgen::Builder::default()
        .header("./llama.cpp/ggml.h")
        .blocklist_function("tokenCallback")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate ggml");

    ggml.write_to_file(&out_src_path.join("ggml.rs"))
        .expect("Couldn't write ggml!");
}

fn compile_opencl(cx: &mut Build, cxx: &mut Build) {
    cx.flag("-DGGML_USE_CLBLAST");
    cxx.flag("-DGGML_USE_CLBLAST");

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=OpenCL");
        println!("cargo:rustc-link-lib=clblast");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=OpenCL");
        println!("cargo:rustc-link-lib=clblast");
    }

    cxx.file("./llama.cpp/ggml-opencl.cpp");
}

fn compile_openblas(cx: &mut Build) {
    cx.flag("-DGGML_USE_OPENBLAS")
        .include("/usr/local/include/openblas")
        .include("/usr/local/include/openblas");
    println!("cargo:rustc-link-lib=openblas");
}

fn compile_blis(cx: &mut Build) {
    cx.flag("-DGGML_USE_OPENBLAS")
        .include("/usr/local/include/blis")
        .include("/usr/local/include/blis");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=blis");
}

fn compile_cuda(cxx_flags: &str) {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!(
            "cargo:rustc-link-search=native={}/targets/x86_64-linux/lib",
            cuda_path
        );
    }

    let libs = "cublas culibos cudart cublasLt pthread dl rt";

    for lib in libs.split_whitespace() {
        println!("cargo:rustc-link-lib={}", lib);
    }

    let mut nvcc = cc::Build::new();

    let env_flags = vec![
        ("LLAMA_CUDA_DMMV_X=32", "-DGGML_CUDA_DMMV_X"),
        ("LLAMA_CUDA_DMMV_Y=1", "-DGGML_CUDA_DMMV_Y"),
        ("LLAMA_CUDA_KQUANTS_ITER=2", "-DK_QUANTS_PER_ITERATION"),
    ];

    let nvcc_flags = "--forward-unknown-to-host-compiler -arch=all";

    for nvcc_flag in nvcc_flags.split_whitespace() {
        nvcc.flag(nvcc_flag);
    }

    for cxx_flag in cxx_flags.split_whitespace() {
        nvcc.flag(cxx_flag);
    }

    for env_flag in env_flags {
        let mut flag_split = env_flag.0.split("=");
        if let Ok(val) = std::env::var(flag_split.next().unwrap()) {
            nvcc.flag(&format!("{}={}", env_flag.1, val));
        } else {
            nvcc.flag(&format!("{}={}", env_flag.1, flag_split.next().unwrap()));
        }
    }

    nvcc.compiler("nvcc")
        .file("./llama.cpp/ggml-cuda.cu")
        .flag("-Wno-pedantic")
        .include("./llama.cpp/ggml-cuda.h")
        .compile("ggml-cuda");
}

//TODO: WIP, verify with llama.cpp Makefile 432,451
fn compile_hip(cxx_flags: &str) {
    println!("cargo:rustc-link-search=native=/usr/local/hip/lib64");
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");

    if let Ok(hip_path) = std::env::var("HIP_PATH") {
        println!(
            "cargo:rustc-link-search=native={}/targets/x86_64-linux/lib",
            hip_path
        );
    }

    let libs = "hipblas amdhip64 rocblas";

    for lib in libs.split_whitespace() {
        println!("cargo:rustc-link-lib={}", lib);
    }

    let mut hipcc = cc::Build::new();

    let env_flags = vec![
        ("LLAMA_CUDA_DMMV_X=32", "-DGGML_CUDA_DMMV_X"),
        ("LLAMA_CUDA_DMMV_Y=1", "-DGGML_CUDA_DMMV_Y"),
        ("LLAMA_CUDA_KQUANTS_ITER=2", "-DK_QUANTS_PER_ITERATION"),
    ];

    let hipcc_flags = "--forward-unknown-to-host-compiler -arch=all";

    for hipcc_flag in hipcc_flags.split_whitespace() {
        hipcc.flag(hipcc_flag);
    }

    for cxx_flag in cxx_flags.split_whitespace() {
        hipcc.flag(cxx_flag);
    }

    for env_flag in env_flags {
        let mut flag_split = env_flag.0.split("=");
        if let Ok(val) = std::env::var(flag_split.next().unwrap()) {
            hipcc.flag(&format!("{}={}", env_flag.1, val));
        } else {
            hipcc.flag(&format!("{}={}", env_flag.1, flag_split.next().unwrap()));
        }
    }

    hipcc
        .compiler("hipcc")
        .file("./llama.cpp/ggml-cuda.cu")
        .flag("-Wno-pedantic")
        .include("./llama.cpp/ggml-cuda.h")
        //TODO: could be hip or could be ggml-cuda
        .compile("hip");
}

fn compile_ggml(cx: &mut Build, cx_flags: &str) {
    for cx_flag in cx_flags.split_whitespace() {
        cx.flag(cx_flag);
    }

    cx.include("./llama.cpp")
        .file("./llama.cpp/ggml.c")
        .file("./llama.cpp/ggml-alloc.c")
        .file("./llama.cpp/k_quants.c")
        .cpp(false)
        .define("_GNU_SOURCE", None)
        .define("GGML_USE_K_QUANTS", None)
        .compile("ggml");
}

fn compile_metal(cx: &mut Build, cxx: &mut Build) {
    cx.flag("-DGGML_USE_METAL").flag("-DGGML_METAL_NDEBUG");
    cxx.flag("-DGGML_USE_METAL");

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=MetalKit");

    cx.include("./llama.cpp/ggml-metal.h")
        .file("./llama.cpp/ggml-metal.m");
}

fn compile_llama(cxx: &mut Build, cxx_flags: &str, out_src_path: &PathBuf, ggml_type: &str) {
    for cxx_flag in cxx_flags.split_whitespace() {
        cxx.flag(cxx_flag);
    }

    let ggml_obj =
        PathBuf::from(env::var("OUT_DIR").expect("No out dir found")).join("llama.cpp/ggml.o");

    cxx.object(ggml_obj);

    if !ggml_type.is_empty() {
        let ggml_feature_obj = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"))
            .join(format!("llama.cpp/ggml-{}.o", ggml_type));
        cxx.object(ggml_feature_obj);
    }

    cxx.shared_flag(true)
        .include("./llama.cpp")
        .file("./llama.cpp/llama.cpp")
        .file("./llama.cpp/common/train.cpp")
        .file("./llama.cpp/common/common.cpp")
        .file("./finetune_binding.cpp")
        .cpp(true)
        .compile("binding");
}

fn main() {
    //generate raw bindings and object files
    let out_src_path = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"));

    compile_bindings(&out_src_path);

    let mut cx_flags = String::from("");
    let mut cxx_flags = String::from("");

    // check if os is linux
    // if so, add -fPIC to cxx_flags
    if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
        cx_flags.push_str(" -std=c11 -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -pthread -march=native -mtune=native");
        cxx_flags.push_str(" -std=c++11 -Wall -Wdeprecated-declarations -Wunused-but-set-variable -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -fPIC -pthread -march=native -mtune=native");
    } else if cfg!(target_os = "windows") {
        cx_flags.push_str(" /W4 /Wall /wd4820 /wd4710 /wd4711 /wd4820 /wd4514");
        cxx_flags.push_str(" /W4 /Wall /wd4820 /wd4710 /wd4711 /wd4820 /wd4514");
    }

    let mut cx = cc::Build::new();

    let mut cxx = cc::Build::new();

    let mut ggml_type = String::new();
    println!("in main");

    cxx.include("./llama.cpp/common")
        .include("./llama.cpp")
        //TODO: does this do anything useful?
        .include("./include_shims");

    compile_raw_bindgen(&out_src_path);

    if cfg!(feature = "opencl") {
        compile_opencl(&mut cx, &mut cxx);
        ggml_type = "opencl".to_string();
    } else if cfg!(feature = "openblas") {
        compile_openblas(&mut cx);
    } else if cfg!(feature = "blis") {
        compile_blis(&mut cx);
    } else if cfg!(feature = "metal") && cfg!(target_os = "macos") {
        compile_metal(&mut cx, &mut cxx);
        ggml_type = "metal".to_string();
    }

    if cfg!(feature = "cuda") {
        cx_flags.push_str(" -DGGML_USE_CUBLAS");
        cxx_flags.push_str(" -DGGML_USE_CUBLAS");

        cx.include("/usr/local/cuda/include")
            .include("/opt/cuda/include");
        cxx.include("/usr/local/cuda/include")
            .include("/opt/cuda/include");

        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            cx.include(format!("{}/targets/x86_64-linux/include", cuda_path));
            cxx.include(format!("{}/targets/x86_64-linux/include", cuda_path));
        }

        compile_ggml(&mut cx, &cx_flags);

        compile_cuda(&cxx_flags);

        compile_llama(&mut cxx, &cxx_flags, &out_src_path, "cuda");

        // compile_hip(&cxx_flags);
    } else {
        compile_ggml(&mut cx, &cx_flags);

        compile_llama(&mut cxx, &cxx_flags, &out_src_path, &ggml_type);

        // compile_hip(&cxx_flags);
    }

    //generate the rust code bindings for idiomatic abstraction
    codegen_lib(&out_src_path);
}

//TODO: extract lib essentials to quote here such is the include! statement
//TODO: this could go into binding.rs at top or bottom? would make lib non codegen abstractions (I like this)
///Generate the idiomatic Rust code from the raw llama bindings using syn and quote like a proper Rustacean.
pub fn codegen_lib(out_src_path: &PathBuf) {
    //TODO: this should be json, markdown or toml so we can render into docs later or something
    let mut logfile = File::create("output_log").unwrap();

    let mut bindings = File::open(out_src_path.join("binding.rs")).unwrap();
    let mut bindings_str = String::new();
    bindings.read_to_string(&mut bindings_str).unwrap();
    let syntax = syn::parse_file(&bindings_str).unwrap();

    let mut type_count: HashMap<String, u32> = HashMap::new();

    //TODO: make this more concise, how can we match with a lambda function?
    //TODO: syn crate needs matching methods (is_#token), maybe with a derive macro and a trait for all syntax tokens
    syntax.items.into_iter().for_each(|syntax_item| {
        if let syn::Item::ForeignMod(extern_entry) = syntax_item {
            let mut func_str = String::new();
            func_str.push_str(&extern_entry.to_token_stream().to_string());
            //TODO: how can this be done more gooder?
            if func_str.contains("extern \"C\" { pub fn ") {
                extern_entry.items.into_iter().for_each(|prop| {
                    if let syn::ForeignItem::Fn(prop) = prop {
                        let primary_oriented_arg = prop.sig.inputs.first();
                        match primary_oriented_arg {
                            None => {}
                            Some(primary_oriented_arg) => {
                                if let syn::FnArg::Typed(primary_oriented_arg) =
                                    primary_oriented_arg
                                {
                                    let arg_stream =
                                        primary_oriented_arg.clone().ty.into_token_stream();
                                    let arg_string = arg_stream.to_string();
                                    if type_count.contains_key(&arg_string) {
                                        let count = type_count.get_mut(&arg_string).unwrap();
                                        *count += 1;
                                    } else {
                                        type_count.insert(arg_string, 1);
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
    });

    let mut type_count = type_count
        .into_iter()
        .map(|(k, v)| (k, v))
        .collect::<Vec<(String, u32)>>();
    type_count.sort_by_key(|(k, v)| v.clone());
    let class_targets = type_count
        .into_iter()
        .filter(|x| x.1 > 1)
        .filter(|x| !x.0.is_empty())
        .rev()
        .collect::<Vec<(String, u32)>>();

    //TODO: extract to a parser and codegen routine/func respectively
    //TODO: use quote to create classes for each class_target type and turn the corresponding functions into methods
    //TODO: we have const and mut of same types. these should still be one class but the methods require going from const to mut and back

    //print out the HashMap to out_file
    logfile
        .write_all(format!("Class Targets: {:#?}\n End of Class Targets", class_targets).as_bytes())
        .unwrap();
}

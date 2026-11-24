// Reference oracle for am2strata-hybrid's tests: llama.cpp (a copy of PolyStrata's bundled one, built CPU-only, never
// shipped; patched for per-layer expert counts, see tools/reference-llamacpp/README.md)
// reads prompt ids, then prints, per step, the greedy token and the top-5 logits as JSON lines.
//   oracle <model.gguf> <n_new> <id,id,...>
//   oracle <model.gguf> <n_new> text:<prompt>     (tokenized by llama.cpp, special tokens parsed; ids on stderr)
//   oracle <model.gguf> -1 text:<prompt>          (only the vocabulary is loaded: prints the ids and exits)
// ORACLE_DUMP=<dir>: every named F32 tensor of the prompt pass (llama.cpp's cb names, e.g. Qcur-3, attn_out-3, ffn_inp-3)
// is written as <dir>/<name>.f32 (raw little-endian floats), with <dir>/index.tsv: name, ne0, ne1, ne2, ne3. A name
// computed twice (a looped layer reuses no names, but split graphs may) keeps its first value.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <set>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct dump_state { std::string dir; bool on = true; std::set<std::string> seen; FILE * index = nullptr; };

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user) {
    auto * d = (dump_state *) user;
    if (!d->on || t->type != GGML_TYPE_F32 || t->name[0] == 0) return ask ? false : true;
    if (ask) return true;
    const std::string name = t->name;
    if (d->seen.count(name) || name.find(' ') != std::string::npos || name.find('/') != std::string::npos) return true;
    d->seen.insert(name);
    std::vector<float> buf(ggml_nelements(t));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    } else {
        return true;
    }
    FILE * f = fopen((d->dir + "/" + name + ".f32").c_str(), "wb");
    if (f) { fwrite(buf.data(), sizeof(float), buf.size(), f); fclose(f); }
    fprintf(d->index, "%s\t%lld\t%lld\t%lld\t%lld\n", name.c_str(), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: oracle model.gguf n_new ids\n"); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.use_extra_bufts = false;   // no CPU repack: a copy of the whole model, which Orca's 85 GB would not fit in RAM
    mp.n_gpu_layers = 0;
    const bool vocab_only = std::atoi(argv[2]) < 0;
    mp.vocab_only = vocab_only;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    std::vector<llama_token> ids;
    std::string s = argv[3];
    if (s.rfind("text:", 0) == 0) {
        const std::string text = s.substr(5);
        ids.resize(text.size() + 16);
        int n = llama_tokenize(llama_model_get_vocab(model), text.c_str(), (int) text.size(), ids.data(), (int) ids.size(), true, true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        ids.resize(n);
        fprintf(stderr, "IDS ");
        for (int i = 0; i < n; ++i) fprintf(stderr, "%s%d", i ? "," : "", ids[i]);
        fprintf(stderr, "\n");
    } else {
        size_t p = 0;
        while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); ids.push_back(std::atoi(s.substr(p, q - p).c_str())); p = q + 1; }
    }
    if (vocab_only) { llama_model_free(model); return 0; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192; cp.n_batch = 8192; cp.n_ubatch = 512;
    dump_state dump;
    if (const char * dd = std::getenv("ORACLE_DUMP")) {
        dump.dir = dd;
        dump.index = fopen((dump.dir + "/index.tsv").c_str(), "w");
        if (!dump.index) { fprintf(stderr, "ORACLE_DUMP: cannot write %s/index.tsv\n", dd); return 1; }
        cp.cb_eval = dump_cb; cp.cb_eval_user_data = &dump;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int n_new = std::atoi(argv[2]);
    llama_batch b = llama_batch_get_one(ids.data(), (int) ids.size());
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
    dump.on = false;
    if (dump.index) fclose(dump.index);
    for (int step = 0; step < n_new; ++step) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        std::vector<int> idx(n_vocab);
        for (int i = 0; i < n_vocab; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int c) { return lg[a] > lg[c]; });
        printf("{\"step\":%d,\"top\":[", step);
        for (int k = 0; k < 5; ++k) printf("%s[%d,%.4f]", k ? "," : "", idx[k], lg[idx[k]]);
        printf("]}\n");
        fflush(stdout);
        llama_token t = idx[0];
        llama_batch nb = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, nb)) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    llama_free(ctx); llama_model_free(model);
    return 0;
}

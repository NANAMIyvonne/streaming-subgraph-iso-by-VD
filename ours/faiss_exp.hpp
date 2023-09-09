#ifndef FAISS_EXPERIMENT_H
#define FAISS_EXPERIMENT_H

#include <string>
#include <iostream>
#include <ostream>
#include <chrono>
#include <fstream>

#include <faiss/IndexFlat.h>

using namespace std;

using idx_t = faiss::idx_t;

template <typename T>
class faiss_exp_base {
public:
    
    faiss_exp_base (T& faiss_class_input, string train_input, string faiss_class_name_input, int cache_size_input, float* cache_input, ofstream& off_file_input)
    : faiss_class(faiss_class_input), train(train_input), faiss_class_name(faiss_class_name_input), cache_size(cache_size_input), cache(cache_input), off_file(&off_file_input) {
        init_and_train();
    }

    void init_and_train() {
        if (train == "yes") {
            // *off_file << faiss_class_name << " is_trained = " << faiss_class.is_trained << (faiss_class.is_trained ? "true" : "false") << endl;
            std::chrono::steady_clock::time_point trainStart = std::chrono::steady_clock::now();
            faiss_class.train(cache_size, cache);
            // *off_file << faiss_class_name << " is_trained = " << faiss_class.is_trained << (faiss_class.is_trained ? "true" : "false") << endl;
            float train_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - trainStart).count()/1000.0;
            *off_file << faiss_class_name <<  " train time: " << train_time << " ms, cache_size: " << cache_size << endl;
        }
        faiss_class.add(cache_size, cache); // add vectors to the index
        *off_file << faiss_class_name <<  " ntotal = " << faiss_class.ntotal << endl;
    };

    void search(int nq, float* xq, int k, float* D, idx_t* I) {
        std::chrono::steady_clock::time_point trainStart = std::chrono::steady_clock::now();
        faiss_class.search(nq, xq, k, D, I);
        float train_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - trainStart).count()/1000.0;
        *off_file << faiss_class_name <<  " search time: " << train_time << " ms, cache_size: " << cache_size;
        *off_file << ", search nq: " << nq << endl;
    }

    void set_output_file(ofstream& off_file_input) {
        off_file = &off_file_input;
    }

    int cache_size;
    float* cache;
    T& faiss_class;
    const string train;
    const string faiss_class_name;
    ofstream* off_file;
    
};

#endif
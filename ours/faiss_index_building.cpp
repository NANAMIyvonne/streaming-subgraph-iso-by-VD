#include <string> 
#include <iostream>
#include <argp.h>
#include <algorithm>
#include <chrono>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <limits>
#include <queue>
#include <stack>
#include "TurboIso.hpp"
#include "Graph.hpp"
#include "SubGraphSearch.hpp"
#include "CommonSubGraph.hpp"
#include "Utils.hpp"
#include "Cache.hpp"
#include <filesystem>
#include <fstream>

#include "faiss_exp.hpp"
#include "ThreadPool.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>


ofstream output_file;

using idx_t = faiss::idx_t;

static void fail(string msg) {
    cerr << msg << endl;
    exit(1);
}

void validate_all(int queryStart, int queryEnd, float* D, idx_t* I, vector<Graph>& coldGraph, vector<Graph>& queryGraph, SubGraphSearch subGraphSlover);
void get_embs(vector<Graph>& GraphVec, int start, int number, int dim, int* n, float* querys);
string extractDatasetName(const string& s);
int validate_result(vector<Graph>& coldGraph, Graph queryGraph, SubGraphSearch subGraphSlover, std::priority_queue<std::pair<float, int>>* distanceQueue);

/*******************************************************************************
                             Command-line arguments
*******************************************************************************/

static char doc[] = "Find a maximum clique in a graph in DIMACS format\vHEURISTIC can be min_max or min_product";
static char args_doc[] = "HEURISTIC FILENAME1 FILENAME2 FILENAME3";
static struct argp_option options[] = {
    {"lfu", 'f', 0, 0, "Use lfu cache"},
    {"lru", 'r', 0, 0, "Use lru cache"},
    {"size", 's', "size", 0, "Specify a max cahce size"},
    {"cache", 'c', 0, 0, "Use cache"},
    {"use case 2", 'h', 0, 0, "Use case 2"},
    {"brutal-force search", 'b', 0, 0, "Use brutal-force search"},
    {"vf2", 'v', 0, 0, "Use vf2"},
    {"use distance utility", 'd', 0, 0, "Use distance utility"},
    {"timeout", 't', "timeout", 0, "Specify a timeout (seconds)"},
    {"nresults", 'e', "nresults", 0, "Specify max number of results"},
    {"ncalls", 'a', "ncalls", 0, "Specify max number of calls"},
    {"omega", 'w', "omega", 0, "Specify omega value"},
    {"k", 'k', "k", 0, "Specify k value"},
    {"nlist", 'n', "nlist", 0, "Specify nlist value"},

    { 0 }
};

static struct {
    bool directed;
    bool edge_labelled;
    bool vertex_labelled;
    char *datagraph;
    char *querydir;
    char *coldstartdir;
    float timeout;
    int arg_num;
    int test_case;
    int maxResult;
    int maxRCall;
    bool useCache;
    bool lru;
    bool lfu;
    bool vf2;
    int cache_size;
    bool case2;
    bool distance;
    int omega;
    int k;
    bool brutal;
    int nlist;
} arguments;

void set_default_arguments() {
    arguments.directed = false;
    arguments.edge_labelled = false;
    arguments.vertex_labelled = true;
    arguments.datagraph = NULL;
    arguments.querydir = NULL;
    arguments.coldstartdir = NULL;
    arguments.timeout = 0.0;
    arguments.arg_num = 0;
    arguments.test_case = 2;
    arguments.maxResult = 100;
    arguments.maxRCall = 5000;
    arguments.useCache = false;
    arguments.lru = false;
    arguments.lfu = false;
    arguments.vf2 = false;
    arguments.case2 = false;
    arguments.distance = false;
    arguments.cache_size = 100;
    arguments.omega = 2;
    arguments.k = 5;
    arguments.brutal = false;
    arguments.nlist = 20;
}

static error_t parse_opt (int key, char *arg, struct argp_state *state) {
    switch (key) {
        case 'f':
            arguments.lfu = true;
            break;
        case 'r':
            arguments.lru = true;
            break;
        case 'v':
            arguments.vf2 = true;
            break;
        case 's':
            arguments.cache_size = std::stoi(arg);
            break;
        case 'c':
            arguments.useCache = true;
            break;
        case 'h':
            arguments.case2 = true;
            break;
        case 'd':
            arguments.distance = true;
            break;
        case 'b':
            arguments.brutal = true;
            break;
        case 't':
            cout << arg << endl;
            arguments.timeout = std::stof(arg);
            break;
        case 'e':
            arguments.maxResult = std::stoi(arg);
            break;        
        case 'a':
            arguments.maxRCall = std::stoi(arg);
            break;        
        case 'w':
            arguments.omega = std::stoi(arg);
            break;
        case 'k':
            arguments.k = std::stoi(arg);
            break;
        case 'n':
            arguments.nlist = std::stoi(arg);
            break;
        case ARGP_KEY_ARG:
            if (arguments.arg_num == 0) {
                arguments.datagraph = arg;
            } else if (arguments.arg_num == 1) {
                arguments.querydir = arg;           
            } else if (arguments.arg_num == 2) {
                arguments.coldstartdir = arg;  
            } else {
                argp_usage(state);
            }
            arguments.arg_num++;
            break;
        case ARGP_KEY_END:
            if (arguments.arg_num == 0)
                argp_usage(state);
            break;
        default: return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };

bool endsWith(const std::string &mainStr, const std::string &toMatch) {
    if(mainStr.size() >= toMatch.size() &&
        mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
        return true;
    else
        return false;
}

int main(int argc, char** argv) {
    float totalTime = 0;
    int numberofFound = 0;
    int numberofFound2 = 0;
    int hit = 0;
    float searchTime = 0;
    set_default_arguments();
    argp_parse(&argp, argc, argv, 0, 0, 0);
    Graph dataGraph = readGraph(arguments.datagraph, arguments.directed,
    arguments.edge_labelled, arguments.vertex_labelled);
    dataGraph.buildData();
    std::vector<Graph> queryGraphVector;
    
    DIR* dirp = opendir(arguments.querydir);
    struct dirent * dp;
    std::string querydir(arguments.querydir);
    std::vector<std::string> filenames;
    // while ((dp = readdir(dirp)) != NULL) {
    //     if (dp->d_name[0] == 'q' && endsWith(dp->d_name, "dimas")) {
    //         std::string filename(dp->d_name);
    //         filenames.push_back(querydir+filename);
    //     }
    // }
    cout << "querydir: " << arguments.querydir << endl;
    for (const auto & entry : std::filesystem::directory_iterator(arguments.querydir)) {
        filenames.push_back(entry.path());
    }

    closedir(dirp);
    sort(filenames.begin(), filenames.end());
    double buildTime = 0;
    for (std::string filename : filenames) {
        char c_filename[filename.size() + 1];
	    strcpy(c_filename, filename.c_str());
        Graph queryGraph = readGraph(c_filename, arguments.directed,
        arguments.edge_labelled, arguments.vertex_labelled);
        std::chrono::steady_clock::time_point beginBuild = std::chrono::steady_clock::now();
        queryGraph.buildData();
        buildTime += (std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - beginBuild).count() + 8300);
        queryGraphVector.push_back(queryGraph);     
    }
    std::cout << "Got " << queryGraphVector.size() << " query graph(s)" << std::endl;

    std::vector<Graph> coldstartGraphVector;
    // change the cold start in here by replace the cache
    // use Faiss to store the embedding, redis to store the embedding-subgraph pair
    if (arguments.coldstartdir) {
        DIR* cdirp = opendir(arguments.coldstartdir);
        struct dirent * cdp;
        std::string cquerydir(arguments.coldstartdir);
        std::vector<std::string> cfilenames;
        while ((cdp = readdir(cdirp)) != NULL) {
            if (cdp->d_name[0] == 'q') {
                std::string filename(cdp->d_name);
                cfilenames.push_back(cquerydir+filename);
            }
        }
        closedir(cdirp);
        for (std::string filename : cfilenames) {
            char c_filename[filename.size() + 1];
            strcpy(c_filename, filename.c_str());
            Graph queryGraph = readGraph(c_filename, arguments.directed,
            arguments.edge_labelled, arguments.vertex_labelled);
            queryGraph.buildData();
            coldstartGraphVector.push_back(queryGraph);     
        }
        std::cout << "Got " << coldstartGraphVector.size() << " coldstart query graph(s)" << std::endl;        
    }

    if (coldstartGraphVector.size() > arguments.cache_size) {
        coldstartGraphVector.resize(arguments.cache_size);
    }
    if (coldstartGraphVector.size() > 0)
        queryGraphVector.resize(coldstartGraphVector.size());

    int dim = 128;

    SubGraphSearch subGraphSlover = SubGraphSearch();
    subGraphSlover.timeOut = arguments.timeout;
    int nq = queryGraphVector.size();
    int nb = coldstartGraphVector.size();
    int qNum, cacheNum;
    faiss::IndexFlatL2 quantizer(dim);

    float* cache = new float[dim * nb];
    float* qVector = new float[dim * nq];

    float step = 0.25;
    float portition = step;
    float portition_1 = step;
    int cacheStart, queryStart;

    idx_t* I;
    float* D;

    while (portition <= 1) {
        cacheNum = portition * nb;
        
        portition_1 = step;
        // delete[] cache;
        int cacheNum = portition * nb;
        cache = new float[dim * cacheNum];
        cacheStart = 0;
        get_embs(coldstartGraphVector, cacheStart, cacheNum, dim, &cacheNum, cache);

        while (portition_1 <= 1) {
            qNum = portition_1 * nq;
            if (qNum > cacheNum) {
                break;
            }
            queryStart = 0;
            get_embs(queryGraphVector, queryStart, qNum, dim, &qNum, qVector);
            
            // IndexIVFFlat
            output_file.open("faiss/" + extractDatasetName(arguments.coldstartdir) + "-IndexFlatL2-k_" + to_string(arguments.k) + "-cacheSize_" + to_string(cacheNum) + "-querySize_" + to_string(qNum) + ".csv");
            
            faiss::IndexFlatL2 indexFlatL2(dim);
            faiss_exp_base indexFlatL2Base(indexFlatL2, "no", "IndexFlatL2", cacheNum, cache, output_file);

            // delete[] I, D;
            I = new idx_t[arguments.k * qNum];
            D = new float[arguments.k * qNum];

            indexFlatL2Base.search(qNum, qVector, arguments.k, D, I);
            validate_all(queryStart, queryStart + qNum, D, I, coldstartGraphVector, queryGraphVector, subGraphSlover);
            output_file.close();

            // IndexIVFFlat
            output_file.open("faiss/" + extractDatasetName(arguments.coldstartdir) + "-IndexIVFFlat-k_" + to_string(arguments.k) + "-cacheSize_" + to_string(cacheNum) + "-querySize_" + to_string(qNum) + ".csv");
            
            faiss::IndexIVFFlat index_IVFFlat(&quantizer, dim, arguments.k);
            faiss_exp_base indexIVFFlatBase(index_IVFFlat, "yes", "IndexIVFFlat", cacheNum, cache, output_file);

            // delete[] I, D;
            I = new idx_t[arguments.k * qNum];
            D = new float[arguments.k * qNum];

            indexIVFFlatBase.search(qNum, qVector, arguments.k, D, I);
            validate_all(queryStart, queryStart + qNum, D, I, coldstartGraphVector, queryGraphVector, subGraphSlover);
            output_file.close();

            // IndexIVFPQ
            output_file.open("faiss/" + extractDatasetName(arguments.coldstartdir) + "-IndexIVFPQ-k_" + to_string(arguments.k) + "-cacheSize_" + to_string(cacheNum) + "-querySize_" + to_string(qNum) + ".csv");
            
            int m = 4;
            faiss::IndexIVFPQ indexIVFPQ(&quantizer, dim, arguments.k, m, 8);
            faiss_exp_base IndexIVFPQBase(indexIVFPQ, "yes", "IndexIVFPQ", cacheNum, cache, output_file);

            // delete[] I, D;
            I = new idx_t[arguments.k * qNum];
            D = new float[arguments.k * qNum];

            IndexIVFPQBase.search(qNum, qVector, arguments.k, D, I);
            validate_all(queryStart, queryStart + qNum, D, I, coldstartGraphVector, queryGraphVector, subGraphSlover);
            output_file.close();

            portition_1 += step;
        }

        portition += step;
    }

    return 0;


    TurboIso turboIsoSlover = TurboIso();
    turboIsoSlover.timeOut = arguments.timeout;

    double overHeadTime = 0;
    double totalScanTime = 0;
    double totalComparation = 0;
    Cache graphCache = Cache(arguments.cache_size, arguments.lru, arguments.lfu, arguments.distance, arguments.brutal, arguments.k);
    int graphIndex = 0;

    for (Graph queryGraph : coldstartGraphVector) {
        graphIndex++;
        std::unordered_map<int, float> distanceMap;
        std::priority_queue<std::pair<float, int>> distanceQueue;
        // graphCache.scanCache(queryGraph, &bestIndex, &minDistance, &distanceMap);
        std::chrono::steady_clock::time_point sStart = std::chrono::steady_clock::now();
        graphCache.kScanCache(queryGraph, &distanceQueue, &distanceMap, arguments.k);
        float distanceUtility = 0;
        while(distanceQueue.size()>0) {
            distanceUtility +=  distanceQueue.top().first;
            distanceQueue.pop();
        }
        std::vector<std::map<int,int>> resultsGraph;
        try {
            turboIsoSlover.timeOut = 10;
            turboIsoSlover.getAllSubGraphs(dataGraph, queryGraph, 0, arguments.maxResult);
        }
        catch(std::runtime_error& e) {
            std::cout << e.what() << std::endl;
        }
        resultsGraph = turboIsoSlover.allMappings;
        std::cout << "Found " << resultsGraph.size() << " results" << std::endl;
        if (resultsGraph.size()>0) {
            float t = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart).count()/1000000.0;
            if (t>0.001) 
                t = 0.001;   
            if (arguments.distance)
                t = t*distanceUtility;
            graphCache.insert(queryGraph, resultsGraph, t, graphIndex, distanceMap);
        }
    }

    // in this place, I will change it to batch mode.
    // do knn search in Faiss for all the query graph at first, then process the result one by one
    for (Graph queryGraph : queryGraphVector) {
        graphIndex++;
        std::cout << "Graph Index: " << graphIndex << ":" << queryGraph.familyIndex <<  std::endl;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (arguments.useCache) {
            int cachedGraphIndexCase1 = -1;
            int cachedGraphIndexCase2 = -1;
            int cachedGraphIndexCase3 = -1;
            std::map<int,int> mapNode;
            float utility = 0;
            float s = 0;
            float t = 0;
            std::chrono::steady_clock::time_point sStart = std::chrono::steady_clock::now();
            Graph * bestElem;
            
            float minDistance = std::numeric_limits<float>::max();
            int bestIndex = -1;
            std::unordered_map<int, float> distanceMap;
            std::priority_queue<std::pair<float, int>> distanceQueue;
            if (!arguments.brutal) {
                graphCache.kScanCache(queryGraph, &distanceQueue, &distanceMap, arguments.k);
                std::cout << "No comparasion " << distanceMap.size() << std::endl;                
            } else {
                std::vector<float> queryVector = queryGraph.embeddingVector;
                for (const auto & cachedVector: graphCache.cachedEmbeddingVectors) {
                    float distance = 0;
                    for (int k = 0; k <queryVector.size();k++) {
                        distance += (cachedVector.second[k] - queryVector[k]) * (cachedVector.second[k] - queryVector[k]);
                        // if (distance > bestDistance) break;
                    }
                    distanceMap.insert({cachedVector.first, distance});
                    if (distanceQueue.size() < arguments.k) {
                        distanceQueue.push({distance, cachedVector.first});
                    } else if (distance < distanceQueue.top().first) {
                        distanceQueue.pop();
                        distanceQueue.push({distance, cachedVector.first});
                    }
                }             
            }
            float curr_search_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart).count();
            searchTime += curr_search_time;
            cout << "Search Using Time: " << curr_search_time/1000 << " ms" << endl;
            totalComparation += distanceMap.size();
            float distanceUtility = 0;
            if (distanceQueue.size()>0)  {
                std::stack<int> S;
                while(distanceQueue.size()>0) {
                    int currIndex = distanceQueue.top().second;
                    distanceUtility +=  distanceQueue.top().first;
                    distanceQueue.pop();
                    S.push(currIndex);
                }
                while(S.size()>0) {
                    int currIndex = S.top();
                    S.pop();
                    bestElem = graphCache.getGraphById(currIndex);
                    // std::cout << currIndex << " " << bestElem->familyIndex << " ";
                    std::map<int,int> tempMapNode;
                    bool timedout = false;
                    std::vector<std::map<int,int>> tempResults;

                    if (queryGraph.n <= bestElem->n) {
                        try {
                            subGraphSlover.timeOut = 0.001;
                            // subGraphSlover.getAllSubGraphs(queryGraph, elem.second, 0, 1);
                            subGraphSlover.getAllSubGraphs(*bestElem,queryGraph, 0, 1);
                        }
                        catch(std::runtime_error& e) {
                            std::cout << "Time out" << std::endl;
                            timedout = true;
                        }
                        tempMapNode = subGraphSlover.largestMapping;
                        // std::cout << tempMapNode.size() << " " << queryGraph.n << " " << bestElem->n << std::endl;
                        if (tempMapNode.size() ==bestElem->n && bestElem->n == queryGraph.n) {
                            cachedGraphIndexCase1 = currIndex;
                            mapNode = tempMapNode;
                            break;
                        } else if (tempMapNode.size() == queryGraph.n && bestElem->n > queryGraph.n) {
                            cachedGraphIndexCase3 = currIndex;
                            mapNode = tempMapNode;
                            break;
                        } 
                        else if (arguments.case2 && tempMapNode.size() > mapNode.size() && tempMapNode.size() +arguments.omega > queryGraph.n && tempMapNode.size()< queryGraph.n) {
                            mapNode = tempMapNode;
                            cachedGraphIndexCase2 =currIndex;
                            break;
                        } 
                    } 
                    else if (arguments.case2) {
                        try {
                            subGraphSlover.timeOut = 0.001;
                            subGraphSlover.getAllSubGraphs(queryGraph, *bestElem, 0, 1);
                            // subGraphSlover.getAllSubGraphs(*bestElem,queryGraph, 0, 1);
                        }
                        catch(std::runtime_error& e) {
                            std::cout << "Time out" << std::endl;
                            timedout = true;
                        }

                        tempMapNode = subGraphSlover.largestMapping;
                        // std::cout << tempMapNode.size()<< " " << queryGraph.n << " " << bestElem->n << std::endl;
                        if (tempMapNode.size() > mapNode.size() && tempMapNode.size() +arguments.omega > queryGraph.n && tempMapNode.size()< queryGraph.n) {
                            mapNode.clear();
                            for (auto i : tempMapNode) 
                                mapNode.insert({i.second, i.first});
                            cachedGraphIndexCase2 =currIndex;
                            break;
                        } 
                    }
                }   
                // std::cout << std::endl;
            }
            float curr_scan_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart).count()/1000.0;
            totalScanTime += curr_scan_time;
            cout << "Scan time: " << curr_scan_time/1000 << " ms" << endl;
            overHeadTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart).count();
            bool useCache = false;
            if (cachedGraphIndexCase1 != -1) {
                hit ++;
                std::cout << "Found case 1:" << cachedGraphIndexCase1 << std::endl;
                std::vector<std::map<int,int>> * cachedEmbedding = graphCache.getEmbeddingsById(cachedGraphIndexCase1); 
                if (cachedEmbedding != NULL) {
                    numberofFound ++;
                    numberofFound2 += cachedEmbedding->size();
                    useCache = true;
                }
                s = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart).count()/1000000.0;
                std::chrono::steady_clock::time_point sStart2 = std::chrono::steady_clock::now();
                graphCache.updateCacheHit(cachedGraphIndexCase1, s, graphIndex);
                overHeadTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart2).count();
            } else if (cachedGraphIndexCase3 != -1) {
                hit ++;
                std::cout << "Found case 3:" << cachedGraphIndexCase3 << std::endl;
                std::vector<std::map<int,int>> * cachedEmbedding = graphCache.getEmbeddingsById(cachedGraphIndexCase3); 
                if (cachedEmbedding != NULL) {
                    numberofFound ++;
                    numberofFound2 += cachedEmbedding->size();
                    useCache = true;               
                }
                s = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart).count()/1000000.0;
                std::chrono::steady_clock::time_point sStart2 = std::chrono::steady_clock::now();
                graphCache.updateCacheHit(cachedGraphIndexCase3, s, graphIndex);
                overHeadTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart2).count();
            } else if (cachedGraphIndexCase2 != -1) {
                std::vector<std::map<int,int>> * cachedEmbedding = graphCache.getEmbeddingsById(cachedGraphIndexCase2); 
                if (cachedEmbedding != NULL) {
                    hit ++;
                    std::cout << "Found case 2:" << cachedGraphIndexCase2 << std::endl;
                    useCache = true;
                    std::set<std::map<int,int>> candidateBigU =  ProjectEmbedding::crossProjectEmbedding(* cachedEmbedding, mapNode);
                    std::vector<std::map<int,int>> resultsGraph;
                    if (arguments.vf2) {
                        try {
                            subGraphSlover.timeOut = arguments.timeout;
                            subGraphSlover.getAllSubGraphs(dataGraph, queryGraph, 0, arguments.maxResult);
                        }
                        catch(std::runtime_error& e) {
                            std::cout << e.what() << std::endl;
                        }
                        resultsGraph = subGraphSlover.allMappings;
                    } else {
                        try {
                            turboIsoSlover.timeOut = arguments.timeout;
                            turboIsoSlover.getAllSubGraphs(dataGraph, queryGraph, 0, arguments.maxResult);
                        }
                        catch(std::runtime_error& e) {
                            std::cout << e.what() << std::endl;
                        }
                        resultsGraph = turboIsoSlover.allMappings;
                    }
                    std::cout << "Found " << resultsGraph.size() << " results" << std::endl;
                    std::chrono::steady_clock::time_point sStart2 = std::chrono::steady_clock::now();
                    if (resultsGraph.size()>0) {
                        if (graphCache.isLRU || graphCache.isLFU) {
                            std::cout << "add query graph " << graphIndex << " to cache" << std::endl; 
                            graphCache.insert(queryGraph, resultsGraph, t, graphIndex, distanceMap); 
                        } else {
                            t = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart).count()/1000000.0;
                            float tempT;
                            if (arguments.distance) {
                                // std::cout << "Ours method" << std::endl;
                                tempT = t*distanceUtility;
                            } else {
                                // std::cout << "Recache method" << std::endl;
                                tempT = t;
                            }   
                            // std::cout << tempT << " vs " << graphCache.minUtility << std::endl;
                            if (tempT>=graphCache.minUtility) {
                                std::cout << "add query graph " << graphIndex << " to cache" << std::endl; 
                                graphCache.insert(queryGraph, resultsGraph, t, graphIndex, distanceMap);
                            }   
                        }
                        numberofFound2 +=  resultsGraph.size();
                        numberofFound ++;  
                    }
                    s = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart).count()/1000000.0;
                    graphCache.updateCacheHit(cachedGraphIndexCase2, s, graphIndex);
                    overHeadTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart2).count();
                } 
            } 
            if (!useCache) {
                std::cout << "Find from the beginning" << std::endl;
                bool timedout = false;
                std::vector<std::map<int,int>> resultsGraph;
                if (arguments.vf2) {
                    try {
                        subGraphSlover.timeOut = arguments.timeout;
                        subGraphSlover.getAllSubGraphs(dataGraph, queryGraph, 0, arguments.maxResult);
                    }
                    catch(std::runtime_error& e) {
                        std::cout << e.what() << std::endl;
                    }
                    resultsGraph = subGraphSlover.allMappings;
                } else {
                    try {
                        turboIsoSlover.timeOut = arguments.timeout;
                        turboIsoSlover.getAllSubGraphs(dataGraph, queryGraph, 0, arguments.maxResult);
                    }
                    catch(std::runtime_error& e) {
                        std::cout << e.what() << std::endl;
                    }
                    resultsGraph = turboIsoSlover.allMappings;
                }
                std::cout << "Found " << resultsGraph.size() << " results" << std::endl;
                if (resultsGraph.size()>0) {
                    std::chrono::steady_clock::time_point sStart2 = std::chrono::steady_clock::now();
                    if (graphCache.isLRU || graphCache.isLFU) {
                        std::cout << "add query graph " << graphIndex << " to cache" << std::endl; 
                        graphCache.insert(queryGraph, resultsGraph, t, graphIndex, distanceMap);       
                    } else {
                        t = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart).count()/1000000.0;
                        float tempT;
                        if (arguments.distance) {
                            // std::cout << "Ours method" << std::endl;
                            tempT = t*distanceUtility;
                        } else {
                            // std::cout << "Recache method" << std::endl;
                            tempT = t;
                        }  
                        std::cout << tempT << " vs " << graphCache.minUtility << std::endl;
                        if (tempT>=graphCache.minUtility) {
                            std::cout << "add query graph " << graphIndex << " to cache" << std::endl; 
                            graphCache.insert(queryGraph, resultsGraph, t, graphIndex, distanceMap);
                        }   
                    }
                    numberofFound2 +=  resultsGraph.size();
                    numberofFound ++;  
                    overHeadTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sStart2).count();                      
                }
            }
            // graphCache.printCache();
        } else {
            bool timedout = false;
            std::vector<std::map<int,int>> resultsGraph;
            if (arguments.vf2) {
                try {
                    subGraphSlover.timeOut = arguments.timeout;
                    subGraphSlover.getAllSubGraphs(dataGraph, queryGraph, 0, arguments.maxResult);
                }
                catch(std::runtime_error& e) {
                    std::cout << e.what() << std::endl;
                }
                resultsGraph = subGraphSlover.allMappings;
            } else {
                try {
                    turboIsoSlover.timeOut = arguments.timeout;
                    turboIsoSlover.getAllSubGraphs(dataGraph, queryGraph, 0, arguments.maxResult);
                }
                catch(std::runtime_error& e) {
                    std::cout << e.what() << std::endl;
                }
                resultsGraph = turboIsoSlover.allMappings;
            }
            std::cout << "Found " << resultsGraph.size() << " results" << std::endl;
            if (resultsGraph.size()>0) {
                std::cout << "add query graph " << graphIndex << " to cache" << std::endl; 
                numberofFound2 +=  resultsGraph.size();
                numberofFound ++;
            }
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        float runningTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()/1000000.0;
        totalTime +=  runningTime;
        // std::cout << "Utility" << std::endl;
        // for (auto i : graphCache.utilities) {
        //     std::cout << i.first << ":" << i.second << std::endl;
        // }
        // std::cout << "next evited " << graphCache.nextIndexToEvicted << ":" << graphCache.minUtility << std::endl;
        std::cout << "Running time: " << runningTime << "[ms]" << std::endl;
        std::cout << "======================================" << std::endl;
    }
}

int validate_result(vector<Graph>& coldGraph, Graph queryGraph, SubGraphSearch subGraphSlover, std::priority_queue<std::pair<float, int>>* distanceQueue) {
    int cachedGraphIndexCase1 = -1;
    int cachedGraphIndexCase2 = -1;
    int cachedGraphIndexCase3 = -1;

    std::map<int,int> mapNode;
    std::chrono::steady_clock::time_point sStart = std::chrono::steady_clock::now();
    Graph * bestElem;
    
    float minDistance = std::numeric_limits<float>::max();
    int bestIndex = -1;

    // graphCache.kScanCache(queryGraph, &distanceQueue, &distanceMap, arguments.k);
    // std::cout << "No comparasion " << distanceMap.size() << std::endl;

    if (distanceQueue->size()>0)  {
        std::stack<int> S;
        while(distanceQueue->size()>0) {
            int currIndex = distanceQueue->top().second;
            distanceQueue->pop();
            S.push(currIndex);
        }
        while(S.size()>0) {
            int currIndex = S.top();
            S.pop();
            bestElem = &(coldGraph[currIndex]);
            // std::cout << currIndex << " " << bestElem->familyIndex << " ";
            std::map<int,int> tempMapNode;
            bool timedout = false;
            std::vector<std::map<int,int>> tempResults;

            if (queryGraph.n <= bestElem->n) {
                try {
                    subGraphSlover.timeOut = 0.001;
                    // subGraphSlover.getAllSubGraphs(queryGraph, elem.second, 0, 1);
                    subGraphSlover.getAllSubGraphs(*bestElem,queryGraph, 0, 1);
                }
                catch(std::runtime_error& e) {
                    // std::cout << "Time out" << std::endl;
                    timedout = true;
                }
                tempMapNode = subGraphSlover.largestMapping;
                // std::cout << tempMapNode.size() << " " << queryGraph.n << " " << bestElem->n << std::endl;
                if (tempMapNode.size() ==bestElem->n && bestElem->n == queryGraph.n) {
                    cachedGraphIndexCase1 = currIndex;
                    mapNode = tempMapNode;
                    return 1;
                    break;
                } else if (tempMapNode.size() == queryGraph.n && bestElem->n > queryGraph.n) {
                    cachedGraphIndexCase3 = currIndex;
                    mapNode = tempMapNode;
                    return 3;
                    break;
                } 
                else if (arguments.case2 && tempMapNode.size() > mapNode.size() && tempMapNode.size() +arguments.omega > queryGraph.n && tempMapNode.size()< queryGraph.n) {
                    mapNode = tempMapNode;
                    cachedGraphIndexCase2 =currIndex;
                    return 2;
                    break;
                } 
            } 
            else if (arguments.case2) {
                try {
                    subGraphSlover.timeOut = 0.001;
                    subGraphSlover.getAllSubGraphs(queryGraph, *bestElem, 0, 1);
                    // subGraphSlover.getAllSubGraphs(*bestElem,queryGraph, 0, 1);
                }
                catch(std::runtime_error& e) {
                    // std::cout << "Time out" << std::endl;
                    timedout = true;
                }

                tempMapNode = subGraphSlover.largestMapping;
                // std::cout << tempMapNode.size()<< " " << queryGraph.n << " " << bestElem->n << std::endl;
                if (tempMapNode.size() > mapNode.size() && tempMapNode.size() +arguments.omega > queryGraph.n && tempMapNode.size()< queryGraph.n) {
                    mapNode.clear();
                    for (auto i : tempMapNode) 
                        mapNode.insert({i.second, i.first});
                    cachedGraphIndexCase2 =currIndex;
                    return 2;
                    break;
                } 
            }
        }
    }

    return -1;
}

string extractDatasetName(const string& s) {
    // The last character of the string
    char lastChar = s[s.length() - 1];

    // Find the last '/' character in the string
    size_t lastSlashPos = (lastChar == '/') ? s.find_last_of('/', s.length() - 2) : s.find_last_of('/');

    // Remove everything after the last '/'
    string temp = s.substr(0, lastSlashPos);

    // Find the next last '/' character in the string
    size_t nextLastSlashPos = temp.find_last_of('/');

    // Extract the string between the two '/' characters
    string result = temp.substr(nextLastSlashPos + 1, lastSlashPos - nextLastSlashPos - 1);

    return result;
}

void get_embs(vector<Graph>& GraphVec, int start, int number, int dim, int* n, float* querys) {
    number = min(int(GraphVec.size()) - start, number);

    if (number <= 0) {
        *n = 0;
        return;
    }

    *n = number;
    for (int i = start; i < *n; i++) {
        int startIndex = i * dim;
        vector<float> currEmb = GraphVec[i].embeddingVector;
        for (int j = 0; j < dim; j++) {
            querys[startIndex + j] = currEmb[j];
        }
    }
}

// void validate_all(int queryStart, int queryEnd, float* D, idx_t* I, vector<Graph>& coldGraph, vector<Graph>& queryGraph, SubGraphSearch subGraphSlover) {
//     for (int i = queryStart; i < queryEnd; i++) {
//         priority_queue<pair<float, int>> distanceQueue = priority_queue<pair<float, int>>();
//         for (int j = 0; j < arguments.k; j++) {
//             distanceQueue.push({D[i * arguments.k + j], I[i * arguments.k + j]});
//         }
//         int result = validate_result(coldGraph, queryGraph[i], subGraphSlover, &distanceQueue);
//         output_file << result << endl;
//     }

// }

void validate_all(int queryStart, int queryEnd, float* D, idx_t* I, vector<Graph>& coldGraph, vector<Graph>& queryGraph, SubGraphSearch subGraphSlover) {
    // Number of threads in the pool will depend on your hardware and use case
    const size_t num_threads = 30;
    ThreadPool pool(num_threads);

    // A vector to store the vectors of results generated by each thread
    vector<vector<int>> results(num_threads);

    // Size of queries to be processed by each thread
    size_t chunk_size = (queryEnd - queryStart + num_threads - 1) / num_threads; // rounded up division

    for (size_t t = 0; t < num_threads; t++) {
        pool.enqueue([&, t]() {
            int start = queryStart + t * chunk_size;
            int temp = queryStart + (t+1) * chunk_size;
            int end = std::min(temp, queryEnd);
            results[t].resize(end - start, -1);

            for (int i = start; i < end; i++) {
                priority_queue<pair<float, int>> distanceQueue;
                for (int j = 0; j < arguments.k; j++) {
                    distanceQueue.push({D[i * arguments.k + j], I[i * arguments.k + j]});
                }
                results[t][i - start] = validate_result(coldGraph, queryGraph[i], subGraphSlover, &distanceQueue);
            }
        });
    }

    // Wait for all tasks to finish before continuing
    pool.wait();

    // Now, output all results at once
    for (const auto &result_set : results) {
        for (const auto &result : result_set) {
            output_file << result << endl;
        }
    }
}


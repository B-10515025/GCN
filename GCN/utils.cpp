#include "utils.h"
#include "metis.h" 
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <time.h>
void error(string msg)
{
    cout << msg << " Error.\n";
    exit(-1);
}
void memory(string msg, int byte)
{
    cout << msg << " Cost " << byte << " Bytes Memory.\n";
}
void getRandom(int* index, int max)
{
    int src, dist, temp;
    for (int i = 0; i < max; i++)
        index[i] = i;
    for (int i = 0; i < max; i++)
    {
        src = rand() % max;
        dist = rand() % max;
        temp = index[src];
        index[src] = index[dist];
        index[dist] = temp;
    }
}
int readFeatures(const char* file, double*& X, double*& Y, vector<string>& ID, vector<string>& Name)
{
    ifstream fin(file);
    string line, word;
    stringstream ss;
    vector<Data> data;
    Data d;
    double n;
    while (getline(fin, line))
    {
        ss.clear();
        ss << line;
        ss >> d.id;
        ss >> d.label;
        d.features.clear();
        while (ss >> word)
        {
            n = atof(d.label.c_str());
            d.features.push_back(n);
            d.label = word;
        }
        if (d.id == "" || d.label == "" || d.features.size() == 0 || (data.size() > 0 && d.features.size() != data[0].features.size()))
            error("Read Features");
        data.push_back(d);
    }
    cout << data[0].features.size() << " Features.\n";
    ID.clear();
    Name.clear();
    bool push;
    for (int i = 0; i < data.size(); i++)
    {
        push = true;
        for (int j = 0; j < ID.size(); j++)
            if (data[i].id == ID[j])
                error("Duplicate ID");
        ID.push_back(data[i].id);
        for (int j = 0; j < Name.size(); j++)
            if (data[i].label == Name[j])
                push = false;
        if (push)
            Name.push_back(data[i].label);
    }
    X = new double[data.size() * data[0].features.size()];
    Y = new double[data.size() * Name.size()];
    memory("Read Features", (data.size() * data[0].features.size() + data.size() * Name.size()) * sizeof(double));
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < data[i].features.size(); j++)
            X[i * data[0].features.size() + j] = data[i].features[j];
        for (int j = 0; j < Name.size(); j++)
            if (data[i].label == Name[j])
                Y[i * Name.size() + j] = 1;
            else
                Y[i * Name.size() + j] = 0;
    }
    return data[0].features.size();
}
void readGraph(const char* file, vector<string>& ID, vector<vector<edge>>& graph)
{
    int srcIndex, distIndex, count = 0;
    ifstream fin(file);
    string line, src, dist;
    vector<edge> edges;
    edge e;
    graph.clear();
    for (int i = 0; i < ID.size(); i++)
    {
        edges.clear();
        e.N = i;
        edges.push_back(e);
        graph.push_back(edges);
    }
    while (getline(fin, line))
    {
        stringstream ss;
        ss.clear();
        ss << line;
        ss >> src >> dist;
        srcIndex = -1;
        distIndex = -1;
        for (int i = 0; i < ID.size(); i++)
        {
            if (ID[i] == src)
                srcIndex = i;
            if (ID[i] == dist)
                distIndex = i;
        }
        if (srcIndex < 0 || distIndex < 0)
            error("Read Graph");
        bool push;
        e.N = distIndex;
        push = true;
        for (int i = 0; i < graph[srcIndex].size(); i++)
            if (graph[srcIndex][i].N == e.N)
                push = false;
        if (push)
            graph[srcIndex].push_back(e);
        e.N = srcIndex;
        push = true;
        for (int i = 0; i < graph[distIndex].size(); i++)
            if (graph[distIndex][i].N == e.N)
                push = false;
        if (push)
            graph[distIndex].push_back(e);
    }
    for (int i = 0; i < graph.size(); i++)
        for (int j = 0; j < graph[i].size(); j++)
        {
            graph[i][j].V = 1 / sqrt(graph[i].size() * graph[graph[i][j].N].size());
            count++;
        }
    memory("Read Graph", graph.size() * sizeof(vector<edge>) + count * (sizeof(int) + sizeof(double)));
}
void nodeSplit(int size, vector<int>& trainNode, int trainWeight, vector<int>& validationNode, int validationWeight, vector<int>& testNode, int testWeight)
{
    trainNode.clear();
    validationNode.clear();
    testNode.clear();
    int n, train = 0, validation = 0, test = 0;
    for (int i = 0; i < size; i++)
    {
        n = rand() % (trainWeight + validationWeight + testWeight);
        if (n < trainWeight)
        {
            train++;
            trainNode.push_back(1);
            validationNode.push_back(0);
            testNode.push_back(0);
        }
        else if (n < trainWeight + validationWeight)
        {
            validation++;
            trainNode.push_back(0);
            validationNode.push_back(1);
            testNode.push_back(0);
        }
        else
        {
            test++;
            trainNode.push_back(0);
            validationNode.push_back(0);
            testNode.push_back(1);
        }
    }
    cout << "Split: " << train << " Train Node, " << validation << " Validation Node, " << test << " Test Node\n";
}
void nodeCluster(vector<vector<int>>& cluster, vector<vector<edge>>& graph, int number, string method,string graphPath)
{
    if (method == "random")
    {
        cluster.clear();
        for (int i = 0; i < number; i++)
            cluster.push_back(vector<int>(graph.size(), 0));
        for (int i = 0; i < graph.size(); i++)
            cluster[rand() % number][i] = 1;
    }
    else
    {
        ifstream ingraph(graphPath);
        if (!ingraph) {
            cout << "failed to open graph file" << endl;
            exit(1);
        }
        int vexnum, edgenum;
        string line;
        getline(ingraph, line);
        istringstream tmp(line);
        tmp >> vexnum >> edgenum;
        vector<idx_t> xadj(0);
        vector<idx_t> adjncy(0);
        vector<idx_t> vwgt(0);

        idx_t a, w;
        for (int i = 0; i < vexnum; i++) {
            xadj.push_back(adjncy.size());
            getline(ingraph, line);
            istringstream tmp(line);
            while (tmp >> a >> w) {
                adjncy.push_back(a);
                vwgt.push_back(w);
            }
        }
        xadj.push_back(adjncy.size());

        ingraph.close();


        idx_t nVertices = xadj.size() - 1;
        idx_t nEdges = adjncy.size() / 2;
        idx_t nWeights = 1;
        idx_t nParts = number;
        idx_t objval;
        std::vector<idx_t> part(nVertices, 0);

        if (nParts < 2) { error("ClUSTER_ERROR"); }

        for (int i = 0; i < nParts; i++) {
            cluster.push_back(vector<int>(graph.size(), 0));
        }

        int ret = METIS_PartGraphKway(&nVertices, &nWeights, xadj.data(), adjncy.data(),
            vwgt.data(), NULL, NULL, &nParts, NULL,
            NULL, NULL, &objval, part.data());


        if (ret != rstatus_et::METIS_OK) { error("METIS_ERROR"); }
        cout << "METIS_OK" << endl;

        for (unsigned node_id = 0; node_id < part.size(); node_id++) {
            cluster[part[node_id]][node_id] = 1;
        }

    }
}
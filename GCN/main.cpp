#include <fstream>
#include <string>
#include <sstream>
#include "GCN.h"
#define NODE 2708
#define FEATURES 1433
#define CLASS 7
using namespace std;
int readFeatures(const char* file, double* &X, double* &Y, vector<string>& ID, vector<string>& Name)
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
    srand(time(NULL));
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
void nodeCluster(vector<vector<int>>& cluster, int number, int node, string method)
{
    if (method == "random")
    {
        cluster.clear();
        for (int i = 0; i < number; i++)
            cluster.push_back(vector<int>(node, 0));
        srand(time(NULL));
        for (int i = 0; i < node; i++)
            cluster[rand() % number][i] = 1;
    }
}
int main()
{
    vector<int> shape;
    vector<string> ID, Name;
    double* X, * Y;
    vector<vector<edge>> graph;
    shape.push_back(readFeatures("data/cora.content", X, Y, ID, Name));
    readGraph("data/cora.cites", ID, graph);
    shape.push_back(16);
    shape.push_back(16);
    shape.push_back(Name.size());
    vector<int> trainNode, validationNode, testNode, predictLabel;
    nodeSplit(graph.size(), trainNode, 1, validationNode, 2, testNode, 7);
    vector<vector<int>> cluster;
    nodeCluster(cluster, 10, ID.size(), "random");
    GCN model(shape, time(NULL));
    model.earlystop = 50;
    model.decay = pow(10, -0.2);
    model.max_epoch = 1000;
    model.train(X, Y, graph, cluster, 2, trainNode, validationNode);
    model.predict(X, predictLabel, graph, testNode, true);
    int acc = 0, count = 0, index;
    for (int i = 0; i < predictLabel.size(); i++)
        if (predictLabel[i] >= 0)
        {
            count++;
            index = 0;
            for (int j = 0; j < Name.size(); j++)
                if (Y[i * Name.size() + j] > 0)
                    index = j;
            cout << i << " " << Name[predictLabel[i]] << " " << Name[index];
            if (predictLabel[i] == index)
            {
                acc++;
                cout << " O\n";
            }
            else
                cout << " X\n";
        }
    cout << acc * 100.0 / count << "% Predicted.\n";
    return 0;
}
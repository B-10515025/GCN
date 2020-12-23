#include "GCN.h"
#include "utils.h"
#include <iostream>
#include <time.h>
int main()
{
    vector<int> shape;
    vector<string> ID, Name;
    double* X, * Y;
    vector<vector<edge>> graph;
    shape.push_back(readFeatures("data/cora.content", X, Y, ID, Name));
    readGraph("data/cora.edges", ID, graph);
    shape.push_back(16);
    shape.push_back(16);
    shape.push_back(Name.size());
    srand(time(NULL));
    vector<int> trainNode, validationNode, testNode, predictLabel;
    nodeSplit(graph.size(), trainNode, 1, validationNode, 2, testNode, 7);
    vector<vector<int>> cluster;
    nodeCluster(cluster, graph, 10, "", "data/cora.graph");
    GCN model(shape, true);
    model.earlystop = 500;
    model.init_learning_rate = 1e-3;
    model.decay = pow(10, -0.1);
    model.show_per_epoch = 100;
    model.max_epoch = 1000;
    model.train(X, Y, graph, cluster, 2, trainNode, validationNode);
    model.predict(X, predictLabel, graph, testNode);
    int acc = 0, count = 0, index;
    for (int i = 0; i < predictLabel.size(); i++)
        if (predictLabel[i] >= 0)
        {
            index = 0;
            for (int j = 0; j < Name.size(); j++)
                if (Y[i * Name.size() + j] > 0)
                    index = j;
            count++;
            if (predictLabel[i] == index)
                acc++;
        }
    cout << acc * 100.0 / count << "% Predicted.\n";
    return 0;
}
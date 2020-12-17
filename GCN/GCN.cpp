#include "GCN.h"
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
    srand(time(NULL));
    for (int i = 0; i < max; i++)
    {
        src = rand() % max;
        dist = rand() % max;
        temp = index[src];
        index[src] = index[dist];
        index[dist] = temp;
    }
}
GCN::GCN(vector<int>& _shape, int seed)
{
	if (_shape.size() < 2)
        error("Shape");
	shape.assign(_shape.begin(), _shape.end());
	weight = new double* [shape.size() - 1];
	delta = new double* [shape.size() - 1];
	node = new double* [shape.size() - 1];
	embedding = new double* [shape.size() - 1];
    if (!(weight && delta && node && embedding))
        error("Memory");
    int byte = 0;
	for (int i = 0; i < shape.size() - 1; i++)
	{
		weight[i] = new double[(shape[i] + 1) * shape[i + 1]];
		delta[i] = new double[(shape[i] + 1) * shape[i + 1]];
        if (!(weight[i] && delta[i]))
            error("Memory");
        else
            byte += (shape[i] + 1) * shape[i + 1] * 2 * sizeof(double);
	}
    memory("Build Model", byte);
	srand(seed);
    for (int i = 0; i < shape.size() - 1; i++)
        for (int j = 0; j < (shape[i] + 1) * shape[i + 1]; j++)
            weight[i][j] = (rand() - rand()) / 65536.0;
    max_epoch = 200;
    show_per_epoch = 1;
    earlystop = 10;
    init_learning_rate = 1e-3;
    min_learning_rate = 1e-8;
    decay = 1;
}
GCN::~GCN()
{
    for (int i = 0; i < shape.size() - 1; i++)
    {
        delete[] weight[i];
        delete[] delta[i];
    }
    delete[] weight;
    delete[] delta;
    delete[] node;
    delete[] embedding;
}
void GCN::train(double* X, double* Y, vector<vector<edge>>& graph, vector<vector<int>>& cluster, int batchSize, vector<int>& trainNode, vector<int>& validationNode)
{
    vector<vector<edge>> batch_graph;
    vector<int> batch_trainNode, batch_validationNode;
    double BEGIN = clock(), learning_rate = init_learning_rate, preloss = 1e3, max_acc = 0, acc_count = 0, acc, min_loss = 1e10, loss_count = 0, loss, val_acc, val_loss, sum, max, loss_threshold = 0;
    int index, batchCount, trainCount = 0, validationCount = 0, byte, maxByte;
    int* groupIndex, * batchNodeIndex, * Mapper;
    double* batchX, * batchY;
    if (cluster.size() % batchSize != 0)
        error("Group Batch Size");
    for (int i = 0; i < cluster.size(); i++)
        if (graph.size() != cluster[i].size())
            error("Group Node Size");
    if (graph.size() != trainNode.size() || graph.size() != validationNode.size())
        error("Node Size");
    groupIndex = new int[cluster.size()];
    batchNodeIndex = new int[graph.size()];
    Mapper = new int[graph.size()];
    for (int i = 0; i < shape.size() - 1; i++)
    {
        node[i] = new double[1];
        embedding[i] = new double[1];;
    }
    for (int i = 0; i < graph.size(); i++)
    {
        trainCount += trainNode[i];
        validationCount += validationNode[i];
    }
    for (int epoch = 0; epoch < max_epoch; epoch++)
    {
        maxByte = 0;
        getRandom(groupIndex, cluster.size());
        acc = 0;
        loss = 0;
        val_acc = 0;
        val_loss = 0;
        for (int batch = 0; batch < cluster.size() / batchSize; batch++)
        {
            for (int i = 0; i < graph.size(); i++)
                batchNodeIndex[i] = 0;
            for (int i = 0; i < batchSize; i++)
                for (int j = 0; j < graph.size(); j++)
                    batchNodeIndex[j] += cluster[batch * batchSize + i][j];
            batchCount = 0;
            for (int i = 0; i < graph.size(); i++)
                if (batchNodeIndex[i] > 0)
                    batchCount++;
            batchX = new double[batchCount * shape[0]];
            batchY = new double[batchCount * shape[shape.size() - 1]];
            batch_graph.clear();
            batch_trainNode.clear();
            batch_validationNode.clear();
            index = 0;
            for (int i = 0; i < graph.size(); i++)
                if (batchNodeIndex[i] > 0)
                {
                    memcpy(batchX + index * shape[0], X + i * shape[0], shape[0] * sizeof(double));
                    memcpy(batchY + index * shape[shape.size() - 1], Y + i * shape[shape.size() - 1], shape[shape.size() - 1] * sizeof(double));
                    vector<edge> edges;
                    edges.clear();
                    for (int j = 0; j < graph[i].size(); j++)
                        if (batchNodeIndex[graph[i][j].N] > 0)
                        {
                            edge e;
                            e.N = graph[i][j].N;
                            e.V = graph[i][j].V;
                            edges.push_back(e);
                        }
                    batch_graph.push_back(edges);
                    batch_trainNode.push_back(trainNode[i]);
                    batch_validationNode.push_back(validationNode[i]);
                    Mapper[i] = index++;
                }
            for (int i = 0; i < batch_graph.size(); i++)
                for (int j = 0; j < batch_graph[i].size(); j++)
                    batch_graph[i][j].N = Mapper[batch_graph[i][j].N];
            byte = fit(batchX, batchY, batch_graph, batch_trainNode, learning_rate);
            if (byte > maxByte)
                maxByte = byte;
            for (int i = 0; i < batch_graph.size(); i++)
            {
                index = 0;
                max = 0;
                for (int output = 0; output < shape[shape.size() - 1]; output++)
                {
                    if (node[shape.size() - 2][i * shape[shape.size() - 1] + output] > max)
                    {
                        max = node[shape.size() - 2][i * shape[shape.size() - 1] + output];
                        index = output;
                    }
                    if (node[shape.size() - 2][i * shape[shape.size() - 1] + output] < 1e-15)
                    {
                        loss -= batchY[i * shape[shape.size() - 1] + output] * log(1e-15) * batch_trainNode[i];
                        val_loss -= batchY[i * shape[shape.size() - 1] + output] * log(1e-15) * batch_validationNode[i];
                    }
                    else
                    {
                        loss -= batchY[i * shape[shape.size() - 1] + output] * log(node[shape.size() - 2][i * shape[shape.size() - 1] + output]) * batch_trainNode[i];
                        val_loss -= batchY[i * shape[shape.size() - 1] + output] * log(node[shape.size() - 2][i * shape[shape.size() - 1] + output]) * batch_validationNode[i];
                    }
                }
                if (batchY[i * shape[shape.size() - 1] + index] > 0)
                {
                    acc += 1 * batch_trainNode[i];
                    val_acc += 1 * batch_validationNode[i];
                }
            }
            delete[] batchX;
            delete[] batchY;
        }
        acc /= trainCount;
        loss /= trainCount;
        val_acc /= validationCount;
        val_loss /= validationCount;
        if (loss > preloss)
            loss_threshold += 1;
        else
            loss_threshold -= 0.5;
        if (loss_threshold < 0)
            loss_threshold = 0;
        if (loss_threshold > 10)
        {
            learning_rate *= decay;
            loss_threshold = 0;
        }
        if (val_acc > max_acc)
        {
            max_acc = val_acc;
            acc_count = 0;
        }
        else
            acc_count++;
        if (val_loss < min_loss)
        {
            min_loss = val_loss;
            loss_count = 0;
        }
        else
            loss_count++;
        preloss = loss;
        if (epoch % show_per_epoch == 0 || learning_rate < min_learning_rate || (acc_count >= earlystop && loss_count >= earlystop))
        {
            cout << "Epoch: " << epoch << " acc: " << acc << " loss: " << loss << " val_acc: " << val_acc << " val_loss: " << val_loss << " learning Rate: " << learning_rate << "\nCost Time: " << (clock() - BEGIN) / CLOCKS_PER_SEC << " seconds\n";
            if (learning_rate < min_learning_rate || acc_count >= earlystop)
                break;
            else
                BEGIN = clock();
        }
    }
    memory("Train Model Max", maxByte);
    for (int i = 0; i < shape.size() - 1; i++)
    {
        delete[] node[i];
        delete[] embedding[i];
    }
    delete[] groupIndex;
}
int GCN::fit(double* X, double* Y, vector<vector<edge>>& graph, vector<int>& activeNode,double learning_rate)
{
    int byte = 0;
    vector<int> label;
    if (graph.size() != activeNode.size())
        error("Node Size");
    for (int i = 0; i < shape.size() - 1; i++)
    {
        delete[] node[i];
        delete[] embedding[i];
        node[i] = new double[graph.size() * shape[i + 1]];
        embedding[i] = new double[graph.size() * shape[i + 1]];
        if (!(node[i] && embedding[i]))
            error("Memory");
        else
            byte += graph.size() * shape[i + 1] * 2 * sizeof(double);
    }
    predict(X, label, graph, activeNode, false);
    for (int layer = shape.size() - 2; layer >= 0; layer--)
    {
        memset(embedding[layer], 0, graph.size() * shape[layer + 1] * sizeof(double));
        memset(delta[layer], 0, (shape[layer] + 1) * shape[layer + 1] * sizeof(double));
        for (int i = 0; i < graph.size(); i++)
            for (int dist = 0; dist < shape[layer + 1]; dist++)
                for (int adj = 0; adj < graph[i].size(); adj++)
                {
                    if (layer == shape.size() - 2)
                        embedding[layer][i * shape[layer + 1] + dist] = (node[layer][graph[i][adj].N * shape[layer + 1] + dist] - Y[graph[i][adj].N * shape[layer + 1] + dist]) * graph[i][adj].V;
                    else
                        for (int src = 0; src < shape[layer + 2]; src++)
                            embedding[layer][i * shape[layer + 1] + dist] += embedding[layer + 1][graph[i][adj].N * shape[layer + 2] + src] * weight[layer + 1][dist * shape[layer + 2] + src] * graph[i][adj].V;
                }
        for (int i = 0; i < graph.size(); i++)
            if (activeNode[i] > 0)
                for (int dist = 0; dist < shape[layer + 1]; dist++)
                {
                    for (int src = 0; src < shape[layer]; src++)
                        if (layer == 0)
                            delta[layer][src * shape[layer + 1] + dist] -= learning_rate * X[i * shape[layer] + src] * embedding[layer][i * shape[layer + 1] + dist];
                        else
                            delta[layer][src * shape[layer + 1] + dist] -= learning_rate * node[layer - 1][i * shape[layer] + src] * embedding[layer][i * shape[layer + 1] + dist];
                    delta[layer][shape[layer] * shape[layer + 1] + dist] -= learning_rate * embedding[layer][i * shape[layer + 1] + dist];
                }
    }
    for (int layer = 0; layer < shape.size() - 1; layer++)
        for (int w = 0; w < (shape[layer] + 1) * shape[layer + 1]; w++)
            weight[layer][w] += delta[layer][w];
    return byte;
}
void GCN::predict(double* X, vector<int>& predictLabel, vector<vector<edge>>& graph, vector<int>& activeNode, bool output)
{
    double sum, max;
    int byte = 0, index;
    if (output)
    {
        if (graph.size() != activeNode.size())
            error("Size");
        for (int i = 0; i < shape.size() - 1; i++)
        {
            node[i] = new double[graph.size() * shape[i + 1]];
            embedding[i] = new double[graph.size() * shape[i + 1]];
            if (!(node[i] && embedding[i]))
                error("Memory");
            else
                byte += graph.size() * shape[i + 1] * 2 * sizeof(double);
        }
        memory("Test Model", byte);
    }
    for (int layer = 0; layer < shape.size() - 1; layer++)
    {
        memset(node[layer], 0, graph.size() * shape[layer + 1] * sizeof(double));
        for (int i = 0; i < graph.size(); i++)
        {
            for (int dist = 0; dist < shape[layer + 1]; dist++)
                for (int adj = 0; adj < graph[i].size(); adj++)
                {
                    for (int src = 0; src < shape[layer]; src++)
                        if (layer == 0)
                            node[layer][graph[i][adj].N * shape[layer + 1] + dist] += X[i * shape[layer] + src] * weight[layer][src * shape[layer + 1] + dist] * graph[i][adj].V;
                        else
                            node[layer][graph[i][adj].N * shape[layer + 1] + dist] += node[layer - 1][i * shape[layer] + src] * weight[layer][src * shape[layer + 1] + dist] * graph[i][adj].V;
                    node[layer][graph[i][adj].N * shape[layer + 1] + dist] += weight[layer][shape[layer] * shape[layer + 1] + dist] * graph[i][adj].V;
                }
            if (layer == shape.size() - 2)
            {
                sum = 0;
                for (int output = 0; output < shape[layer + 1]; output++)
                {
                    if (node[layer][i * shape[layer + 1] + output] > 500)
                        node[layer][i * shape[layer + 1] + output] = 500;
                    sum += pow(E, node[layer][i * shape[layer + 1] + output]);
                }
                for (int output = 0; output < shape[layer + 1]; output++)
                    node[layer][i * shape[layer + 1] + output] = pow(E, node[layer][i * shape[layer + 1] + output]) / sum;
            }
        }
    }
    if (output)
    {
        predictLabel.clear();
        for (int i = 0; i < graph.size(); i++)
        {
            index = 0;
            max = 0;
            for (int output = 0; output < shape[shape.size() - 1]; output++)
            {
                if (node[shape.size() - 2][i * shape[shape.size() - 1] + output] > max)
                {
                    max = node[shape.size() - 2][i * shape[shape.size() - 1] + output];
                    index = output;
                }
            }
            if (activeNode[i] > 0)
                predictLabel.push_back(index);
            else
                predictLabel.push_back(-1);
        }
        for (int i = 0; i < shape.size() - 1; i++)
        {
            delete[] node[i];
            delete[] embedding[i];
        }
    }
}
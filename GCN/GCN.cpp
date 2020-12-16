#include "GCN.h"
GCN::GCN(vector<int>& _shape, int seed)
{
	if (_shape.size() < 2)
    {
        cout << "Shape Error\n";
        exit(0);
    }
	shape.assign(_shape.begin(), _shape.end());
	weight = new double* [shape.size() - 1];
	delta = new double* [shape.size() - 1];
	node = new double* [shape.size() - 1];
	embedding = new double* [shape.size() - 1];
    if (!(weight && delta && node && embedding))
    {
        cout<< "Memory Error\n";
        exit(0);
    }
    int byte = 0;
	for (int i = 0; i < shape.size() - 1; i++)
	{
		weight[i] = new double[(shape[i] + 1) * shape[i + 1]];
		delta[i] = new double[(shape[i] + 1) * shape[i + 1]];
        if (!(weight[i] && delta[i]))
        {
            cout << "Memory Error\n";
            exit(0);
        }
        else
            byte += (shape[i] + 1) * shape[i + 1] * 2 * sizeof(double);
	}
    cout << "Build Model Cost " << byte << " Bytes Memory\n";
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
void GCN::fit(double* X, double* Y, vector<vector<edge>>& graph, vector<int>& trainNode, vector<int>& validationNode)
{
    double BEGIN = clock(), learning_rate = init_learning_rate, preloss = 1e3, max_acc = 0, acc_count = 0, sum, max, acc, loss, val_acc, val_loss;
    int byte = 0, trainCount = 0, validationCount = 0, index;
    if (graph.size() != trainNode.size() || graph.size() != validationNode.size())
    {
        cout << "Size Error\n";
        exit(0);
    }
    for (int i = 0; i < graph.size() - 1; i++)
    {
        trainCount += trainNode[i];
        validationCount += validationNode[i];
    }
    for (int i = 0; i < shape.size() - 1; i++)
    {
        node[i] = new double[graph.size() * shape[i + 1]];
        embedding[i] = new double[graph.size() * shape[i + 1]];
        if (!(node[i] && embedding[i]))
        {
            cout << "Memory Error\n";
            exit(0);
        }
        else
            byte += graph.size() * shape[i + 1] * 2 * sizeof(double);
    }
    cout << "Train Model Cost " << byte << " Bytes Memory\n";
    for (int epoch = 0; epoch < max_epoch; epoch++)
    {
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
        acc = 0;
        loss = 0;
        val_acc = 0;
        val_loss = 0;
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
                if (node[shape.size() - 2][i * shape[shape.size() - 1] + output] < 1e-15)
                {
                    loss -= Y[i * shape[shape.size() - 1] + output] * log(1e-15) * trainNode[i];
                    val_loss -= Y[i * shape[shape.size() - 1] + output] * log(1e-15) * validationNode[i];
                }
                else
                {
                    loss -= Y[i * shape[shape.size() - 1] + output] * log(node[shape.size() - 2][i * shape[shape.size() - 1] + output]) * trainNode[i];
                    val_loss -= Y[i * shape[shape.size() - 1] + output] * log(node[shape.size() - 2][i * shape[shape.size() - 1] + output]) * validationNode[i];
                }
            }
            if (Y[i * shape[shape.size() - 1] + index] > 0)
            {
                acc += 1 * trainNode[i];
                val_acc += 1 * validationNode[i];
            }
        }
        acc /= trainCount;
        loss /= trainCount;
        val_acc /= validationCount;
        val_loss /= validationCount;
        if (loss > preloss)
            learning_rate *= decay;
        if (val_acc > max_acc)
        {
            max_acc = val_acc;
            acc_count = 0;
        }
        else
            acc_count++;
        preloss = loss;
        if (learning_rate < min_learning_rate || acc_count >= earlystop)
        {
            cout << "Epoch: " << epoch << " acc: " << acc << " loss: " << loss << " val_acc: " << val_acc << " val_loss: " << val_loss << " learning Rate: " << learning_rate << "\nCost Time: " << (clock() - BEGIN) / CLOCKS_PER_SEC << " seconds\n";
            break;
        }
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
                if(trainNode[i]>0)
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
        if (epoch % show_per_epoch == 0)
        {
            cout << "Epoch: " << epoch << " acc: " << acc << " loss: " << loss << " val_acc: " << val_acc << " val_loss: " << val_loss << " learning Rate: " << learning_rate << "\nCost Time: " << (clock() - BEGIN) / CLOCKS_PER_SEC << " seconds\n";
            BEGIN = clock();
        }
    }
    for (int i = 0; i < shape.size() - 1; i++)
    {
        delete[] node[i];
        delete[] embedding[i];
    }
}
void GCN::predict(double* X, vector<int>& predictLabel, vector<vector<edge>>& graph, vector<int>& testNode)
{
    double sum, max;
    int byte = 0, testCount = 0, validationCount = 0, index;
    if (graph.size() != testNode.size())
    {
        cout << "Size Error\n";
        exit(0);
    }
    for (int i = 0; i < graph.size() - 1; i++)
        testCount += testNode[i];
    for (int i = 0; i < shape.size() - 1; i++)
    {
        node[i] = new double[graph.size() * shape[i + 1]];
        embedding[i] = new double[graph.size() * shape[i + 1]];
        if (!(node[i] && embedding[i]))
        {
            cout << "Memory Error\n";
            exit(0);
        }
        else
            byte += graph.size() * shape[i + 1] * 2 * sizeof(double);
    }
    cout << "Test Model Cost " << byte << " Bytes Memory\n";
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
        if (testNode[i] > 0)
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
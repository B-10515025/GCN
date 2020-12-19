#include "GCN.h"
#include "utils.h"
#include <iostream>
#include <time.h>
#include <algorithm>
GCN::GCN(vector<int>& _shape, bool _gpu)
{
    if (_shape.size() < 2)
        error("Shape");
    shape.assign(_shape.begin(), _shape.end());
    weightCount = 0;
    nodeOffset = new int[shape.size()];
    weightOffset = new int[shape.size()];
    for (int i = 1; i < shape.size(); i++)
    {
        weightOffset[i - 1] = weightCount;
        weightCount += (shape[i - 1] + 1) * shape[i];
    }
    weight = new double[weightCount];
    best_weight = new double[weightCount];
    delta = new double[weightCount];
    if (!(weight && delta))
        error("Memory");
    memory("Build Model", 3 * weightCount * sizeof(double));
    GPU = InitCUDA() & _gpu;
    if (GPU)
    {
        int cuda = 0;
        cuda |= cudaMalloc((void**)&dev_weight, weightCount * sizeof(double));
        cuda |= cudaMalloc((void**)&dev_delta, weightCount * sizeof(double));
        if (cuda != cudaSuccess)
            error("Cuda Malloc Memory");
        cout << "Using CUDA GPU.\n";
    }
    else
        cout << "Using CPU.\n";
    srand(time(NULL));
    for (int i = 0; i < weightCount; i++)
        weight[i] = (rand() - rand()) / 65536.0;
    max_epoch = 200;
    show_per_epoch = 1;
    earlystop = 10;
    init_learning_rate = 1e-3;
    min_learning_rate = 1e-8;
    decay = 1;
}
GCN::~GCN()
{
    if (GPU)
    {
        cudaFree(dev_weight);
        cudaFree(dev_delta);
        cudaFree(dev_edgeOffset);
        cudaFree(dev_edgeIndex);
        cudaFree(dev_edgeValue);
    }
    delete[] nodeOffset;
    delete[] weightOffset;
    delete[] weight;
    delete[] best_weight;
    delete[] delta;
}
bool GCN::InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0)
        return false;
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
            if (prop.major >= 1)
            {
                cudaSetDevice(i);
                return true;
            }
    }
    return false;
}
void GCN::computeSubtractionCPU(double* srcNode, double* distNode, double* weight, int Count)
{
    for (int node = 0; node < Count; node++)
        distNode[node] = srcNode[node] - weight[node];
}
void GCN::computeEmbeddingCPU(double* srcNode, double* distNode, double* weight, int srcCount, int distCount, int nodeCount, bool back)
{
    for (int node = 0; node < nodeCount; node++)
        for (int dist = 0; dist < distCount; dist++)
            if (back)
                for (int src = 0; src < srcCount; src++)
                    distNode[node * distCount + dist] += srcNode[node * srcCount + src] * weight[dist * srcCount + src];
            else
            {
                for (int src = 0; src < srcCount; src++)
                    distNode[node * distCount + dist] += srcNode[node * srcCount + src] * weight[src * distCount + dist];
                distNode[node * distCount + dist] += weight[srcCount * distCount + dist];
            }
}
void GCN::computeNodeCPU(double* srcNode, double* distNode, vector<vector<edge>>& graph, int Count, bool last)
{
    for (int node = 0; node < graph.size(); node++)
        if (last)
        {
            double sum = 0;
            for (int dist = 0; dist < Count; dist++)
            {
                if (srcNode[node * Count + dist] > 500)
                    srcNode[node * Count + dist] = 500;
                sum += pow(E, srcNode[node * Count + dist]);
            }
            for (int dist = 0; dist < Count; dist++)
                distNode[node * Count + dist] = pow(E, srcNode[node * Count + dist]) / sum;
        }
        else
            for (int dist = 0; dist < Count; dist++)
                for (int src = 0; src < graph[node].size(); src++)
                    distNode[graph[node][src].N * Count + dist] += srcNode[node * Count + dist] * graph[node][src].V;
}
void GCN::computeDeltaCPU(double* srcNode, double* distNode, double* weight, int srcCount, int distCount, int nodeCount, vector<int>& activeNode)
{
    for (int node = 0; node < nodeCount; node++)
        for (int dist = 0; dist < distCount; dist++)
        {
            for (int src = 0; src < srcCount; src++)
                distNode[src * distCount + dist] -= srcNode[node * srcCount + src] * weight[node * distCount + dist] * activeNode[node];
            distNode[srcCount * distCount + dist] -= weight[node * distCount + dist] * activeNode[node];
        }
}
void GCN::computeWeightCPU(double* srcNode, double* distNode, int Count, double learning_rate)
{
    for (int i = 0; i < Count; i++)
        distNode[i] += srcNode[i] * learning_rate;
}
void GCN::HostToDevice(double* X, double* Y, vector<vector<edge>>& graph, vector<int>& activeNode)
{
    int edgeCount = 0, cuda = 0;
    int* edgeOffset, * edgeIndex, * node;
    double* edgeValue;
    edgeOffset = new int[graph.size()];
    for (int i = 0; i < graph.size(); i++)
    {
        edgeCount += graph[i].size();
        edgeOffset[i] = edgeCount;
    }
    edgeIndex = new int[edgeCount];
    edgeValue = new double[edgeCount];
    for (int i = 0; i < graph.size(); i++)
        for (int j = 1; j <= graph[i].size(); j++)
        {
            edgeIndex[edgeOffset[i] - j] = graph[i][j - 1].N;
            edgeValue[edgeOffset[i] - j] = graph[i][j - 1].V;
        }
    node = new int[activeNode.size()];
    for (int i = 0; i < activeNode.size(); i++)
        node[i] = activeNode[i];
    cudaFree(dev_edgeOffset);
    cudaFree(dev_edgeIndex);
    cudaFree(dev_edgeValue);
    cuda |= cudaMalloc((void**)&dev_edgeOffset, graph.size() * sizeof(int));
    cuda |= cudaMalloc((void**)&dev_edgeIndex, edgeCount * sizeof(int));
    cuda |= cudaMalloc((void**)&dev_edgeValue, edgeCount * sizeof(double));
    if (cuda != cudaSuccess)
        error("Cuda Malloc Memory");   
    cuda |= cudaMemcpy(dev_X, X, graph.size() * shape[0] * sizeof(double), cudaMemcpyHostToDevice);
    cuda |= cudaMemcpy(dev_Y, Y, graph.size() * shape[shape.size() - 1] * sizeof(double), cudaMemcpyHostToDevice);
    cuda |= cudaMemcpy(dev_weight, weight, weightCount * sizeof(double), cudaMemcpyHostToDevice);
    cuda |= cudaMemcpy(dev_edgeOffset, edgeOffset, graph.size() * sizeof(int), cudaMemcpyHostToDevice);
    cuda |= cudaMemcpy(dev_edgeIndex, edgeIndex, edgeCount * sizeof(int), cudaMemcpyHostToDevice);
    cuda |= cudaMemcpy(dev_edgeValue, edgeValue, edgeCount * sizeof(double), cudaMemcpyHostToDevice);
    cuda |= cudaMemcpy(dev_activeNode, node, activeNode.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda != cudaSuccess)
        error("Cuda Memory Copy");
    delete[] edgeOffset;
    delete[] edgeIndex;
    delete[] edgeValue;
    delete[] node;
}
void GCN::DeviceToHost()
{
    int cuda = 0;
    cuda |= cudaMemcpy(node, dev_node, nodeCount * sizeof(double), cudaMemcpyDeviceToHost);
    cuda |= cudaMemcpy(weight, dev_weight, weightCount * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda != cudaSuccess)
        error("Cuda Memory Copy");
}
__global__ void computeSubtractionGPU(double* srcNode, double* distNode, double* weight, int Count)
{
    for (int node = blockIdx.x * blockDim.x + threadIdx.x; node < Count; node += gridDim.x * blockDim.x)
        distNode[node] = srcNode[node] - weight[node];
}
__global__ void computeEmbeddingGPU(double* srcNode, double* distNode, double* weight, int srcCount, int distCount, int nodeCount, bool back)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nodeCount * distCount; i += gridDim.x * blockDim.x)
    {
        int node = i / distCount, dist = i % distCount;
        if (back)
            for (int src = 0; src < srcCount; src++)
                distNode[i] += srcNode[node * srcCount + src] * weight[dist * srcCount + src];
        else
        {
            for (int src = 0; src < srcCount; src++)
                distNode[i] += srcNode[node * srcCount + src] * weight[src * distCount + dist];
            distNode[i] += weight[srcCount * distCount + dist];
        }
    }
}
__global__ void computeNodeGPU(double* srcNode, double* distNode, int nodeCount, int* edgeOffset, int* edgeIndex, double* edgeValue, int Count, bool last)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nodeCount * Count; i += gridDim.x * blockDim.x)
    {
        int node = i / Count, dist = i % Count;
        if (last)
        {
            double sum = 0;
            for (int src = 0; src < Count; src++)
            {
                if (srcNode[node * Count + src] > 500)
                    srcNode[node * Count + src] = 500;
                sum += pow(E, srcNode[node * Count + src]);
            }
            distNode[i] = pow(E, srcNode[node * Count + dist]) / sum;
        }
        else
        {
            int start = 0;
            if (node > 0)
                start = edgeOffset[node - 1];
            for (int src = start; src < edgeOffset[node]; src++)
                distNode[i] += srcNode[edgeIndex[src] * Count + dist] * edgeValue[src];
        }
    }
}
__global__ void computeDeltaGPU(double* srcNode, double* distNode, double* weight, int srcCount, int distCount, int nodeCount, int* activeNode)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (srcCount + 1) * distCount; i += gridDim.x * blockDim.x)
    {
        int src = i / distCount, dist = i % distCount;
        for (int node = 0; node < nodeCount; node++)
            if (src < srcCount)
                distNode[src * distCount + dist] -= srcNode[node * srcCount + src] * weight[node * distCount + dist] * activeNode[node];
            else
                distNode[src * distCount + dist] -= weight[node * distCount + dist] * activeNode[node];
    }
}
__global__ void computeWeightGPU(double* srcNode, double* distNode, int Count, double learning_rate)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Count; i+= gridDim.x * blockDim.x)
        distNode[i] += srcNode[i] * learning_rate;
}
void GCN::train(double* X, double* Y, vector<vector<edge>>& graph, vector<vector<int>>& cluster, int batchSize, vector<int>& trainNode, vector<int>& validationNode)
{
    vector<vector<edge>> batch_graph;
    vector<int> batch_trainNode, batch_validationNode, clusterSize;
    double BEGIN = clock(), learning_rate = init_learning_rate, preloss = 1e3, max_acc = 0, acc_count = 0, acc, min_loss = 1e10, loss_count = 0, loss, val_acc, val_loss, sum, max, loss_threshold = 0;
    int index, batchCount, trainCount = 0, validationCount = 0, maxCount = 0, byte = 0;
    int* groupIndex, * batchNodeIndex, * Mapper;
    double* batchX, * batchY;
    if (cluster.size() % batchSize != 0)
        error("Group Batch Size");
    for (int i = 0; i < cluster.size(); i++)
        if (graph.size() != cluster[i].size())
            error("Group Node Size");
    if (graph.size() != trainNode.size() || graph.size() != validationNode.size())
        error("Node Size");
    clusterSize = vector<int>(cluster.size(), 0);
    for (int i = 0; i < cluster.size(); i++)
        for (int j = 0; j < cluster[i].size(); j++)
            clusterSize[i] += cluster[i][j];
    sort(clusterSize.begin(), clusterSize.end());
    for (int i = 1; i <= batchSize; i++)
        maxCount += clusterSize[clusterSize.size() - i];
    for (int i = 1; i < shape.size(); i++)
        maxNodeCount += shape[i] * maxCount;
    groupIndex = new int[cluster.size()];
    batchNodeIndex = new int[graph.size()];
    Mapper = new int[graph.size()];
    node = new double[maxNodeCount];
    differential = new double[maxNodeCount];
    embedding = new double[maxNodeCount];
    byte = (cluster.size() + 2 * graph.size()) * sizeof(int) + maxNodeCount * 3 * sizeof(double);
    if (GPU)
    {
        int cuda = 0;
        cuda |= cudaMalloc((void**)&dev_X, shape[0] * maxCount * sizeof(double));
        cuda |= cudaMalloc((void**)&dev_Y, shape[shape.size() - 1] * maxCount * sizeof(double));
        cuda |= cudaMalloc((void**)&dev_embedding, maxNodeCount * sizeof(double));
        cuda |= cudaMalloc((void**)&dev_node, maxNodeCount * sizeof(double));
        cuda |= cudaMalloc((void**)&dev_differential, maxNodeCount * sizeof(double));
        cuda |= cudaMalloc((void**)&dev_activeNode, maxCount * sizeof(int));
        if (cuda != cudaSuccess)
            error("Cuda Malloc Memory");
        byte += shape[0] * maxCount * sizeof(double) + shape[shape.size() - 1] * maxCount * sizeof(double) + maxCount * sizeof(int);
    }
    if (!(groupIndex && batchNodeIndex && Mapper && node && differential && embedding))
        error("Memory");
    memory("Train Model", byte);
    for (int i = 0; i < graph.size(); i++)
    {
        trainCount += trainNode[i];
        validationCount += validationNode[i];
    }
    for (int epoch = 0; epoch < max_epoch; epoch++)
    {
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
            fit(batchX, batchY, batch_graph, batch_trainNode, learning_rate);
            for (int i = 0; i < batch_graph.size(); i++)
            {
                index = 0;
                max = 0;
                for (int output = 0; output < shape[shape.size() - 1]; output++)
                {
                    if (node[nodeOffset[shape.size() - 2] + i * shape[shape.size() - 1] + output] > max)
                    {
                        max = node[nodeOffset[shape.size() - 2] + i * shape[shape.size() - 1] + output];
                        index = output;
                    }
                    if (node[nodeOffset[shape.size() - 2] + i * shape[shape.size() - 1] + output] < 1e-15)
                    {
                        loss -= batchY[i * shape[shape.size() - 1] + output] * log(1e-15) * batch_trainNode[i];
                        val_loss -= batchY[i * shape[shape.size() - 1] + output] * log(1e-15) * batch_validationNode[i];
                    }
                    else
                    {
                        loss -= batchY[i * shape[shape.size() - 1] + output] * log(node[nodeOffset[shape.size() - 2] + i * shape[shape.size() - 1] + output]) * batch_trainNode[i];
                        val_loss -= batchY[i * shape[shape.size() - 1] + output] * log(node[nodeOffset[shape.size() - 2] + i * shape[shape.size() - 1] + output]) * batch_validationNode[i];
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
            memcpy(best_weight, weight, weightCount * sizeof(double));
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
    memcpy(weight, best_weight, weightCount * sizeof(double));
    delete[] groupIndex;
    delete[] batchNodeIndex;
    delete[] Mapper;
    delete[] node;
    delete[] differential;
    delete[] embedding;
    if (GPU)
    {
        cudaFree(dev_X);
        cudaFree(dev_Y);
        cudaFree(dev_embedding);
        cudaFree(dev_node);
        cudaFree(dev_differential);
        cudaFree(dev_activeNode);
    }
}
void GCN::fit(double* X, double* Y, vector<vector<edge>>& graph, vector<int>& activeNode, double learning_rate)
{
    int byte = 0;
    vector<int> label;
    if (graph.size() != activeNode.size())
        error("Node Size");
    nodeCount = 0;
    for (int i = 1; i < shape.size(); i++)
    {
        nodeOffset[i - 1] = nodeCount;
        nodeCount += graph.size() * shape[i];
    }
    if (GPU)
    {
        int cuda = 0;
        HostToDevice(X, Y, graph, activeNode);
        cuda |= cudaMemset(dev_embedding, 0, nodeCount * sizeof(double));
        cuda |= cudaMemset(dev_node, 0, nodeCount * sizeof(double));
        if (cuda != cudaSuccess)
            error("Cuda Memory Set");
        for (int layer = 0; layer < shape.size() - 1; layer++)
        {
            if (layer == 0)
                computeEmbeddingGPU << <64, 256 >> > (dev_X, dev_embedding + nodeOffset[layer], dev_weight + weightOffset[layer], shape[layer], shape[layer + 1], graph.size(), false);
            else
                computeEmbeddingGPU << <64, 256 >> > (dev_node + nodeOffset[layer - 1], dev_embedding + nodeOffset[layer], dev_weight + weightOffset[layer], shape[layer], shape[layer + 1], graph.size(), false);
            computeNodeGPU << <64, 256 >> > (dev_embedding + nodeOffset[layer], dev_node + nodeOffset[layer], graph.size(), dev_edgeOffset, dev_edgeIndex, dev_edgeValue, shape[layer + 1], layer == shape.size() - 2);
        }
        cuda |= cudaMemset(dev_differential, 0, nodeCount * sizeof(double));
        cuda |= cudaMemset(dev_embedding, 0, nodeCount * sizeof(double));
        cuda |= cudaMemset(dev_delta, 0, weightCount * sizeof(double));
        if (cuda != cudaSuccess)
            error("Cuda Memory Set");
        for (int layer = shape.size() - 2; layer >= 0; layer--)
        {
            if (layer == shape.size() - 2)
                computeSubtractionGPU << <64, 256 >> > (dev_node + nodeOffset[layer], dev_embedding + nodeOffset[layer], dev_Y, graph.size() * shape[layer + 1]);
            else
                computeEmbeddingGPU << <64, 256 >> > (dev_differential + nodeOffset[layer + 1], dev_embedding + nodeOffset[layer], dev_weight + weightOffset[layer + 1], shape[layer + 2], shape[layer + 1], graph.size(), true);
            computeNodeGPU << <64, 256 >> > (dev_embedding + nodeOffset[layer], dev_differential + nodeOffset[layer], graph.size(), dev_edgeOffset, dev_edgeIndex, dev_edgeValue, shape[layer + 1], false);
            if (layer > 0)
                computeDeltaGPU << <64, 256 >> > (dev_node + nodeOffset[layer - 1], dev_delta + weightOffset[layer], dev_differential + nodeOffset[layer], shape[layer], shape[layer + 1], graph.size(), dev_activeNode);
            else
                computeDeltaGPU << <64, 256 >> > (dev_X, dev_delta + weightOffset[layer], dev_differential + nodeOffset[layer], shape[layer], shape[layer + 1], graph.size(), dev_activeNode);
        }
        computeWeightGPU << <64, 256 >> > (dev_delta, dev_weight, weightCount, learning_rate);
        DeviceToHost();
    }
    else
    {
        memset(embedding, 0, nodeCount * sizeof(double));
        memset(node, 0, nodeCount * sizeof(double));
        for (int layer = 0; layer < shape.size() - 1; layer++)
        {
            if (layer == 0)
                computeEmbeddingCPU(X, embedding + nodeOffset[layer], weight + weightOffset[layer], shape[layer], shape[layer + 1], graph.size(), false);
            else
                computeEmbeddingCPU(node + nodeOffset[layer - 1], embedding + nodeOffset[layer], weight + weightOffset[layer], shape[layer], shape[layer + 1], graph.size(), false);
            computeNodeCPU(embedding + nodeOffset[layer], node + nodeOffset[layer], graph, shape[layer + 1], layer == shape.size() - 2);
        }
        memset(differential, 0, nodeCount * sizeof(double));
        memset(embedding, 0, nodeCount * sizeof(double));
        memset(delta, 0, weightCount * sizeof(double));
        for (int layer = shape.size() - 2; layer >= 0; layer--)
        {
            if (layer == shape.size() - 2)
                computeSubtractionCPU(node + nodeOffset[layer], embedding + nodeOffset[layer], Y, graph.size() * shape[layer + 1]);
            else
                computeEmbeddingCPU(differential + nodeOffset[layer + 1], embedding + nodeOffset[layer], weight + weightOffset[layer + 1], shape[layer + 2], shape[layer + 1], graph.size(), true);
            computeNodeCPU(embedding + nodeOffset[layer], differential + nodeOffset[layer], graph, shape[layer + 1], false);
            if (layer > 0)
                computeDeltaCPU(node + nodeOffset[layer - 1], delta + weightOffset[layer], differential + nodeOffset[layer], shape[layer], shape[layer + 1], graph.size(), activeNode);
            else
                computeDeltaCPU(X, delta + weightOffset[layer], differential + nodeOffset[layer], shape[layer], shape[layer + 1], graph.size(), activeNode);
        }
        computeWeightCPU(delta, weight, weightCount, learning_rate);
    }
}
void GCN::predict(double* X, vector<int>& predictLabel, vector<vector<edge>>& graph, vector<int>& activeNode)
{
    double sum, max;
    int byte = 0, index;
    if (graph.size() != activeNode.size())
        error("Size");
    nodeCount = 0;
    for (int i = 1; i < shape.size(); i++)
    {
        nodeOffset[i - 1] = nodeCount;
        nodeCount += graph.size() * shape[i];
    }
    node = new double[nodeCount];
    embedding = new double[nodeCount];
    if (!(node && embedding))
        error("Memory");
    memory("Test Model", nodeCount * 2 * sizeof(double));
    memset(embedding, 0, nodeCount * sizeof(double));
    memset(node, 0, nodeCount * sizeof(double));
    for (int layer = 0; layer < shape.size() - 1; layer++)
    {
        if (layer == 0)
            computeEmbeddingCPU(X, embedding + nodeOffset[layer], weight + weightOffset[layer], shape[layer], shape[layer + 1], graph.size(), false);
        else
            computeEmbeddingCPU(node + nodeOffset[layer - 1], embedding + nodeOffset[layer], weight + weightOffset[layer], shape[layer], shape[layer + 1], graph.size(), false);
        computeNodeCPU(embedding + nodeOffset[layer], node + nodeOffset[layer], graph, shape[layer + 1], layer == shape.size() - 2);
    }
    predictLabel.clear();
    for (int i = 0; i < graph.size(); i++)
    {
        index = 0;
        max = 0;
        for (int output = 0; output < shape[shape.size() - 1]; output++)
        {
            if (node[nodeOffset[shape.size() - 2] + i * shape[shape.size() - 1] + output] > max)
            {
                max = node[nodeOffset[shape.size() - 2] + i * shape[shape.size() - 1] + output];
                index = output;
            }
        }
        if (activeNode[i] > 0)
            predictLabel.push_back(index);
        else
            predictLabel.push_back(-1);
    }
    delete[] node;
    delete[] embedding;
}
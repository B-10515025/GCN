#pragma once
#include "utils.h"
#include <cuda_runtime.h>
#include <vector>
#define E 2.7182818284
using namespace std;
class GCN
{
	bool GPU;
	int nodeCount, maxNodeCount, weightCount;
	int* nodeOffset, * weightOffset;
	double* weight, * delta, * node, * differential, * embedding;
	int* dev_edgeOffset, * dev_edgeIndex, * dev_activeNode;
	double* dev_X, * dev_Y, * dev_weight, * dev_delta, * dev_node, * dev_differential, * dev_embedding, * dev_edgeValue;
	vector<int> shape;
	bool InitCUDA();
	void fit(double*, double*, vector<vector<edge>>&, vector<int>&, double);
	void computeSubtractionCPU(double*, double*, double*, int);
	void computeEmbeddingCPU(double*, double*, double*, int, int, int, bool);
	void computeNodeCPU(double*, double*, vector<vector<edge>>&, int, bool);
	void computeDeltaCPU(double*, double*, double*, int, int, int, vector<int>&);
	void computeWeightCPU(double*, double*, int, double);
	void HostToDevice(double*, double*, vector<vector<edge>>& graph, vector<int>& activeNode);
	void DeviceToHost();
public:
	int max_epoch, show_per_epoch, earlystop;
	double init_learning_rate, min_learning_rate, decay;
	GCN(vector<int>&, bool);
	~GCN();
	void train(double*, double*, vector<vector<edge>>&, vector<vector<int>>&, int, vector<int>&, vector<int>&);
	void predict(double*, vector<int>&, vector<vector<edge>>&, vector<int>&);
};
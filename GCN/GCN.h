#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <time.h>
#define E 2.7182818284
using namespace std;
struct Data
{
	string id, label;
	vector<double> features;
};
struct edge
{
	int N;
	double V;
};
class GCN
{
	vector<int> shape;
	double** weight, ** delta, ** node, ** embedding;
public:
	int max_epoch, show_per_epoch, earlystop;
	double init_learning_rate, min_learning_rate, decay;
	GCN(vector<int>&, int);
	~GCN();
	void train(double*, double*, vector<vector<edge>>&, vector<vector<int>>&, int, vector<int>&, vector<int>&);
	int fit(double*, double*, vector<vector<edge>>&, vector<int>&, double);
	void predict(double*, vector<int>&, vector<vector<edge>>&, vector<int>&, bool);
};
void error(string msg);
void memory(string msg, int byte);
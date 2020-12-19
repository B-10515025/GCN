#pragma once
#include <string>
#include <vector>
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
void error(string msg);
void memory(string msg, int byte);
void getRandom(int* index, int max);
int readFeatures(const char*, double*&, double*&, vector<string>&, vector<string>&);
void readGraph(const char*, vector<string>&, vector<vector<edge>>&);
void nodeSplit(int, vector<int>&, int, vector<int>&, int, vector<int>&, int);
void nodeCluster(vector<vector<int>>&, vector<vector<edge>>&, int, string);
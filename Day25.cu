#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define NUM_SIMULATIONS 1024  
#define MAX_DEPTH 100         


struct GameState {
    int moves[10];  
    int num_moves;
    bool is_terminal;
    float reward;  

    __device__ GameState next_state(int action) {
        GameState new_state = *this;
  
        new_state.reward += (action % 2 == 0) ? 1.0f : -1.0f;
        new_state.is_terminal = (new_state.reward > 10 || new_state.reward < -10);
        return new_state;
    }

    __device__ int get_random_action(curandState* state) {
        if (num_moves == 0) return -1;
        return moves[curand(state) % num_moves];
    }
};

struct Node {
    GameState state;
    int visits;
    float value;
};

__device__ float rollout(GameState state, curandState* rand_state) {
    int depth = 0;
    while (!state.is_terminal && depth < MAX_DEPTH) {
        int action = state.get_random_action(rand_state);
        if (action == -1) break; 
        state = state.next_state(action);
        depth++;
    }
    return state.reward;
}

__global__ void mcts_kernel(Node* nodes, int num_nodes, float* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_nodes) return;

    curandState rand_state;
    curand_init(idx, 0, 0, &rand_state);

    float total_reward = 0;
    for (int i = 0; i < NUM_SIMULATIONS; i++) {
        total_reward += rollout(nodes[idx].state, &rand_state);
    }

    results[idx] = total_reward / NUM_SIMULATIONS;
}

void run_mcts(Node* host_nodes, int num_nodes) {
    Node* device_nodes;
    float* device_results;
    float* host_results = (float*)malloc(num_nodes * sizeof(float));

    cudaMalloc(&device_nodes, num_nodes * sizeof(Node));
    cudaMalloc(&device_results, num_nodes * sizeof(float));

    cudaMemcpy(device_nodes, host_nodes, num_nodes * sizeof(Node), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;

    mcts_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_nodes, num_nodes, device_results);

    cudaMemcpy(host_results, device_results, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_nodes; i++) {
        host_nodes[i].value = host_results[i];
    }

    free(host_results);
    cudaFree(device_nodes);
    cudaFree(device_results);
}

int main() {
   
    Node root;
    root.state.num_moves = 10;
    root.state.is_terminal = false;
    root.visits = 0;
    root.value = 0;

    run_mcts(&root, 1);

    printf("MCTS result: %f\n", root.value);
    return 0;
}
#include <bits/stdc++.h>
using namespace std;

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dsigmoid(double x) { return x * (1 - x); }
const double learnRate = 0.2;
const int n = 784, in_channel = 500, out_channel = 10;

double x[n], hidden_in[in_channel], hidden_out[in_channel],
    output_in[out_channel], output_out[out_channel], d[out_channel];
double w1[n][in_channel], w2[in_channel][out_channel];
double ranDouble() { return rand() % 10 / 5. - 1; }

void init() {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < in_channel; j++)
      w1[i][j] = ranDouble();

  for (int i = 0; i < in_channel; i++)
    for (int j = 0; j < out_channel; j++)
      w2[i][j] = ranDouble();
}
void forward() {
  memset(hidden_in, 0, sizeof(hidden_in));
  for (int i = 0; i < n; i++)
    for (int j = 0; j < in_channel; j++)
      hidden_in[j] += x[i] * w1[i][j];
  for (int i = 0; i < in_channel; i++)
    hidden_out[i] = sigmoid(hidden_in[i]);
  memset(output_in, 0, sizeof(output_in));
  for (int i = 0; i < in_channel; i++)
    for (int j = 0; j < out_channel; j++)
      output_in[j] += hidden_out[i] * w2[i][j];
  for (int i = 0; i < out_channel; i++)
    output_out[i] = sigmoid(output_in[i]);
}
int getAns() {
  int ans = 0;
  for (int i = 1; i < out_channel; i++)
    if (output_out[i] > output_out[ans])
      ans = i;
  return ans;
}
void back() {
  for (int i = 0; i < in_channel; i++)
    for (int j = 0; j < out_channel; j++) {
      double delta =
          (output_out[j] - d[j]) * dsigmoid(output_out[j]) * hidden_out[i];
      w2[i][j] -= delta * learnRate;
    }
  double W2[in_channel];
  memset(W2, 0, sizeof(W2));
  for (int j = 0; j < in_channel; j++)
    for (int k = 0; k < out_channel; k++)
      W2[j] += (output_out[k] - d[k]) * dsigmoid(output_out[k]) * w2[j][k];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < in_channel; j++) {
      double delta = dsigmoid(hidden_out[j]) * x[i] * W2[j];
      w1[i][j] -= delta * learnRate;
    }
}

FILE *fImg, *fAns;
FILE *tImg, *tAns;
int last[100000];
int train_success = 0;

void train(int cas) {
  unsigned char img[n], num;
  fread(img, 1, n, fImg);
  for (int i = 0; i < n; i++)
    x[i] = img[i] / 255.;
  fread(&num, 1, 1, fAns);
  for (int i = 0; i < out_channel; i++)
    d[i] = (num == i);
  forward();
  int ans = getAns();
  last[cas] = (num == ans);
  back();
  int chunk = 100, success = 0;
  if (cas % chunk == 0) {
    for (int i = 0; i < chunk; i++)
      success += last[cas - i];
    train_success += success;
    cout << cas << "tasks finised." << " Chunk accuracy: " << success << "%\n";
  }
}

int test_success = 0;
void test(int cas) {
  unsigned char img[n], num;
  fread(img, 1, n, tImg);
  for (int i = 0; i < n; i++)
    x[i] = img[i] / 255.;
  fread(&num, 1, 1, tAns);
  for (int i = 0; i < out_channel; i++)
    d[i] = (num == i);
  forward();
  int ans = getAns();
  last[cas] = (num == ans);
  int chunk = 100, success = 0;
  if (cas % chunk == 0) {
    for (int i = 0; i < chunk; i++)
      success += last[cas - i];
    test_success += success;
    cout << cas << "tests finised." << " Chunk accuracy: " << success << "%\n";
  }
}

int main() {
  fImg = fopen("train-images.idx3-ubyte", "rb");
  fseek(fImg, 16, SEEK_SET);
  fAns = fopen("train-labels.idx1-ubyte", "rb");
  fseek(fAns, 8, SEEK_SET);
  tImg = fopen("t10k-images.idx3-ubyte", "rb");
  fseek(tImg, 16, SEEK_SET);
  tAns = fopen("t10k-labels.idx1-ubyte", "rb");
  fseek(tAns, 8, SEEK_SET);
  init();
  for (int cas = 1; cas <= 60000; cas++) {
    train(cas);
  }
  cout << "Training Accuracy: " << train_success * 100.0 / 60000 << '\n';
  for (int cas = 1; cas <= 10000; ++cas) {
    test(cas);
  }
  cout << "Test Accuracy: " << test_success * 100.0 / 10000 << '\n';
  // cerr<<"done\n";
}
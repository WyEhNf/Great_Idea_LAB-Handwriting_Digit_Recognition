\*
This aims to test the the accuracy of the model when the model's trained more times and to test the effect of dropout layer.
  *\
#include <bits/stdc++.h>

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using Mat = cv::Mat;
const int full = 0, valid = 1, same = 2;
const int AvePool = 0, MaxPool = 1;
// definition
struct Train_Option {
  int it_num;
  float learning_rate;
};
typedef Train_Option Opts;
struct Covolutional_Layer {
  int inWidth, inHeight, mapSize, inChannels, outChannels;
  bool isFull;
  vector<vector<Mat> > mapData;
  Mat basicData;
  vector<Mat> v, y, grad;
};
typedef Covolutional_Layer CovLayer;

struct Pooling_Layer {
  int inWidth, inHeight, mapSize, inChannels, outChannels, poolType;
  Mat basicData;
  vector<Mat> y, grad, max_pos;
};
typedef Pooling_Layer PoolLayer;

struct Dropout_Layer {
  int inWidth, inHeight, inChannels;
  float dropout_rate;
  vector<Mat> mask;
  vector<Mat> y;
  bool isTraining;
};
typedef Dropout_Layer DropLayer;

struct Output_Layer {
  int inNum, outNum;
  bool isFull;
  Mat wData, basicData;
  Mat v, y, grad;
};
typedef Output_Layer OutLayer;

struct CNN_netWork {
  int layers;
  CovLayer C1, C3;
  PoolLayer S2, S4;
  DropLayer D1;
  OutLayer O5;
  Mat eps, L;
};
typedef CNN_netWork CNN;

// initialize
CovLayer initCov(int inWidth, int inHeight, int mapSize, int inChannels,
                 int outChannels) {
  CovLayer cur;

  cur = (CovLayer){inWidth, inHeight, mapSize, inChannels, outChannels, true};

  srand((unsigned)time(0));
  for (int i = 0; i < inChannels; ++i) {
    vector<Mat> tmp;
    for (int j = 0; j < outChannels; ++j) {
      Mat obj(mapSize, mapSize, CV_32FC1);
      for (int r = 0; r < mapSize; ++r) {
        for (int c = 0; c < mapSize; ++c) {
          float coefficient =
              ((rand() & 1) ? 1 : -1) * (1.0 * rand() / RAND_MAX);
          float base =
              sqrt(6.0 / (mapSize * mapSize * (inChannels + outChannels)));
          obj.ptr<float>(r)[c] = coefficient * base;
        }
      }
      tmp.push_back(obj.clone());
    }
    cur.mapData.push_back(tmp);
  }

  cur.basicData = Mat::zeros(1, outChannels, CV_32FC1);
  int validW = inWidth - mapSize + 1, validH = inHeight - mapSize + 1;
  Mat Valid = Mat::zeros(validH, validW, CV_32FC1);
  for (int i = 0; i < outChannels; ++i) {
    cur.grad.push_back(Valid.clone());
    cur.y.push_back(Valid.clone());
    cur.v.push_back(Valid.clone());
  }

  return cur;
}

PoolLayer initPool(int inWidth, int inHeight, int mapSize, int inChannels,
                   int outChannels, int poolType) {
  PoolLayer cur;

  cur = (PoolLayer){inWidth,    inHeight,    mapSize,
                    inChannels, outChannels, poolType};
  cur.basicData = Mat::zeros(1, outChannels, CV_32FC1);
  int W = inWidth / mapSize, H = inHeight / mapSize;

  Mat tmpD = Mat::zeros(H, W, CV_32FC1), tmpI = Mat::zeros(H, W, CV_32SC1);
  for (int i = 0; i < outChannels; ++i) {
    cur.grad.push_back(tmpD.clone());
    cur.y.push_back(tmpD.clone());
    cur.max_pos.push_back(tmpI.clone());
  }

  return cur;
}

DropLayer initDropout(int inWidth, int inHeight, int inChannels,
                      float dropout_rate = 0.25) {
  DropLayer cur;
  cur.inWidth = inWidth;
  cur.inHeight = inHeight;
  cur.inChannels = inChannels;
  cur.dropout_rate = dropout_rate;
  cur.isTraining = true;

  for (int i = 0; i < inChannels; ++i) {
    Mat y_mat = Mat::zeros(inHeight, inWidth, CV_32FC1);
    Mat mask_mat = Mat::ones(inHeight, inWidth, CV_32FC1);

    cur.y.push_back(y_mat);
    cur.mask.push_back(mask_mat);
  }

  return cur;
}

OutLayer initOut(int inNum, int outNum) {
  OutLayer cur;

  cur = (OutLayer){inNum, outNum, true};
  cur.grad = Mat::zeros(1, outNum, CV_32FC1);
  cur.y = Mat::zeros(1, outNum, CV_32FC1);
  cur.v = Mat::zeros(1, outNum, CV_32FC1);
  cur.basicData = Mat::zeros(1, outNum, CV_32FC1);
  cur.wData = Mat::zeros(outNum, inNum, CV_32FC1);

  srand(time(0));
  for (int i = 0; i < outNum; ++i) {
    for (int j = 0; j < inNum; ++j) {
      float coefficient = ((rand() & 1) ? 1 : -1) * (1.0 * rand() / RAND_MAX);
      float base = sqrt(6.0 / (inNum + outNum));
      cur.wData.ptr<float>(i)[j] = base * coefficient;
    }
  }

  return cur;
}

void initCNN(CNN &cnn, int R, int C, int outSize) {
  cnn.layers = 5;

  int mapSize = 5, outChannels = 6, inR = R, inC = C;
  cnn.C1 = initCov(inR, inC, mapSize, 1, outChannels);

  inR = R - mapSize + 1, inC = C - mapSize + 1;
  mapSize = 2;
  cnn.S2 = initPool(inR, inC, mapSize, outChannels, outChannels, MaxPool);

  inR /= mapSize, inC /= mapSize;
  mapSize = 5;
  outChannels = 12;
  cnn.C3 = initCov(inR, inC, mapSize, cnn.S2.outChannels, outChannels);

  inR = inR - mapSize + 1, inC = inC - mapSize + 1;
  mapSize = 2;
  cnn.S4 = initPool(inR, inC, mapSize, outChannels, outChannels, MaxPool);

  inR /= mapSize, inC /= mapSize;
  cnn.D1 = initDropout(inR, inC, outChannels, 0.5);

  cnn.O5 = initOut(inR * inC * outChannels, outSize);

  cnn.eps = Mat::zeros(1, cnn.O5.outNum, CV_32FC1);
}

// calculative functions
Mat Covolute(Mat map, Mat inputData, int type) {
  const int row = map.rows, col = map.cols, row_2 = map.rows / 2,
            col_2 = map.cols / 2, in_row = inputData.rows,
            in_col = inputData.cols;
  Mat exInput;
  copyMakeBorder(inputData, exInput, row_2, row_2, col_2, col_2,
                 cv::BORDER_CONSTANT, 0);
  Mat outputData;
  filter2D(exInput, outputData, exInput.depth(), map);
  if (type == full)
    return outputData;
  else if (type == valid) {
    int out_row = in_row - row + 1, out_col = in_col - col + 1;
    Mat res;
    outputData(cv::Rect(2 * col_2, 2 * row_2, out_col, out_row)).copyTo(res);
    return res;
  } else {
    Mat res;
    outputData(cv::Rect(col_2, row_2, in_col, in_row)).copyTo(res);
    return res;
  }
}

float Relu(float input, float bas) {
  float tmp = input + bas;
  return (tmp > 0 ? tmp : 0);
}

void SoftMax(OutLayer &O) {
  float sum = 0;
  float *tmp_y = O.y.ptr<float>(0);
  float *tmp_v = O.v.ptr<float>(0);
  float *tmp_b = O.basicData.ptr<float>(0);

  for (int i = 0; i < O.outNum; ++i) {
    float Y_i = exp(tmp_v[i] + tmp_b[i]);
    sum += Y_i;
    tmp_y[i] = Y_i;
  }
  for (int i = 0; i < O.outNum; ++i) {
    tmp_y[i] = tmp_y[i] / sum;
  }
}

float MultiVec(Mat vec1, float *vec2) {
  float res = 0;
  for (int i = 0; i < vec1.cols; ++i) {
    res += vec1.ptr<float>(0)[i] * vec2[i];
  }
  return res;
}

// pooling
void AvePooling(Mat input, Mat &output, int mapSize) {
  assert(0);
  int W = input.cols / mapSize, H = input.rows / mapSize;
  float len = 1.0 * mapSize * mapSize;
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      float sum = 0;
      for (int p = i * mapSize; p < (i + 1) * mapSize; ++p) {
        for (int q = j * mapSize; q < (j + 1) * mapSize; ++q) {
          sum += input.ptr<float>(p)[q];
        }
      }
      output.ptr<float>(i)[j] = sum / len;
    }
  }
}

void MaxPooling(Mat input, Mat &max_pos, Mat &output, int mapSize) {
  float len = 1.0 * mapSize * mapSize;
  int W = input.cols / mapSize;
  int H = input.rows / mapSize;
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      float mx = -99999.0;
      int mx_pos = 0;
      for (int p = i * mapSize; p < (i + 1) * mapSize; ++p) {
        for (int q = j * mapSize; q < (j + 1) * mapSize; ++q) {
          if (mx < input.ptr<float>(p)[q]) {
            mx = input.ptr<float>(p)[q];
            mx_pos = p * input.cols + q;
          }
        }
      }
      output.ptr<float>(i)[j] = mx;
      max_pos.ptr<int>(i)[j] = mx_pos;
    }
  }
}

void Dropout_Forward(vector<Mat> inputData, DropLayer &D) {
  if (D.isTraining) {
    float scale = 1.0f / (1.0f - D.dropout_rate);

    for (int i = 0; i < D.inChannels; ++i) {
      randu(D.mask[i], 0.0f, 1.0f);
      threshold(D.mask[i], D.mask[i], D.dropout_rate, 1.0f, THRESH_BINARY);
      multiply(inputData[i], D.mask[i], D.y[i]);
      D.y[i] = D.y[i] * scale;
    }
  } else {
    for (int i = 0; i < D.inChannels; ++i) {
      inputData[i].copyTo(D.y[i]);
    }
  }
}

// forward spreading
void Cov_Forward(vector<Mat> inputData, CovLayer &C, int type) {
  for (int i = 0; i < C.outChannels; ++i) {
    C.v[i] = Mat::zeros(C.v[i].rows, C.v[i].cols, CV_32FC1);

    for (int j = 0; j < C.inChannels; ++j) {
      Mat res = Covolute(C.mapData[j][i], inputData[j], type);
      C.v[i] = C.v[i] + res;
    }
    int outR = C.y[i].rows, outC = C.y[i].cols;
    for (int r = 0; r < outR; ++r) {
      for (int c = 0; c < outC; ++c) {
        C.y[i].ptr<float>(r)[c] =
            Relu(C.v[i].ptr<float>(r)[c], C.basicData.ptr<float>(0)[i]);
      }
    }
  }
}

void Pool_Forward(vector<Mat> inputData, PoolLayer &S, int type) {
  if (type == AvePool) {
    for (int i = 0; i < S.outChannels; ++i) {
      AvePooling(inputData[i], S.y[i], S.mapSize);
    }
  } else if (type == MaxPool) {
    for (int i = 0; i < S.outChannels; ++i) {
      MaxPooling(inputData[i], S.max_pos[i], S.y[i], S.mapSize);
    }
  } else
    cout << "Pool Error\n";
}

void Vec_Forward(Mat input, Mat W, Mat &output) {
  for (int i = 0; i < output.cols; ++i) {
    output.ptr<float>(0)[i] = MultiVec(input, W.ptr<float>(i));
  }
}

void Out_Forward(vector<Mat> inputData, OutLayer &O) {
  Mat inData(1, O.inNum, CV_32FC1);
  float *tmp = inData.ptr<float>(0);
  int R = inputData[0].rows, C = inputData[0].cols;
  int len = inputData.size();
  for (int i = 0; i < len; ++i) {
    for (int r = 0; r < R; ++r) {
      for (int c = 0; c < C; ++c) {
        tmp[i * R * C + r * C + c] = inputData[i].ptr<float>(r)[c];
      }
    }
  }
  Vec_Forward(inData, O.wData, O.v);
  SoftMax(O);
}

void CNN_Forward(CNN &cnn, Mat inputData) {
  vector<Mat> tmp;
  tmp.push_back(inputData);
  Cov_Forward(tmp, cnn.C1, valid);
  Pool_Forward(cnn.C1.y, cnn.S2, MaxPool);
  Cov_Forward(cnn.S2.y, cnn.C3, valid);
  Pool_Forward(cnn.C3.y, cnn.S4, MaxPool);
  Dropout_Forward(cnn.S4.y, cnn.D1);
  Out_Forward(cnn.D1.y, cnn.O5);
}

// backward spreading
float D_Relu(float tmp) {
  if (tmp > 0)
    return 1;
  else
    return 0;
}

void SoftMax_Backward(Mat outputData, Mat &eps, OutLayer &O) {
  for (int i = 0; i < O.outNum; ++i) {
    eps.ptr<float>(0)[i] = O.y.ptr<float>(0)[i] - outputData.ptr<float>(0)[i];
  }
  for (int i = 0; i < O.outNum; ++i) {
    O.grad.ptr<float>(0)[i] = eps.ptr<float>(0)[i];
  }
}

void Pool_Backward(OutLayer O, PoolLayer &S) {
  int R = S.inHeight / S.mapSize, C = S.inWidth / S.mapSize;
  for (int i = 0; i < S.outChannels; ++i) {
    for (int r = 0; r < R; ++r) {
      for (int c = 0; c < C; ++c) {
        int rev_ind = i * R * C + r * C + c;
        for (int j = 0; j < O.outNum; ++j) {
          float g = O.grad.ptr<float>(0)[j];
          S.grad[i].ptr<float>(r)[c] =
              S.grad[i].ptr<float>(r)[c] + g * O.wData.ptr<float>(j)[rev_ind];
        }
      }
    }
  }
}

Mat AveUpSample(Mat mat, int up_row, int up_col) {
  assert(0);
  int C = mat.cols, R = mat.rows;
  Mat res(R * up_row, C * up_col, CV_32FC1);
  float pooling_size = 1.0 / (up_row * up_col);
  for (int i = 0; i < R * up_row; i += up_row) {
    for (int j = 0; j < C * up_col; j += up_col) {
      for (int m = 0; m < up_col; ++m) {
        res.ptr<float>(i)[j + m] =
            mat.ptr<float>(i / up_row)[j / up_col] * pooling_size;
      }
      for (int n = 0; n < up_row; ++n) {
        res.ptr<float>(i + n)[j] = res.ptr<float>(i)[j];
      }
    }
  }
  return res;
}

Mat MaxUpSample(Mat mat, Mat max_pos, int up_col, int up_row) {
  int C = mat.cols, R = mat.rows;
  int outR = R * up_row, outC = C * up_col;
  Mat res = Mat::zeros(outR, outC, CV_32FC1);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      int ind_r = max_pos.ptr<int>(i)[j] / outC;
      int ind_c = max_pos.ptr<int>(i)[j] % outC;
      res.ptr<float>(ind_r)[ind_c] = mat.ptr<float>(i)[j];
    }
  }
  return res;
}

void Cov_Backward(PoolLayer S, CovLayer &C) {
  for (int i = 0; i < C.outChannels; ++i) {
    Mat eps;
    if (S.poolType == AvePool)
      eps = AveUpSample(S.grad[i], S.mapSize, S.mapSize);
    else if (S.poolType == MaxPool)
      eps = MaxUpSample(S.grad[i], S.max_pos[i], S.mapSize, S.mapSize);
    for (int r = 0; r < S.inHeight; ++r) {
      for (int c = 0; c < S.inWidth; ++c) {
        C.grad[i].ptr<float>(r)[c] =
            eps.ptr<float>(r)[c] * D_Relu(C.y[i].ptr<float>(r)[c]);
      }
    }
  }
}

Mat cov(Mat map, Mat inData, int type) {
  Mat flipmap;
  flip(map, flipmap, -1);
  Mat res = Covolute(flipmap, inData, type);
  return res;
}

void CovPool_Backward(CovLayer C, PoolLayer &S, int type) {
  for (int i = 0; i < S.outChannels; ++i) {
    for (int j = 0; j < S.inChannels; ++j) {
      Mat rev = cov(C.mapData[i][j], C.grad[j], type);
      S.grad[i] = S.grad[i] + rev;
    }
  }
}

void Dropout_Backward(DropLayer &D, vector<Mat> &next_grad) {
  if (D.isTraining) {
    float scale = 1.0f / (1.0f - D.dropout_rate);
    for (int i = 0; i < D.inChannels; ++i) {
      multiply(next_grad[i], D.mask[i], next_grad[i]);
      next_grad[i] = next_grad[i] * scale;
    }
  }
}

void CNN_Backward(CNN &cnn, Mat outData) {
  SoftMax_Backward(outData, cnn.eps, cnn.O5);
  Pool_Backward(cnn.O5, cnn.S4);
  Dropout_Backward(cnn.D1, cnn.S4.grad);
  Cov_Backward(cnn.S4, cnn.C3);
  CovPool_Backward(cnn.C3, cnn.S2, full);
  Cov_Backward(cnn.S2, cnn.C1);
}

// update
void Upd_Full(vector<Mat> inData, Opts opts, OutLayer &O) {
  int R = inData[0].rows, C = inData[0].cols;
  Mat Linear(1, R * C * inData.size(), CV_32FC1);
  for (int i = 0; i < inData.size(); ++i) {
    for (int r = 0; r < R; ++r) {
      for (int c = 0; c < C; ++c) {
        Linear.ptr<float>(0)[i * R * C + r * C + c] =
            inData[i].ptr<float>(r)[c];
      }
    }
  }
  for (int j = 0; j < O.outNum; ++j) {
    for (int i = 0; i < O.inNum; ++i) {
      O.wData.ptr<float>(j)[i] = O.wData.ptr<float>(j)[i] -
                                 opts.learning_rate * O.grad.ptr<float>(0)[j] *
                                     Linear.ptr<float>(0)[i];
    }
    O.basicData.ptr<float>(0)[j] = O.basicData.ptr<float>(0)[j] -
                                   opts.learning_rate * O.grad.ptr<float>(0)[j];
  }
}

void Upd_Cov(vector<Mat> inData, Opts opts, CovLayer &C) {
  for (int i = 0; i < C.outChannels; ++i) {
    for (int j = 0; j < C.inChannels; ++j) {
      Mat dlt = Covolute(C.grad[i], inData[j], valid);
      dlt = dlt * (-opts.learning_rate);
      C.mapData[j][i] = C.mapData[j][i] + dlt;
    }
    float d_sum = (float)cv::sum(C.grad[i])[0];
    C.basicData.ptr<float>(0)[i] =
        C.basicData.ptr<float>(0)[i] - opts.learning_rate * d_sum;
  }
}

void CNN_Upd(CNN &cnn, Opts opts, Mat inData) {
  vector<Mat> tmp;
  tmp.push_back(inData);
  Upd_Cov(tmp, opts, cnn.C1);
  Upd_Cov(cnn.S2.y, opts, cnn.C3);
  Upd_Full(cnn.D1.y, opts, cnn.O5);
}

// clear
void Clear_Cov(CovLayer &C) {
  int tmpR = C.grad[0].rows, tmpC = C.grad[0].cols;
  for (int i = 0; i < C.outChannels; ++i) {
    for (int r = 0; r < tmpR; ++r) {
      for (int c = 0; c < tmpC; ++c) {
        C.v[i].ptr<float>(r)[c] = 0.0;
        C.y[i].ptr<float>(r)[c] = 0.0;
        C.grad[i].ptr<float>(r)[c] = 0.0;
      }
    }
  }
}

void Clear_Pool(PoolLayer &S) {
  int tmpR = S.grad[0].rows, tmpC = S.grad[0].cols;
  for (int i = 0; i < S.outChannels; ++i) {
    for (int r = 0; r < tmpR; ++r) {
      for (int c = 0; c < tmpC; ++c) {
        S.y[i].ptr<float>(r)[c] = 0.0;
        S.grad[i].ptr<float>(r)[c] = 0.0;
      }
    }
  }
}

void Clear_Dropout(DropLayer &D) {
  for (int i = 0; i < D.inChannels; ++i) {
    D.y[i] = Mat::zeros(D.y[i].rows, D.y[i].cols, CV_32FC1);
  }
}

void Clear_Out(OutLayer &O) {
  for (int j = 0; j < O.outNum; ++j) {
    O.grad.ptr<float>(0)[j] = 0.0;
    O.v.ptr<float>(0)[j] = 0.0;
    O.y.ptr<float>(0)[j] = 0.0;
  }
}

void CNN_Clear(CNN &cnn) {
  Clear_Cov(cnn.C1);
  Clear_Pool(cnn.S2);
  Clear_Cov(cnn.C3);
  Clear_Pool(cnn.S4);
  Clear_Dropout(cnn.D1);
  Clear_Out(cnn.O5);
}

int Vec_Max(Mat vec) {
  float mx = -1;
  int pos = 0;
  for (int i = 0; i < vec.cols; ++i) {
    if (vec.ptr<float>(0)[i] > mx) {
      mx = vec.ptr<float>(0)[i];
      pos = i;
    }
  }
  return pos;
}

float CNNtest(CNN cnn, vector<Mat> inData, vector<Mat> outData) {
  cnn.D1.isTraining = false;

  int WA = 0;
  for (int i = 0; i < inData.size(); ++i) {
    CNN_Forward(cnn, inData[i]);
    cerr << Vec_Max(cnn.O5.y) << ' ' << Vec_Max(outData[i]);
    if (Vec_Max(cnn.O5.y) != Vec_Max(outData[i])) {
      WA++;
      cerr << "Recognition failed.\n";
    } else
      cerr << "Recognition Succeeded\n";
    CNN_Clear(cnn);
  }
  cerr << "Incorrect Attempts: " << WA << '\n';
  cerr << "Number of Samples: " << inData.size() << '\n';
  return (float)WA / (float)inData.size();
}

// train
void CNNtrain(CNN &cnn, vector<Mat> inData, vector<Mat> outData, Opts opts,
              int trainNum, vector<Mat> test_list, vector<Mat> test_label) {
  cnn.D1.isTraining = true;

  cnn.L = Mat(1, trainNum, CV_32FC1).clone();
  for (int id = 0; id < opts.it_num; ++id) {
    for (int n = 0; n < trainNum; ++n) {
      opts.learning_rate = 0.03 - 0.029 * n / (trainNum - 1);
      CNN_Forward(cnn, inData[n]);
      CNN_Backward(cnn, outData[n]);
      CNN_Upd(cnn, opts, inData[n]);
      float L = 0.0;
      for (int i = 0; i < cnn.O5.outNum; ++i) {
        L = L - outData[n].ptr<float>(0)[i] * log(cnn.O5.y.ptr<float>(0)[i]);
      }
      cnn.L.ptr<float>(0)[n] = L;
      CNN_Clear(cnn);
      cerr << "n=" << n << ",  f=" << cnn.L.ptr<float>(0)[n]
           << ", alpha=" << opts.learning_rate << '\n';
    }
    float failure = CNNtest(cnn, test_list, test_label);
    cout << "success: " << 1.0 - failure << '\n';
  }
}

// read
int Rev_int(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = (i & 0xff);
  ch2 = (i >> 8) & 0xff;
  ch3 = (i >> 16) & 0xff;
  ch4 = (i >> 24) & 0xff;
  return (int)(ch1 << 24) + (int)(ch2 << 16) + (int)(ch3 << 8) + (int)ch4;
}

vector<Mat> Read_Img(const char *filename) {
  FILE *fp = NULL;
  fp = fopen(filename, "rb");
  if (fp == NULL) cout << "read failed\n";
  assert(fp);

  int magic_number = 0;
  int number_of_images = 0;
  int n_rows = 0;
  int n_cols = 0;

  fread(&magic_number, sizeof(int), 1, fp);
  magic_number = Rev_int(magic_number);

  fread(&number_of_images, sizeof(int), 1, fp);
  number_of_images = Rev_int(number_of_images);

  fread(&n_rows, sizeof(int), 1, fp);
  n_rows = Rev_int(n_rows);

  fread(&n_cols, sizeof(int), 1, fp);
  n_cols = Rev_int(n_cols);

  int i, r, c;

  int img_size = n_rows * n_cols;
  vector<Mat> img_list;
  for (i = 0; i < number_of_images; ++i) {
    Mat tmp(n_rows, n_cols, CV_8UC1);
    fread(tmp.data, sizeof(uchar), img_size, fp);
    tmp.convertTo(tmp, CV_32F);
    tmp = tmp / 255.0;
    img_list.push_back(tmp.clone());
  }

  fclose(fp);
  return img_list;
}
vector<Mat> Read_Label(const char *filename) {
  FILE *fp = NULL;
  fp = fopen(filename, "rb");
  if (fp == NULL) {
    cout << "read failed\n";
  }
  assert(fp);
  int magic_number = 0;
  int number_of_labels = 0;
  int label_long = 10;

  fread(&magic_number, sizeof(int), 1, fp);
  magic_number = Rev_int(magic_number);

  fread(&number_of_labels, sizeof(int), 1, fp);
  number_of_labels = Rev_int(number_of_labels);
  int i, l;
  vector<Mat> label_list;
  for (i = 0; i < number_of_labels; ++i) {
    Mat tmp = Mat::zeros(1, label_long, CV_32FC1);
    unsigned char temp = 0;
    fread(&temp, sizeof(unsigned char), 1, fp);
    tmp.ptr<float>(0)[(int)temp] = 1.0;
    label_list.push_back(tmp.clone());
  }
  fclose(fp);
  return label_list;
}

// work
void MINST_CNN_test() {
  vector<Mat> train_list = Read_Img("train-images.idx3-ubyte");
  vector<Mat> train_label = Read_Label("train-labels.idx1-ubyte");
  vector<Mat> test_list = Read_Img("t10k-images.idx3-ubyte");
  vector<Mat> test_label = Read_Label("t10k-labels.idx1-ubyte");
  int train_num = train_list.size(), testnum = test_list.size();
  int R = train_list[0].rows, C = train_list[0].cols;
  int out_num = test_label[0].cols;
  Opts opts;
  opts.learning_rate = 0.03;
  opts.it_num = 20;
  CNN cnn;
  initCNN(cnn, R, C, out_num);
  CNNtrain(cnn, train_list, train_label, opts, train_num, test_list,
           test_label);
  cerr << "!!\n";
}
int main() {
  freopen("result.out", "w", stdout);
  MINST_CNN_test();
  return 0;
}

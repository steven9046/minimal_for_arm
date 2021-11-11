/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
/* for OpenCV */
#include <opencv2/opencv.hpp>
#ifdef ARMNN_DELEGATE
// for armnn delegate
#include <armnn_delegate.hpp>
#endif

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

using namespace tflite;
#define DEFAULT_INPUT_IMAGE           RESOURCE_DIR"/cat_dog.jpg"
#define OUTPUT_NAME_0 "TFLite_Detection_PostProcess"   /* The number of the detected boxes. */
#define OUTPUT_NAME_1 "TFLite_Detection_PostProcess:1"   /* The scores of the detected boxes. */
#define OUTPUT_NAME_2 "TFLite_Detection_PostProcess:2"   /* The categories of the detected boxes. */
#define OUTPUT_NAME_3 "TFLite_Detection_PostProcess:3"   /* The locations of the detected boxes. */
#define INPUT_NAME  "normalized_input_image_tensor"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

class BoundingBox {
public:
    BoundingBox()
        :class_id(0), label(""), score(0), x(0), y(0), w(0), h(0)
    {}

    BoundingBox(int32_t _class_id, std::string _label, float _score, int32_t _x, int32_t _y, int32_t _w, int32_t _h)
        :class_id(_class_id), label(_label), score(_score), x(_x), y(_y), w(_w), h(_h)
    {}

    int32_t     class_id;
    std::string label;
    float       score;
    int32_t     x;
    int32_t     y;
    int32_t     w;
    int32_t     h;
};


int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

#ifdef ARMNN_DELEGATE
  // Create the ArmNN Delegate
  std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
  armnnDelegate::DelegateOptions delegateOptions(backends);
  std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                        theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                         armnnDelegate::TfLiteArmnnDelegateDelete);
  // Modify armnnDelegateInterpreter to use armnnDelegate
  interpreter->ModifyGraphWithDelegate(theArmnnDelegate.get());
#endif


  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  // 按照新的输入张量的大小重新分配内存
  interpreter->AllocateTensors();

  // 输入张量信息
  int input_tensor_id;
  for (auto i : interpreter->inputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == INPUT_NAME) {
      input_tensor_id = i;
      printf("input tensor id : %d\n", input_tensor_id);
      printf("input tensor name : %s\n", tensor->name);
    }
  }

  // 输出张量信息
  float* out_tensor_data_0;
  for (auto i : interpreter->outputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == OUTPUT_NAME_0) {
      printf("out tensor id: %d\n", i );
      printf("out tensor name : %s\n", tensor->name);
      out_tensor_data_0 = interpreter->typed_tensor<float>(i);
    }
  }
  float* out_tensor_data_1;
  for (auto i : interpreter->outputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == OUTPUT_NAME_1) {
      printf("out tensor id: %d\n", i );
      printf("out tensor name : %s\n", tensor->name);
      out_tensor_data_1 = interpreter->typed_tensor<float>(i);
    }
  }
  float* out_tensor_data_2;
  for (auto i : interpreter->outputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == OUTPUT_NAME_2) {
      printf("out tensor id: %d\n", i );
      printf("out tensor name : %s\n", tensor->name);
      out_tensor_data_2 = interpreter->typed_tensor<float>(i);
    }
  }
  float* out_tensor_data_3;
  for (auto i : interpreter->outputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == OUTPUT_NAME_3) {
      printf("out tensor id: %d\n", i );
      printf("out tensor name : %s\n", tensor->name);
      out_tensor_data_3 = interpreter->typed_tensor<float>(i);
    }
  }

  // 加载图片
  cv::Mat origin_img = cv::imread(DEFAULT_INPUT_IMAGE);
  // 调整图片大小和颜色顺序
  cv::Mat img_src = cv::Mat::zeros(300, 300, CV_8UC3);
  cv::resize(origin_img, origin_img, cv::Size(300, 300));
  cv::cvtColor(origin_img, img_src, cv::COLOR_BGR2RGB);
  // 数据复制给input tensor
  uint8_t* dst = interpreter->typed_tensor<uint8_t>(input_tensor_id); 
  uint8_t* src = (uint8_t*)(img_src.data);
  std::copy(src, src + 300*300*3, dst);

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Read output buffers
  // TODO(user): Insert getting data out code.

  int32_t num_det = static_cast<int32_t>(out_tensor_data_3[0]);
  printf("num_det: %d\n", num_det);
  
  float threshold_confidence = 0.4;
    for (int32_t i = 0; i < num_det; i++) {
        if (out_tensor_data_2[i] < threshold_confidence) continue;
        printf("find a target!\n");
        BoundingBox bbox;
        bbox.class_id = static_cast<int32_t>(out_tensor_data_1[i]);
        // bbox.label = label_list_[bbox.class_id];
        // bbox.score = score_raw_list[i];
        bbox.x = static_cast<int32_t>(out_tensor_data_0[i * 4 + 1] * 300);
        bbox.y = static_cast<int32_t>(out_tensor_data_0[i * 4 + 0] * 300);
        bbox.w = static_cast<int32_t>((out_tensor_data_0[i * 4 + 3] - out_tensor_data_0[i * 4 + 1]) * 300);
        bbox.h = static_cast<int32_t>((out_tensor_data_0[i * 4 + 2] - out_tensor_data_0[i * 4 + 0]) * 300);
        cv::rectangle(img_src, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h),cv::Scalar(0, 0, 0), 1);
    }
  cv::namedWindow("result", 0);
  cv::imshow("result", img_src);
  cv::waitKey(0);
  return 0;
}

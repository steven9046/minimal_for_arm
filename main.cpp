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
// for armnn delegate
#include <armnn_delegate.hpp>

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
#define INPUT_DIMS  { 1, 300, 300, 3 }
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

/**
 * @param   org                 输入图像
 * @param   dst                 输出图像    这个是根据tensor info 来设置的
 * @param   crop_x              裁剪的左上角 0 , 0
 * @param   crop_w              裁剪的长宽   960,717
 * @param   is_rgb              默认值true
 * @param   crop_type           kCropTypeExpand
 * @param   resize_by_linear    默认true  这个就是选择什么样的resize方法而已
 * @brief   1. resize
 *          2. crop
 *          这里是做了一个保持长款比的缩放
 */
void CropResizeCvt(const cv::Mat& org, cv::Mat& dst, int32_t& crop_x, int32_t& crop_y, int32_t& crop_w, int32_t& crop_h) {
  cv::Mat src = org(cv::Rect(crop_x, crop_y, crop_w, crop_h));
  printf("crop_x: %d\n", crop_x);
  printf("crop_y: %d\n", crop_y);
  printf("crop_w: %d\n", crop_w);
  printf("crop_h: %d\n", crop_h);

  printf("dst height: %d\n", dst.rows);
  printf("dst width : %d\n", dst.cols);

  float aspect_ratio_src = static_cast<float>(src.cols) / src.rows;
  float aspect_ratio_dst = static_cast<float>(dst.cols) / dst.rows;
  printf("aspect_ratio_src: %f\n", aspect_ratio_src);
  printf("aspect_ratio_dst: %f\n", aspect_ratio_dst);

  cv::Rect target_rect(0, 0, dst.cols, dst.rows);
  if (aspect_ratio_src > aspect_ratio_dst) {
    target_rect.height = static_cast<int32_t>(target_rect.width / aspect_ratio_src);
    target_rect.y = (dst.rows - target_rect.height) / 2;
  } else {
    target_rect.width = static_cast<int32_t>(target_rect.height * aspect_ratio_src);
    target_rect.x = (dst.cols - target_rect.width) / 2;
  }
  cv::Mat target = dst(target_rect);
  printf("target x: %d\n", target_rect.x);
  printf("target y: %d\n", target_rect.y);
  printf("target width: %d\n", target_rect.width);
  printf("target height: %d\n", target_rect.height);

  cv::resize(src, target, target.size(), 0, 0, cv::INTER_LINEAR);
  crop_x -= target_rect.x * crop_w / target_rect.width;
  crop_y -= target_rect.y * crop_h / target_rect.height;
  crop_w = dst.cols * crop_w / target_rect.width;
  crop_h = dst.rows * crop_h / target_rect.height;
  printf("crop_x: %d\n", crop_x);
  printf("crop_y: %d\n", crop_y);
  printf("crop_w: %d\n", crop_w);
  printf("crop_h: %d\n", crop_h);
  printf("converting bgr to rgb!\n");
  cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
}

void PreProcessImage(const cv::Mat input_img, uint8_t* dst)
{
  const int32_t img_width = input_img.cols;
  const int32_t img_height = input_img.rows;
  const int32_t img_channel = input_img.channels();
  uint8_t* src = (uint8_t*)(input_img.data);
  std::copy(src, src + 300*300*3, dst);
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

  DisplayModelInfo(*interpreter_);
  printf("=== Pre-delegate Interpreter State ===\n");

    // Create the ArmNN Delegate
  std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    // std::string backends = "GpuAcc";
  armnnDelegate::DelegateOptions delegateOptions(backends);
  std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                        theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                         armnnDelegate::TfLiteArmnnDelegateDelete);
  // Modify armnnDelegateInterpreter to use armnnDelegate
  interpreter->ModifyGraphWithDelegate(theArmnnDelegate.get());


  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
  // kInputIndex 是输入张量索引，kNum 是输入图片张数，即 batch size。
  int kInputIndex = 0;
  int kNum = 1;
  int kInputHeight = 300;
  int kInputWidth = 300;
  int kInputChannels = 3;
  interpreter->ResizeInputTensor(kInputIndex, {kNum, kInputHeight, kInputWidth, kInputChannels});

  // 按照新的输入张量的大小重新分配内存
  interpreter->AllocateTensors();

  // 输入张量信息
  int input_tensor_id;
  for (auto i : interpreter->inputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == INPUT_NAME) {
      input_tensor_id = i;
      printf("input tensor id: %d\n", input_tensor_id);
    }
  }
  // 输出张量信息
  float* out_tensor_data_0;
  for (auto i : interpreter->outputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == OUTPUT_NAME_0) {
      printf("name 0 out number: %d\n", i );
      std::cout << "name: " << std::string(tensor->name) << std::endl;
      out_tensor_data_0 = interpreter->typed_tensor<float>(i);
    }
  }
  float* out_tensor_data_1;
  for (auto i : interpreter->outputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == OUTPUT_NAME_1) {
      printf("name 1 out number: %d\n", i );
      std::cout << "name: " << std::string(tensor->name) << std::endl;
      out_tensor_data_1 = interpreter->typed_tensor<float>(i);
    }
  }
  float* out_tensor_data_2;
  for (auto i : interpreter->outputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == OUTPUT_NAME_2) {
      printf("name 2 out number: %d\n", i );
      std::cout << "name: " << std::string(tensor->name) << std::endl;
      out_tensor_data_2 = interpreter->typed_tensor<float>(i);
    }
  }
  float* out_tensor_data_3;
  for (auto i : interpreter->outputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == OUTPUT_NAME_3) {
      printf("name 3 out number: %d\n", i );
      std::cout << "name: " << std::string(tensor->name) << std::endl;
      out_tensor_data_3 = interpreter->typed_tensor<float>(i);
    }
  }

  // // 循环填充输入张量的内存，其中 kInputIndex 是输入张量索引。
  uint8_t *input_buffer = interpreter->typed_tensor<uint8_t>(kInputIndex);  //这个就是一个指针，数组，存放输入的tensor,所以索引 0 就是第一个，我们这里只输入一张图片，所以只用第一个
  const int kInputBytes = sizeof(uint8_t) * kInputWidth * kInputHeight * kInputChannels;
  cv::Size input_buffer_size(kInputWidth, kInputHeight);
  int buffer_index = 0;
  cv::Mat origin_img = cv::imread(DEFAULT_INPUT_IMAGE);
  int crop_w = origin_img.cols;
  int crop_h = origin_img.rows;
  int crop_x = 0;
  int crop_y = 0;
  cv::Mat img_src = cv::Mat::zeros(300, 300, CV_8UC3);
  CropResizeCvt(origin_img, img_src, crop_x, crop_y, crop_w, crop_h);
  cv::Mat input_image;
  // 输入预处理操作。
  cv::resize(origin_img, input_image, input_buffer_size);
  cv::cvtColor(input_image,input_image,cv::COLOR_BGR2RGB);
  cv::namedWindow("orig_img", 0);
  cv::imshow("orig_img", input_image);
  cv::waitKey(0);

    // /* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
    // float mean[3];// 0
    // float norm[3];// 1
    // mean[0] = 0.0f;
    // mean[1] = 0.0f;
    // mean[2] = 0.0f;
    // norm[0] = 1.0f / 255.0f;
    // norm[1] = 1.0f / 255.0f;
    // norm[2] = 1.0f / 255.0f;
    // for (int32_t i = 0; i < 3; i++) {
    //     mean[i] *= 255.0f;
    //     norm[i] *= 255.0f;
    //     norm[i] = 1.0f / norm[i];
    // }
  
  uint8_t* dst = interpreter->typed_tensor<uint8_t>(input_tensor_id); 
  PreProcessImage(input_image, dst);
  // cv::cvtColor(input_image, input_image, cv::COLOR_BGR2GRAY);
  // //   input_image.convertTo(input_image, CV_32F, 2.f / 255, -0.5);
  //   // 填充输入张量的内存，batch size  > 1 时，注意
  //   // input_buffer 的数据类型需要强制转换。因为 buffer_index 是按 byte 为单位进行地址偏移的。
  // memcpy(input_buffer, input_image.data, kInputBytes);
  // }


  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.

  int32_t num_det = static_cast<int32_t>(out_tensor_data_3[0]);
  // int32_t num_det2 = static_cast<int32_t>(out_tensor_data_3[1]);
  // int32_t num_det3 = static_cast<int32_t>(out_tensor_data_3[2]);
  // int32_t num_det4 = static_cast<int32_t>(out_tensor_data_3[3]);
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
        cv::rectangle(input_image, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h),cv::Scalar(0, 0, 0), 1);
        printf("data 1 (label): %f\n", out_tensor_data_1[i]);
        printf("data 2 (score): %f\n", out_tensor_data_2[i]);
        printf("data 3 (bbox): %f\n", out_tensor_data_0[i * 4 + 0]);
        printf("data 3 (bbox): %f\n", out_tensor_data_0[i * 4 + 1]);
        printf("data 3 (bbox): %f\n", out_tensor_data_0[i * 4 + 2]);
        printf("data 3 (bbox): %f\n", out_tensor_data_0[i * 4 + 3]);
    }
  cv::namedWindow("result", 0);
  cv::imshow("result", input_image);
  cv::waitKey(0);
  return 0;
}

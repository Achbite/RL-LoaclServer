#include "ai/onnx_inferencer.h"
#include "log/logger.h"

#include <algorithm>

// ---- 构造函数 ----
OnnxInferencer::OnnxInferencer()
    : env_(ORT_LOGGING_LEVEL_WARNING, "MazeInferencer") {
    // 单线程推理即可（每次推理 batch=1）
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
}

// ---- 加载 ONNX 模型（线程安全）----
bool OnnxInferencer::LoadModel(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(load_mutex_);

    try {
        // 创建新 Session（加载失败会抛异常，旧 Session 不受影响）
        auto new_session = std::make_shared<Ort::Session>(
            env_, model_path.c_str(), session_options_);

        // 原子替换：使用 atomic_store 保证与 Infer() 端 atomic_load 的线程安全
        std::atomic_store(&session_, new_session);
        current_model_path_ = model_path;
        loaded_.store(true);

        LOG_INFO("OnnxInferencer", "模型加载成功: %s", model_path.c_str());
        return true;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("OnnxInferencer", "模型加载失败: %s, 错误: %s",
                  model_path.c_str(), e.what());
        return false;
    }
}

// ---- 推理（线程安全，无锁读取）----
bool OnnxInferencer::Infer(const std::vector<float>& obs, int obs_dim,
                           std::vector<float>& action_probs, float& value) {
    // 原子读取 shared_ptr（与 LoadModel 端 atomic_store 配合，保证线程安全）
    auto session = std::atomic_load(&session_);
    if (!session) {
        return false;
    }

    try {
        // ---- 构建输入 Tensor ----
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(obs_dim)};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            const_cast<float*>(obs.data()),
            obs.size(),
            input_shape.data(),
            input_shape.size());

        // ---- 执行推理 ----
        const char* input_names[] = {INPUT_NAME};
        const char* output_names[] = {OUTPUT_ACTION_PROBS, OUTPUT_VALUE};

        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 2);

        // ---- 解析输出：action_probs [1, action_dim] ----
        float* probs_data = outputs[0].GetTensorMutableData<float>();
        auto probs_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int action_dim = static_cast<int>(probs_shape[1]);

        action_probs.assign(probs_data, probs_data + action_dim);

        // ---- 解析输出：value [1, 1] ----
        float* value_data = outputs[1].GetTensorMutableData<float>();
        value = value_data[0];

        return true;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("OnnxInferencer", "推理失败: %s", e.what());
        return false;
    }
}

// ---- 是否已加载模型 ----
bool OnnxInferencer::IsLoaded() const {
    return loaded_.load();
}

// ---- 获取当前模型路径 ----
std::string OnnxInferencer::GetModelPath() const {
    std::lock_guard<std::mutex> lock(load_mutex_);
    return current_model_path_;
}

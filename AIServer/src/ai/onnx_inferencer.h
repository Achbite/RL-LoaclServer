#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

// ---- ONNX 推理封装（线程安全，支持模型热更新）----
// 使用 shared_ptr<Ort::Session> + std::atomic_load/store 实现真正的原子读写。
// LoadModel() 通过 atomic_store 更新 session_，Infer() 通过 atomic_load 读取。
// 多个并行 Episode 可同时调用 Infer()，不需要外部加锁。

class OnnxInferencer {
public:
    OnnxInferencer();
    ~OnnxInferencer() = default;

    // 加载 ONNX 模型（线程安全，内部互斥）
    // 返回 true 表示加载成功，false 表示加载失败（保留旧模型）
    bool LoadModel(const std::string& model_path);

    // 推理：输入 obs 向量，输出动作概率和状态价值
    // 线程安全，多线程可同时调用
    bool Infer(const std::vector<float>& obs, int obs_dim,
               std::vector<float>& action_probs, float& value);

    // 是否已加载模型
    bool IsLoaded() const;

    // 获取当前加载的模型路径
    std::string GetModelPath() const;

private:
    Ort::Env env_;                                  // ONNX Runtime 环境（全局唯一）
    Ort::SessionOptions session_options_;            // 会话选项

    std::shared_ptr<Ort::Session> session_;          // 当前推理会话（通过 atomic_load/store 保证线程安全）
    mutable std::mutex load_mutex_;                  // 模型加载互斥锁
    std::atomic<bool> loaded_{false};                // 是否已加载模型
    std::string current_model_path_;                 // 当前模型路径

    // 输入输出名称（与 Learner 端 ONNX 导出对齐）
    static constexpr const char* INPUT_NAME = "obs";
    static constexpr const char* OUTPUT_ACTION_PROBS = "action_probs";
    static constexpr const char* OUTPUT_VALUE = "value";
};

#pragma once

#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <mutex>

// ---- 日志级别 ----
enum class LogLevel {
    DEBUG = 0,      // 详细调试信息（Client 默认不输出）
    INFO  = 1,      // 常规信息
    WARN  = 2,      // 警告
    ERROR = 3,      // 错误
};

// ---- 线程安全日志器（Client 版：仅控制台输出）----
// 单例模式，只打印执行效果（接收到的指令 + 执行结果）
class Logger {
public:
    // 获取单例
    static Logger& Instance() {
        static Logger instance;
        return instance;
    }

    // ---- 设置控制台最低输出级别 ----
    void SetConsoleLevel(LogLevel level) {
        console_level_ = level;
    }

    // ---- 日志输出（printf 风格）----
    void Log(LogLevel level, const char* tag, const char* fmt, ...) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (level < console_level_) return;

        // 格式化用户消息
        char msg_buf[2048];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
        va_end(args);

        const char* level_str = LevelToStr(level);
        std::printf("[%s][%s] %s\n", level_str, tag, msg_buf);
        std::fflush(stdout);
    }

    ~Logger() = default;

private:
    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // ---- 日志级别转字符串 ----
    static const char* LevelToStr(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERROR: return "ERROR";
            default:              return "?????";
        }
    }

    std::mutex mutex_;
    LogLevel console_level_ = LogLevel::INFO;   // 默认只输出 INFO 及以上
};

// ---- 便捷宏（自动填充 tag）----
#define LOG_DEBUG(tag, fmt, ...) Logger::Instance().Log(LogLevel::DEBUG, tag, fmt, ##__VA_ARGS__)
#define LOG_INFO(tag, fmt, ...)  Logger::Instance().Log(LogLevel::INFO,  tag, fmt, ##__VA_ARGS__)
#define LOG_WARN(tag, fmt, ...)  Logger::Instance().Log(LogLevel::WARN,  tag, fmt, ##__VA_ARGS__)
#define LOG_ERROR(tag, fmt, ...) Logger::Instance().Log(LogLevel::ERROR, tag, fmt, ##__VA_ARGS__)

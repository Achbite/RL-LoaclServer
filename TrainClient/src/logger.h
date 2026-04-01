#pragma once

#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <string>
#include <mutex>
#include <sys/stat.h>

// ---- 日志级别 ----
enum class LogLevel {
    DEBUG = 0,      // 详细调试信息（默认仅写文件）
    INFO  = 1,      // 常规信息
    WARN  = 2,      // 警告
    ERROR = 3,      // 错误
};

// ---- 线程安全日志器（Client 版：控制台 + 文件双输出）----
// 单例模式，支持将帧级日志写入 log/ 目录下的文件
class Logger {
public:
    // 获取单例
    static Logger& Instance() {
        static Logger instance;
        return instance;
    }

    // ---- 初始化（创建 log 目录 + 打开日志文件）----
    bool Init(const std::string& log_dir = "log") {
        std::lock_guard<std::mutex> lock(mutex_);

        // 创建日志目录
        mkdir(log_dir.c_str(), 0755);

        // 生成日志文件名：client_YYYYMMDD_HHMMSS.log
        std::time_t now = std::time(nullptr);
        std::tm* tm = std::localtime(&now);
        char filename[256];
        std::snprintf(filename, sizeof(filename),
                      "%s/client_%04d%02d%02d_%02d%02d%02d.log",
                      log_dir.c_str(),
                      tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                      tm->tm_hour, tm->tm_min, tm->tm_sec);

        log_file_ = std::fopen(filename, "w");
        if (!log_file_) {
            std::printf("[Logger] 无法创建日志文件: %s\n", filename);
            return false;
        }

        log_path_ = filename;
        initialized_ = true;
        std::printf("[Logger] 日志文件: %s\n", filename);
        return true;
    }

    // ---- 设置控制台最低输出级别 ----
    void SetConsoleLevel(LogLevel level) {
        console_level_ = level;
    }

    // ---- 设置文件最低输出级别 ----
    void SetFileLevel(LogLevel level) {
        file_level_ = level;
    }

    // ---- 日志输出（printf 风格）----
    void Log(LogLevel level, const char* tag, const char* fmt, ...) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 格式化时间戳
        char time_buf[32];
        FormatTime(time_buf, sizeof(time_buf));

        // 格式化用户消息
        char msg_buf[2048];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
        va_end(args);

        const char* level_str = LevelToStr(level);

        // 写入控制台（受级别过滤）
        if (level >= console_level_) {
            std::printf("[%s][%s] %s\n", level_str, tag, msg_buf);
            std::fflush(stdout);
        }

        // 写入文件（受级别过滤）
        if (initialized_ && log_file_ && level >= file_level_) {
            std::fprintf(log_file_, "[%s][%s][%s] %s\n",
                         time_buf, level_str, tag, msg_buf);
            std::fflush(log_file_);
        }
    }

    // ---- 仅写文件（不输出到控制台，用于高频帧级日志）----
    void FileOnly(LogLevel level, const char* tag, const char* fmt, ...) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!initialized_ || !log_file_ || level < file_level_) return;

        char time_buf[32];
        FormatTime(time_buf, sizeof(time_buf));

        char msg_buf[2048];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
        va_end(args);

        std::fprintf(log_file_, "[%s][%s][%s] %s\n",
                     time_buf, LevelToStr(level), tag, msg_buf);
        std::fflush(log_file_);
    }

    // ---- 关闭日志文件 ----
    void Close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (log_file_) {
            std::fclose(log_file_);
            log_file_ = nullptr;
        }
        initialized_ = false;
    }

    ~Logger() { Close(); }

private:
    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // ---- 格式化当前时间 ----
    static void FormatTime(char* buf, size_t buf_size) {
        std::time_t now = std::time(nullptr);
        std::tm* tm = std::localtime(&now);
        std::snprintf(buf, buf_size, "%04d-%02d-%02d %02d:%02d:%02d",
                      tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                      tm->tm_hour, tm->tm_min, tm->tm_sec);
    }

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
    FILE* log_file_       = nullptr;
    bool  initialized_    = false;
    std::string log_path_;
    LogLevel console_level_ = LogLevel::INFO;   // 控制台默认只输出 INFO 及以上
    LogLevel file_level_    = LogLevel::DEBUG;   // 文件默认记录所有级别
};

// ---- 便捷宏（自动填充 tag）----
#define LOG_DEBUG(tag, fmt, ...) Logger::Instance().Log(LogLevel::DEBUG, tag, fmt, ##__VA_ARGS__)
#define LOG_INFO(tag, fmt, ...)  Logger::Instance().Log(LogLevel::INFO,  tag, fmt, ##__VA_ARGS__)
#define LOG_WARN(tag, fmt, ...)  Logger::Instance().Log(LogLevel::WARN,  tag, fmt, ##__VA_ARGS__)
#define LOG_ERROR(tag, fmt, ...) Logger::Instance().Log(LogLevel::ERROR, tag, fmt, ##__VA_ARGS__)

// 仅写文件（高频帧级日志专用）
#define LOG_FILE(tag, fmt, ...) Logger::Instance().FileOnly(LogLevel::DEBUG, tag, fmt, ##__VA_ARGS__)

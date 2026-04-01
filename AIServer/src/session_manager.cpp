#include "session_manager.h"
#include "logger.h"

// ---- 创建新会话 ----
int SessionManager::CreateSession() {
    std::lock_guard<std::mutex> lock(mutex_);

    int sid = next_session_id_++;
    sessions_[sid] = Session{};
    sessions_[sid].session_id = sid;

    LOG_INFO("SessionManager", "创建会话 session_id=%d, 活跃会话数=%zu",
             sid, sessions_.size());

    return sid;
}

// ---- 获取指定会话 ----
SessionManager::Session* SessionManager::GetSession(int session_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return nullptr;
    }
    return &it->second;
}

// ---- 获取或创建会话（session_id=0 时使用默认会话）----
SessionManager::Session* SessionManager::GetOrCreateSession(int session_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        return &it->second;
    }

    // 不存在则创建
    sessions_[session_id] = Session{};
    sessions_[session_id].session_id = session_id;

    LOG_INFO("SessionManager", "自动创建会话 session_id=%d, 活跃会话数=%zu",
             session_id, sessions_.size());

    return &sessions_[session_id];
}

// ---- 销毁指定会话 ----
void SessionManager::DestroySession(int session_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        sessions_.erase(it);
        LOG_INFO("SessionManager", "销毁会话 session_id=%d, 剩余会话数=%zu",
                 session_id, sessions_.size());
    }
}

// ---- 获取当前活跃会话数 ----
int SessionManager::GetActiveSessionCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(sessions_.size());
}

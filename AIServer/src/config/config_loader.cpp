#include "config/config_loader.h"
#include "log/logger.h"

#include <fstream>
#include <sstream>
#include <vector>

// ---- 去除字符串首尾空白 ----
static std::string Trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end   = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

// ---- 去除字符串值的引号包裹 ----
static std::string StripQuotes(const std::string& s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') ||
         (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

// ---- 安全转换辅助 ----
static int SafeInt(const std::string& val, int def) {
    if (val.empty()) return def;
    try { return std::stoi(val); } catch (...) { return def; }
}

// ---- YAML 键值对 ----
struct YamlEntry {
    std::string section;
    std::string key;
    std::string value;
};

// ---- 解析 YAML 文本为键值对列表 ----
static std::vector<YamlEntry> ParseYaml(const std::string& text) {
    std::vector<YamlEntry> entries;
    std::istringstream stream(text);
    std::string line;
    std::string current_section;

    while (std::getline(stream, line)) {
        // 去除行内注释
        size_t comment_pos = std::string::npos;
        bool in_quotes = false;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '"' || line[i] == '\'') {
                in_quotes = !in_quotes;
            } else if (line[i] == '#' && !in_quotes) {
                comment_pos = i;
                break;
            }
        }
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        std::string trimmed = Trim(line);
        if (trimmed.empty()) continue;

        size_t colon_pos = trimmed.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key_part = Trim(trimmed.substr(0, colon_pos));
        std::string val_part = Trim(trimmed.substr(colon_pos + 1));

        bool has_indent = (!line.empty() && (line[0] == ' ' || line[0] == '\t'));

        if (!has_indent && val_part.empty()) {
            current_section = key_part;
        } else if (has_indent && !key_part.empty()) {
            YamlEntry entry;
            entry.section = current_section;
            entry.key     = key_part;
            entry.value   = StripQuotes(val_part);
            entries.push_back(entry);
        }
    }

    return entries;
}

// ---- 在解析结果中查找指定 section.key 的值 ----
static std::string FindValue(const std::vector<YamlEntry>& entries,
                             const std::string& section,
                             const std::string& key) {
    for (const auto& e : entries) {
        if (e.section == section && e.key == key) {
            return e.value;
        }
    }
    return "";
}

// ---- 从 YAML 文件加载配置 ----
bool LoadServerConfig(const std::string& yaml_path, AIServerConfig& out_config) {
    std::ifstream ifs(yaml_path);
    if (!ifs.is_open()) {
        LOG_WARN("Config", "无法打开配置文件: %s，使用默认值", yaml_path.c_str());
        out_config = AIServerConfig{};
        return false;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();
    ifs.close();

    LOG_INFO("Config", "加载配置文件: %s", yaml_path.c_str());

    std::vector<YamlEntry> entries = ParseYaml(content);

    // --- server ---
    out_config.server.listen_port = SafeInt(FindValue(entries, "server", "listen_port"), 9002);
    out_config.server.max_agents  = SafeInt(FindValue(entries, "server", "max_agents"),  10);
    out_config.server.run_mode    = SafeInt(FindValue(entries, "server", "run_mode"),    3);

    // --- strategy ---
    out_config.strategy.grid_size        = SafeInt(FindValue(entries, "strategy", "grid_size"),        500);
    out_config.strategy.replan_interval  = SafeInt(FindValue(entries, "strategy", "replan_interval"),  10);

    // --- model ---
    std::string local_dir = FindValue(entries, "model", "local_dir");
    if (!local_dir.empty()) {
        out_config.model.local_dir = local_dir;
    }
    std::string p2p_dir = FindValue(entries, "model", "p2p_dir");
    if (!p2p_dir.empty()) {
        out_config.model.p2p_dir = p2p_dir;
    }
    out_config.model.poll_interval = SafeInt(FindValue(entries, "model", "poll_interval"), 10);
    std::string save_name = FindValue(entries, "model", "save_name");
    if (!save_name.empty()) {
        out_config.model.save_name = save_name;
    }

    // --- learner ---
    std::string lhost = FindValue(entries, "learner", "host");
    if (!lhost.empty()) {
        out_config.learner.host = lhost;
    }
    out_config.learner.port              = SafeInt(FindValue(entries, "learner", "port"),              9003);
    out_config.learner.send_interval     = SafeInt(FindValue(entries, "learner", "send_interval"),     32);
    out_config.learner.sample_batch_size = SafeInt(FindValue(entries, "learner", "sample_batch_size"), 128);
    out_config.learner.send_timeout      = SafeInt(FindValue(entries, "learner", "send_timeout"),      10);
    out_config.learner.max_retries       = SafeInt(FindValue(entries, "learner", "max_retries"),       3);

    // --- 运行模式名称映射 ---
    const char* mode_names[] = {"未知", "训练", "推理", "A*测试"};
    int mode_idx = out_config.server.run_mode;
    const char* mode_name = (mode_idx >= 1 && mode_idx <= 3) ? mode_names[mode_idx] : mode_names[0];

    LOG_INFO("Config", "server: port=%d, max_agents=%d, run_mode=%d(%s)",
             out_config.server.listen_port, out_config.server.max_agents,
             out_config.server.run_mode, mode_name);
    LOG_INFO("Config", "strategy: grid=%d, replan=%d",
             out_config.strategy.grid_size,
             out_config.strategy.replan_interval);
    LOG_INFO("Config", "model: local=%s, p2p=%s, poll=%ds",
             out_config.model.local_dir.c_str(),
             out_config.model.p2p_dir.c_str(),
             out_config.model.poll_interval);
    LOG_INFO("Config", "learner: %s:%d, interval=%d, batch_size=%d, timeout=%ds, retries=%d",
             out_config.learner.host.c_str(),
             out_config.learner.port,
             out_config.learner.send_interval,
             out_config.learner.sample_batch_size,
             out_config.learner.send_timeout,
             out_config.learner.max_retries);

    return true;
}

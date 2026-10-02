"""全局配置：DeepSeek API / 本地模型均可。

用法：
  - 官方 API:  设置环境变量 DEEPSEEK_API_KEY 即可（默认 base_url 为官方地址）
  - 本地模型:  设置 DEEPSEEK_BASE_URL 指向本地 OpenAI 兼容服务
    例:
      export DEEPSEEK_BASE_URL=http://localhost:11434/v1   # Ollama
      export DEEPSEEK_BASE_URL=http://localhost:1234/v1    # LM Studio
      export DEEPSEEK_API_KEY=local  # 本地服务通常不校验 key，随便填
"""
import os


def _load_dotenv(path: str) -> int:
    """极简 .env 读取（零依赖，不引入 python-dotenv）。

    只支持 KEY=VALUE、# 注释、值两端可选的引号。已存在的环境变量优先
    （用 setdefault），所以命令行 export 的值不会被 .env 顶掉。

    返回实际写入的条目数，便于自检/排查「为什么我的 .env 没生效」。
    """
    if not os.path.isfile(path):
        return 0
    n = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]           # 去掉成对的引号
                if key and key not in os.environ:
                    os.environ[key] = val
                    n += 1
    except OSError:
        return 0                              # 配置文件坏了不该让程序起不来
    return n


# 必须在下面所有 os.getenv 之前执行
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# 注意：旧名 deepseek-chat / deepseek-reasoner 已于 2026-07-24 被官方弃用，
# 二者分别对应 deepseek-v4-flash 的非思考 / 思考模式，故默认改用 V4 名。
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 校对打回重写的最大轮数（防止无限循环）
MAX_REVISION_ROUNDS = 3

# 是否让模型进入「思考模式」。
# DeepSeek V4 默认开启思考（默认 high 强度），对写作类任务收益有限，
# 但耗时会变成数倍，且容易把输出额度全耗在思考里、导致正式内容为空 → 默认关闭。
ENABLE_THINKING = os.getenv("DEEPSEEK_THINKING", "0") == "1"

# 单次调用最大输出 token。
# 必须显式设置：JSON 模式下若模型没被正确引导，会输出「无尽空白」直到上限，
# 表现为前端长时间卡住；设上限可让它快速失败而不是拖死整个请求。
MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192"))

# 「空闲超时」（秒）——注意语义，不是请求总时长。
#
# 所有请求都走流式（stream=True），该值表示「相邻两段数据之间」最长能等多久：
# 只要模型还在持续吐字，请求就不会被判超时；只有连续这么长时间收不到任何数据
# （含思考型模型的 thinking 分片）才中断。
#
# 为什么必须这样：非流式请求要等服务端把全文生成完才一次性返回，
# 写一章 3000 字动辄两三分钟，用"总时长超时"必然误杀。
LLM_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "90"))

# 失败重试次数。默认 0：对流式超时来说，自动重试只会让用户白等一倍时间，
# 而且大概率再次失败。宁可立刻报错并说明原因，让用户自己决定重试时机。
LLM_MAX_RETRIES = int(os.getenv("DEEPSEEK_MAX_RETRIES", "0"))

# 校对（Reviewer）输出彻底无法解析时的策略：
#   1（默认）= 放行：跳过校对直接润色，并在界面醒目告警，保证已写好的正文不白费
#   0        = 严格：直接中断，宁可失败也不让未经校验的章节过关
# 之所以默认放行：校对只是质检环节，模型的输出格式问题不该毁掉一整章创作。
REVIEW_FAIL_OPEN = os.getenv("STORY_REVIEW_FAIL_OPEN", "1") == "1"

# 生成参数
TEMPERATURE_PLANNER = 0.7
TEMPERATURE_WRITER = 0.85
TEMPERATURE_REVIEWER = 0.2   # 校对要稳，低温
TEMPERATURE_POLISHER = 0.6

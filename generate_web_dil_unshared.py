import argparse
import html
import os
import socket
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import torch
import torch.nn.functional as F

# ============================================================
# 重要：
#
# 请确保你的完整训练/模型代码文件名为：
#
#     convtransformer_dil_unshared.py
#
# 并且它和本文件 generate_web_dil_unshared.py 在同一个目录。
#
# 这里复用原文件里的：
# - load_checkpoint
# - decode_ids
#
# 如果你的原文件名不是 convtransformer_dil_unshared.py，
# 请把下面这一行改成对应文件名。
# ============================================================

from convtransformer_dil_unshared import load_checkpoint, decode_ids


# ============================================================
# 1. 文本规范化
# ============================================================

def normalize_input_text(text: str) -> str:
    """
    规范化网页输入文本。

    主要处理：
    - Windows/浏览器换行 \\r\\n -> \\n
    - 单独的 \\r -> \\n
    - BOM \\ufeff 删除
    - 零宽空格 \\u200b 删除
    - 不间断空格 \\xa0 -> 普通空格

    注意：
    不在这里偷偷删除所有 OOV 字符。
    如果仍有词表外字符，应该明确报错，方便用户知道是哪一个字符。
    """

    if text is None:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\xa0", " ")

    return text


# ============================================================
# 2. OOV 检查
# ============================================================

def find_oov_characters(text: str, stoi) -> list:
    """
    找出输入文本中不在词表里的字符。

    返回：
    [
        {
            "char": ch,
            "repr": repr(ch),
            "ord": ord(ch),
            "pos": i
        },
        ...
    ]
    """

    result = []
    seen = set()

    for i, ch in enumerate(text):
        if ch not in stoi:
            key = ch
            if key not in seen:
                seen.add(key)
                result.append(
                    {
                        "char": ch,
                        "repr": repr(ch),
                        "ord": ord(ch),
                        "pos": i
                    }
                )

    return result


def format_oov_error(oov_list: list) -> str:
    """
    格式化词表外字符错误信息。
    """

    if not oov_list:
        return ""

    lines = []
    lines.append("输入文本包含词表外字符。")
    lines.append("")
    lines.append("前若干个词表外字符如下：")

    for item in oov_list[:20]:
        lines.append(
            f"- 字符 {item['repr']}，Unicode/ord={item['ord']}，首次位置={item['pos']}"
        )

    if len(oov_list) > 20:
        lines.append("")
        lines.append(f"还有 {len(oov_list) - 20} 个不同的词表外字符未显示。")

    lines.append("")
    lines.append("建议：")
    lines.append("1. 检查输入是否复制了特殊符号。")
    lines.append("2. 如果是网页换行导致的 \\r，本程序已经自动转换。")
    lines.append("3. 如果是特殊标点或 emoji，需要替换成训练集中出现过的字符。")
    lines.append("4. 不建议直接无声删除所有 OOV 字符，因为会改变原文。")

    return "\n".join(lines)


# ============================================================
# 3. 纯生成函数：不再右侧 padding 到 block_size
# ============================================================

@torch.no_grad()
def generate_without_right_padding(
    model,
    start_text: str,
    stoi,
    itos,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device
) -> str:
    """
    网页生成函数。

    当前版本针对 dil_unshared 模型的网页推理：

    - 不再把输入右侧 padding 到训练 block_size；
    - 保持原始输入长度；
    - 如果上下文长度超过 block_size，只截取最后 block_size 个 token；
    - 每步直接用当前 context 做 forward；
    - 下一个 token 从 logits[:, -1, :] 取得。

    这和之前某些模型不同：
    之前的模型为了保证时间块划分/池化位置和训练一致，需要 padding 到 block_size。
    当前请求明确要求不要 padding，保持原始输入长度。
    """

    model.eval()

    start_text = normalize_input_text(start_text)

    if len(start_text) == 0:
        raise ValueError("输入文本不能为空。")

    oov_list = find_oov_characters(start_text, stoi)
    if oov_list:
        raise ValueError(format_oov_error(oov_list))

    ids = [
        stoi[ch]
        for ch in start_text
    ]

    input_ids = torch.tensor(
        ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    block_size = model.config.block_size

    if max_new_tokens <= 0:
        return decode_ids(
            input_ids[0].tolist(),
            itos
        )

    print(
        f"[GENERATE-INFO] block_size={block_size}, "
        f"input_tokens={input_ids.size(1)}, "
        f"max_new_tokens={max_new_tokens}, "
        f"padding=disabled",
        flush=True
    )

    for step in range(max_new_tokens):
        if (
            step == 0
            or (step + 1) % 10 == 0
            or step + 1 == max_new_tokens
        ):
            print(
                f"[GENERATING] {step + 1}/{max_new_tokens}",
                flush=True
            )

        context = input_ids[:, -block_size:]

        logits, _ = model(
            input_ids=context,
            targets=None
        )

        # 不做右侧 padding。
        # 直接取当前真实 context 的最后一个位置作为下一个 token 的预测。
        next_logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True
            )
        else:
            next_logits = next_logits / temperature

            if top_k is not None and top_k > 0:
                top_k_value = min(
                    int(top_k),
                    next_logits.size(-1)
                )

                values, _ = torch.topk(
                    next_logits,
                    k=top_k_value,
                    dim=-1
                )

                threshold = values[:, -1].unsqueeze(-1)

                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(
                        next_logits,
                        float("-inf")
                    ),
                    next_logits
                )

            probs = F.softmax(
                next_logits,
                dim=-1
            )

            next_id = torch.multinomial(
                probs,
                num_samples=1
            )

        input_ids = torch.cat(
            [
                input_ids,
                next_id
            ],
            dim=1
        )

    return decode_ids(
        input_ids[0].tolist(),
        itos
    )


# ============================================================
# 4. 全局状态
# ============================================================

class AppState:
    def __init__(self):
        self.model = None
        self.stoi = None
        self.itos = None
        self.config = None
        self.ckpt = None
        self.device = None
        self.checkpoint_path = None
        self.lock = threading.Lock()


STATE = AppState()


# ============================================================
# 5. HTML 页面
# ============================================================

def render_page(
    input_text: str = "",
    output_text: str = "",
    error_text: str = "",
    max_new_tokens: int = 20,
    temperature: float = 0.9,
    top_k: int = 50
) -> str:
    checkpoint_path = STATE.checkpoint_path or ""
    device_text = str(STATE.device) if STATE.device is not None else ""
    block_size = STATE.config.block_size if STATE.config is not None else ""
    vocab_size = STATE.config.vocab_size if STATE.config is not None else ""

    input_text = normalize_input_text(input_text)

    input_safe = html.escape(input_text)
    output_safe = html.escape(output_text)
    error_safe = html.escape(error_text)

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dilated Unshared Dynamic Conv LM 生成网页</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
            background: #f5f5f5;
            color: #222;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }}
        h1 {{
            margin-top: 0;
            font-size: 28px;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        label {{
            font-weight: 600;
            display: block;
            margin-bottom: 8px;
        }}
        textarea {{
            width: 100%;
            min-height: 260px;
            box-sizing: border-box;
            font-size: 16px;
            line-height: 1.6;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 8px;
            resize: vertical;
            font-family: "Consolas", "Microsoft YaHei", monospace;
        }}
        input {{
            font-size: 15px;
            padding: 8px;
            border-radius: 6px;
            border: 1px solid #ccc;
            margin-right: 12px;
        }}
        button {{
            font-size: 16px;
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            cursor: pointer;
            font-weight: 600;
        }}
        button:hover {{
            background: #1d4ed8;
        }}
        button:disabled {{
            background: #94a3b8;
            cursor: not-allowed;
        }}
        .row {{
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            align-items: center;
            margin-bottom: 16px;
        }}
        .meta {{
            color: #555;
            line-height: 1.7;
            font-size: 14px;
        }}
        .error {{
            background: #fee2e2;
            color: #991b1b;
            padding: 12px;
            border-radius: 8px;
            white-space: pre-wrap;
        }}
        .output {{
            white-space: pre-wrap;
            line-height: 1.7;
            font-size: 16px;
            font-family: "Consolas", "Microsoft YaHei", monospace;
        }}
        .hint {{
            color: #666;
            font-size: 14px;
            line-height: 1.6;
            margin-top: 8px;
        }}
        .warn {{
            color: #92400e;
            background: #fef3c7;
            padding: 10px 12px;
            border-radius: 8px;
            line-height: 1.6;
            font-size: 14px;
        }}
        .small {{
            font-size: 13px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dilated Unshared Dynamic Conv LM 网页生成</h1>

        <div class="card meta">
            <div><b>Checkpoint:</b> {html.escape(checkpoint_path)}</div>
            <div><b>Device:</b> {html.escape(device_text)}</div>
            <div><b>模型 block_size:</b> {block_size}</div>
            <div><b>词表大小:</b> {vocab_size}</div>
            <div><b>推理策略:</b> 不做右侧 padding，保持原始输入长度；超过 block_size 时只保留最后 block_size 个 token。</div>
        </div>

        <div class="card warn">
            <b>性能提醒：</b>
            当前模型 block_size 是 {block_size}。
            如果上下文很长，或者生成 token 数很大，推理会比较慢。
            建议先用 1、5、10、20 个 token 测试。
        </div>

        <form method="POST" action="/generate" id="generate-form">
            <div class="card">
                <label>输入上下文</label>
                <textarea name="input_text">{input_safe}</textarea>
                <div class="hint">
                    这里可以粘贴很长的上下文，不再通过 Python 命令行参数传入，因此不会遇到命令行参数长度限制。
                    程序会自动把 Windows 换行 \\r\\n 转换为 \\n。
                </div>
            </div>

            <div class="card">
                <div class="row">
                    <div>
                        <label>生成 token 数</label>
                        <input type="number" name="max_new_tokens" value="{int(max_new_tokens)}" min="1" max="2000">
                        <div class="small">建议先填 1、5、10、20。</div>
                    </div>
                    <div>
                        <label>temperature</label>
                        <input type="number" name="temperature" value="{float(temperature)}" step="0.05">
                    </div>
                    <div>
                        <label>top_k</label>
                        <input type="number" name="top_k" value="{int(top_k)}" min="0">
                    </div>
                </div>
                <button type="submit" id="submit-button">开始生成</button>
                <span id="status-text" class="small"></span>
            </div>
        </form>

        {f'<div class="card"><div class="error">{error_safe}</div></div>' if error_text else ''}

        {f'<div class="card"><label>生成结果</label><div class="output">{output_safe}</div></div>' if output_text else ''}
    </div>

    <script>
        const form = document.getElementById("generate-form");
        const button = document.getElementById("submit-button");
        const statusText = document.getElementById("status-text");

        form.addEventListener("submit", function() {{
            button.disabled = true;
            button.textContent = "正在生成...";
            statusText.textContent = " 请求已提交，服务端正在推理。请不要重复点击。";
        }});
    </script>
</body>
</html>"""

    return page


# ============================================================
# 6. HTTP Handler
# ============================================================

class GenerateHandler(BaseHTTPRequestHandler):
    # 单个 POST 最大请求体。
    # 这里设为 256MB，足够长上下文 textarea 使用。
    MAX_BODY_BYTES = 256 * 1024 * 1024

    def do_GET(self):
        try:
            if self.path == "/" or self.path.startswith("/index"):
                self.safe_send_html(
                    render_page()
                )
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()
                self.safe_write_bytes(b"404 Not Found")
        except BrokenPipeError:
            print("[CLIENT-DISCONNECTED] GET BrokenPipeError", flush=True)
        except ConnectionResetError:
            print("[CLIENT-DISCONNECTED] GET ConnectionResetError", flush=True)
        except Exception:
            print("[GET-ERROR]", flush=True)
            traceback.print_exc()

    def do_POST(self):
        print(
            f"[POST-ENTER] path={self.path}, client={self.client_address}",
            flush=True
        )

        if self.path != "/generate":
            try:
                self.send_response(404)
                self.end_headers()
                self.safe_write_bytes(b"404 Not Found")
            except Exception:
                pass
            return

        input_text = ""
        max_new_tokens = 20
        temperature = 0.9
        top_k = 50

        try:
            content_length = int(
                self.headers.get("Content-Length", "0")
            )

            print(
                f"[POST-RECEIVED] Content-Length={content_length}",
                flush=True
            )

            if content_length <= 0:
                raise ValueError("POST 请求体为空。")

            if content_length > self.MAX_BODY_BYTES:
                raise ValueError(
                    f"请求体太大：{content_length} bytes，最大允许 {self.MAX_BODY_BYTES} bytes。"
                )

            raw_body_bytes = self.rfile.read(content_length)

            print(
                f"[POST-BODY-READ] bytes={len(raw_body_bytes)}",
                flush=True
            )

            raw_body = raw_body_bytes.decode(
                "utf-8",
                errors="replace"
            )

            form = parse_qs(
                raw_body,
                keep_blank_values=True
            )

            input_text = form.get("input_text", [""])[0]
            input_text = normalize_input_text(input_text)

            max_new_tokens = int(form.get("max_new_tokens", ["20"])[0])
            temperature = float(form.get("temperature", ["0.9"])[0])
            top_k = int(form.get("top_k", ["50"])[0])

            if max_new_tokens < 0:
                raise ValueError("max_new_tokens 不能小于 0。")

            if max_new_tokens > 2000:
                raise ValueError("网页版为了避免浏览器超时，max_new_tokens 最大限制为 2000。")

            if STATE.model is None:
                raise RuntimeError("模型尚未加载。")

            print(
                f"[GENERATE-START] input_chars={len(input_text)}, "
                f"max_new_tokens={max_new_tokens}, "
                f"temperature={temperature}, "
                f"top_k={top_k}, "
                f"padding=disabled",
                flush=True
            )

            start_time = time.time()

            with STATE.lock:
                output_text = generate_without_right_padding(
                    model=STATE.model,
                    start_text=input_text,
                    stoi=STATE.stoi,
                    itos=STATE.itos,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=STATE.device
                )

            elapsed = time.time() - start_time

            print(
                f"[GENERATE-DONE] elapsed={elapsed:.2f}s",
                flush=True
            )

            output_text = (
                output_text
                + "\n\n"
                + "=" * 80
                + "\n"
                + f"生成耗时: {elapsed:.2f} 秒\n"
                + f"输入字符数: {len(input_text)}\n"
                + f"max_new_tokens: {max_new_tokens}\n"
                + f"temperature: {temperature}\n"
                + f"top_k: {top_k}\n"
                + "padding: disabled\n"
                + "=" * 80
            )

            self.safe_send_html(
                render_page(
                    input_text=input_text,
                    output_text=output_text,
                    error_text="",
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
            )

        except BrokenPipeError:
            print(
                "[CLIENT-DISCONNECTED] POST BrokenPipeError: 浏览器已断开，生成结果无法写回。",
                flush=True
            )

        except ConnectionResetError:
            print(
                "[CLIENT-DISCONNECTED] POST ConnectionResetError: 浏览器已断开，生成结果无法写回。",
                flush=True
            )

        except socket.timeout:
            print(
                "[CLIENT-DISCONNECTED] POST socket.timeout",
                flush=True
            )

        except Exception as e:
            print("[POST-ERROR]", flush=True)
            traceback.print_exc()

            error_text = str(e)

            try:
                self.safe_send_html(
                    render_page(
                        input_text=input_text,
                        output_text="",
                        error_text=error_text,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )
                )
            except BrokenPipeError:
                print(
                    "[CLIENT-DISCONNECTED] POST error response BrokenPipeError",
                    flush=True
                )
            except ConnectionResetError:
                print(
                    "[CLIENT-DISCONNECTED] POST error response ConnectionResetError",
                    flush=True
                )
            except Exception:
                print("[POST-ERROR-RESPONSE-FAILED]", flush=True)
                traceback.print_exc()

    def safe_write_bytes(self, data: bytes):
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            raise
        except ConnectionResetError:
            raise

    def safe_send_html(self, text: str):
        data = text.encode("utf-8")

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except BrokenPipeError:
            raise
        except ConnectionResetError:
            raise

    def log_message(self, format, *args):
        print(
            "[%s] %s" % (
                self.log_date_time_string(),
                format % args
            ),
            flush=True
        )


# ============================================================
# 7. 加载模型
# ============================================================

def load_model_for_web(
    checkpoint_path: str,
    device_arg: str,
    compile_model: bool = False
):
    if device_arg == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_arg)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"没有找到 checkpoint: {checkpoint_path}"
        )

    torch.backends.cudnn.benchmark = True

    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model, stoi, itos, config, ckpt = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device
    )

    model.eval()

    if compile_model:
        if device.type == "cuda":
            print(
                "正在 torch.compile 编译模型。注意：首次生成可能非常慢。",
                flush=True
            )
            model = torch.compile(model)
        else:
            print(
                "当前不是 cuda，跳过 torch.compile。",
                flush=True
            )

    STATE.model = model
    STATE.stoi = stoi
    STATE.itos = itos
    STATE.config = config
    STATE.ckpt = ckpt
    STATE.device = device
    STATE.checkpoint_path = checkpoint_path

    print("=" * 80, flush=True)
    print("模型加载完成", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Epoch: {ckpt.get('epoch')}", flush=True)
    print(f"Global step: {ckpt.get('global_step')}", flush=True)
    print(f"Best val loss: {ckpt.get('best_val_loss')}", flush=True)
    print(f"block_size: {config.block_size}", flush=True)
    print(f"vocab_size: {config.vocab_size}", flush=True)
    print("inference padding: disabled", flush=True)
    print("=" * 80, flush=True)


# ============================================================
# 8. argparse
# ============================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Dilated Unshared Dynamic Conv LM Web Generator"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="checkpoint 路径，例如 ./ckpt_dil_unshared/best_model.pth"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto / cuda / cpu"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="网页服务 host。WSL 下如果 Windows 浏览器访问不稳定，可用 0.0.0.0"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="网页服务端口"
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="是否尝试 torch.compile。首次生成可能很慢，不稳定时不要开启。"
    )

    return parser


# ============================================================
# 9. main
# ============================================================

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    load_model_for_web(
        checkpoint_path=args.checkpoint_path,
        device_arg=args.device,
        compile_model=args.compile
    )

    server = ThreadingHTTPServer(
        (args.host, args.port),
        GenerateHandler
    )

    # 避免请求连接长时间卡住。
    # 注意：生成本身可能很久，这里不强行杀生成，只影响 socket 操作。
    server.timeout = 1.0

    url = f"http://{args.host}:{args.port}"

    print("=" * 80, flush=True)
    print("网页生成服务已启动", flush=True)
    print(f"打开浏览器访问: {url}", flush=True)
    print("按 Ctrl+C 退出", flush=True)
    print("=" * 80, flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("", flush=True)
        print("正在关闭服务...", flush=True)
    finally:
        server.server_close()
        print("服务已关闭。", flush=True)


if __name__ == "__main__":
    main()
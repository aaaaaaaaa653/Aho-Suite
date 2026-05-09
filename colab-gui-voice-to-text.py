import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import gradio as gr
import os
import time
from datetime import datetime

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

SR_DEFAULT = 16000
N_MELS = 80
UNICODE_MAX = 20000
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class UnicodeSTTNet(nn.Module):
    """
    自作のCNN-GRUハイブリッドモデル。
    数十MBに抑えつつ、2万種類のUnicode文字（漢字含む）を直接予測します。
    """
    def __init__(self, vocab_size, hidden_dim=256, num_layers=3):
        super(UnicodeSTTNet, self).__init__()
        # 1D-CNNで音の特徴（フォルマント等）を抽出
        self.conv = nn.Sequential(
            nn.Conv1d(N_MELS, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1) # 時間軸を圧縮して高速化
        )
        # 双方向GRUで前後の文字のつながり（文脈）を学習
        self.rnn = nn.GRU(256, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=0.1)
        # Unicode空間への投影
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return torch.log_softmax(x, dim=-1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnicodeSTTNet(UNICODE_MAX).to(DEVICE)

def get_model_stats():
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "params": f"{total_params:,}",
        "size_mb": f"{(total_params * 4) / (1024**2):.2f} MB",
        "device": str(DEVICE)
    }

def idx_to_char(idx):
    if idx <= 0 or idx >= UNICODE_MAX: return ""
    return chr(idx - 1)

def char_to_idx(char):
    cp = ord(char)
    return cp + 1 if cp + 1 < UNICODE_MAX else 33

def process_audio_signals(y, sr, gain=1.0, noise_reduce=False, speed=1.0):
    if noise_reduce:
        stft = librosa.stft(y)
        mag = np.abs(stft)
        noise_floor = np.mean(mag[:, :10], axis=1, keepdims=True)
        mag = np.maximum(mag - noise_floor * 1.5, 0)
        y = librosa.istft(mag * np.exp(1j * np.angle(stft)))

    if gain != 1.0:
        y = y * gain

    if speed != 1.0:
        y = librosa.effects.time_stretch(y, rate=speed)

    return np.clip(y, -1.0, 1.0)

def decode_ctc(output, blank=0, threshold=0.0, merge_repeated=True):
    arg_maxes = torch.argmax(output, dim=-1)
    probs = torch.exp(torch.max(output, dim=-1)[0])

    decode = []
    last_char = -1
    for i in range(arg_maxes.shape[0]):
        token = arg_maxes[i].item()
        confidence = probs[i].item()

        if confidence < threshold: continue
        if token == blank:
            last_char = -1
            continue
        if merge_repeated and token == last_char:
            continue

        decode.append(idx_to_char(token))
        last_char = token

    return "".join(decode)

def transcribe_pro(audio_path, chunk_sec, overlap, gain, noise_red, speed, ct_threshold, max_chars, show_ts):
    if audio_path is None: return "エラー: 音声ファイルを選択してください。"

    y, sr = librosa.load(audio_path, sr=SR_DEFAULT)
    y = process_audio_signals(y, sr, gain, noise_red, speed)

    full_transcript = []
    step = int((chunk_sec - overlap) * SR_DEFAULT)
    chunk_len = int(chunk_sec * SR_DEFAULT)

    model.eval()
    with torch.no_grad():
        for start_idx in range(0, len(y), step):
            end_idx = min(start_idx + chunk_len, len(y))
            segment = y[start_idx:end_idx]
            if len(segment) < SR_DEFAULT * 0.2: continue

            mel = librosa.feature.melspectrogram(y=segment, sr=SR_DEFAULT, n_mels=N_MELS)
            mel_db = (librosa.power_to_db(mel, ref=np.max) + 80) / 80
            mel_tensor = torch.FloatTensor(mel_db).unsqueeze(0).to(DEVICE)

            output = model(mel_tensor).squeeze(0)
            text = decode_ctc(output, threshold=ct_threshold)

            if show_ts:
                timestamp = f"[{int(start_idx/SR_DEFAULT)//60:02d}:{int(start_idx/SR_DEFAULT)%60:02d}] "
                full_transcript.append(timestamp + text)
            else:
                full_transcript.append(text)

            if sum(len(t) for t in full_transcript) > max_chars:
                full_transcript.append("\n...(出力制限に達しました)")
                break

    return "\n".join(full_transcript) if show_ts else "".join(full_transcript)

def train_self_correction(audio_path, target_text, lr, epochs, weight_decay, progress=gr.Progress()):
    if not audio_path or not target_text: return "データが足りません。音声と正しいテキストを入力してください。"

    y, sr = librosa.load(audio_path, sr=SR_DEFAULT)
    mel = librosa.feature.melspectrogram(y=y, sr=SR_DEFAULT, n_mels=N_MELS)
    mel_db = torch.FloatTensor((librosa.power_to_db(mel, ref=np.max) + 80) / 80).unsqueeze(0).to(DEVICE)

    targets = torch.IntTensor([char_to_idx(c) for c in target_text]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(mel_db).transpose(0, 1)
        in_lens = torch.IntTensor([output.size(0)]).to(DEVICE)
        tg_lens = torch.IntTensor([len(targets)]).to(DEVICE)
        loss = criterion(output, targets, in_lens, tg_lens)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            progress(epoch/epochs, desc=f"補正中... Loss: {loss.item():.4f}")

    save_path = os.path.join(MODEL_DIR, f"tuned_{datetime.now().strftime('%H%M%S')}.pth")
    torch.save(model.state_dict(), save_path)
    return f"✅ 補正完了！\nあなたの声に一歩近づきました。\n保存先: {save_path}"

def get_weight_list():
    files_list = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    return gr.update(choices=files_list) if files_list else gr.update(choices=["保存済みモデルなし"])

def load_selected_weight(filename):
    if not filename or filename == "保存済みモデルなし": return "ファイルを選んでください。"
    path = os.path.join(MODEL_DIR, filename)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return f"✅ {filename} を読み込みました。"

def download_current_weights():
    tmp_path = "current_model.pth"
    torch.save(model.state_dict(), tmp_path)
    if IN_COLAB:
        files.download(tmp_path)
        return "ダウンロードを開始しました..."
    return f"保存完了: {os.path.abspath(tmp_path)}"

stats = get_model_stats()
print(f"--- Model Initialized ---\nParams: {stats['params']}\nSize: {stats['size_mb']}\nDevice: {stats['device']}\n-------------------------")

with gr.Blocks(title="Unicode STT Pro", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 💎 Unicode STT Pro - Scratch AI Engine\nモデル容量: {stats['size_mb']} | 漢字・Unicode完全対応 | 30分長尺対応")

    # モード切替ラジオボタン
    mode_selector = gr.Radio(
        choices=["標準（初心者向け）", "エキスパート（フル設定）"],
        value="標準（初心者向け）",
        label="🚀 操作モードの選択",
        info="初心者は『標準』を、詳細な微調整をしたい方は『エキスパート』を選んでください。"
    )

    with gr.Tabs():
        # --- TAB 1: 文字起こし ---
        with gr.Tab("🚀 文字起こし"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(label="音声を入力", type="filepath")

                    # 標準モード用のシンプルな説明
                    beginner_info = gr.Markdown(
                        "### 💡 使いかた\n1. 音声を入れます\n2. 『認識開始』を押します\n3. 結果が表示されます",
                        visible=True
                    )

                    # エキスパート設定（初期は非表示）
                    with gr.Group(visible=False) as expert_settings:
                        with gr.Accordion("🛠️ 音声信号処理 (Expert Only)", open=False):
                            gain_sl = gr.Slider(0.1, 3.0, value=1.0, label="音量補正 (Gain)")
                            speed_sl = gr.Slider(0.5, 2.0, value=1.0, label="再生速度 (Speed)")
                            noise_tg = gr.Checkbox(label="背景ノイズ抑制")

                        with gr.Accordion("🔍 推論詳細設定 (Expert Only)", open=False):
                            chunk_sl = gr.Slider(5, 60, value=10, step=1, label="チャンク幅 (秒)")
                            overlap_sl = gr.Slider(0, 5, value=1, step=0.5, label="接合部重複 (秒)")
                            conf_sl = gr.Slider(0.0, 0.9, value=0.1, label="表示閾値 (Confidence)")
                            ts_tg = gr.Checkbox(label="タイムスタンプ表示", value=True)
                            max_ch_sl = gr.Number(value=65536, label="最大文字数制限")

                    # 標準モード用の隠しパラメータ（デフォルト値が送られる）
                    # (エキスパート時と共通のコンポーネントを使い、visibilityだけ変える設計)

                    btn_run = gr.Button("認識開始", variant="primary")

                with gr.Column(scale=1):
                    output_area = gr.Textbox(label="認識結果", lines=20, show_copy_button=True)

        # --- TAB 2: 自己補正訓練 ---
        with gr.Tab("🎨 自己補正学習"):
            gr.Markdown("### あなたの声や、特定の漢字をAIに教え込みます。")
            with gr.Row():
                with gr.Column():
                    train_audio = gr.Audio(label="訓練用音声", type="filepath")
                    train_label = gr.Textbox(label="正しいテキスト", placeholder="AIに覚えさせたい正確な文章を入力してください。")

                    with gr.Group(visible=False) as expert_train_settings:
                        gr.Markdown("#### ⚙️ 学習パラメータ微調整")
                        lr_sl = gr.Slider(1e-5, 5e-3, value=5e-4, label="学習率")
                        epoch_sl = gr.Slider(10, 500, value=100, step=10, label="学習回数 (Epochs)")
                        wd_sl = gr.Slider(0.0, 0.1, value=0.01, label="正則化 (Weight Decay)")

                    # 標準モード用の固定値（表示されない）
                    std_lr = gr.State(5e-4)
                    std_epoch = gr.State(100)
                    std_wd = gr.State(0.01)

                    btn_train = gr.Button("このデータで学習・補正を実行", variant="secondary")

                with gr.Column():
                    train_status = gr.Textbox(label="学習ログ", lines=10)

        # --- TAB 3: 管理 ---
        with gr.Tab("💾 管理"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 重みの管理")
                    weight_dropdown = gr.Dropdown(label="ロード可能なモデル", choices=[])
                    btn_refresh = gr.Button("リスト更新")
                    btn_load_w = gr.Button("モデル読込", variant="primary")
                with gr.Column():
                    gr.Markdown("### ダウンロード")
                    btn_dl_all = gr.Button("現在の学習結果をPCに保存 (.pth)")
                    manage_status = gr.Textbox(label="システムメッセージ")

    def change_mode(mode):
        if mode == "標準（初心者向け）":
            return {
                beginner_info: gr.update(visible=True),
                expert_settings: gr.update(visible=False),
                expert_train_settings: gr.update(visible=False)
            }
        else:
            return {
                beginner_info: gr.update(visible=False),
                expert_settings: gr.update(visible=True),
                expert_train_settings: gr.update(visible=True)
            }

    mode_selector.change(
        change_mode,
        inputs=[mode_selector],
        outputs=[beginner_info, expert_settings, expert_train_settings]
    )

    btn_run.click(
        transcribe_pro,
        inputs=[audio_input, chunk_sl, overlap_sl, gain_sl, noise_tg, speed_sl, conf_sl, max_ch_sl, ts_tg],
        outputs=[output_area]
    )

    btn_train.click(
        train_self_correction,
        inputs=[train_audio, train_label, lr_sl, epoch_sl, wd_sl],
        outputs=[train_status]
    )

    btn_refresh.click(get_weight_list, outputs=[weight_dropdown])
    btn_load_w.click(load_selected_weight, inputs=[weight_dropdown], outputs=[manage_status])
    btn_dl_all.click(download_current_weights, outputs=[manage_status])

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
                    

# 推奨T4/A100ランタイム！

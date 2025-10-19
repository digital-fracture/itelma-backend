import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- МОДЕЛЬ (как у тебя) ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # [B,T,D]
        return x + self.pe[:, : x.size(1), :]


class TemporalMTL(nn.Module):
    def __init__(
        self,
        in_dim=9,
        d_model=128,
        nhead=4,
        nlayers=3,
        dropout=0.1,
        enc_type="transformer",
        pred_len=60,
    ):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        if enc_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
            self.pos = PositionalEncoding(d_model, max_len=1200)
        else:
            self.backbone = nn.LSTM(
                d_model,
                d_model // 2,
                num_layers=nlayers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
            )
            self.pos = None
        self.head_seq = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        self.head_clf = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 3)
        )

    def forward(self, x):
        h = self.inp(x)
        if self.pos is not None:
            h = self.pos(h)
        if isinstance(self.backbone, nn.TransformerEncoder):
            h = self.backbone(h)
        else:
            h, _ = self.backbone(h)
        h_pool = h.mean(dim=1)
        y_seq = self.head_seq(h_pool)
        y_clf = self.head_clf(h_pool)
        return y_seq, y_clf


# ---------- ХЕЛПЕРЫ ПРЕПРОЦЕССА (без файлов) ----------
def _pairs_to_df(
    pairs: list[tuple[float, float]], clip_min: float = None, clip_max: float = None
) -> pd.DataFrame:
    """list[(t, v)] -> DataFrame(time_sec,value,source='raw'), сортировка, дедуп."""
    if len(pairs) == 0:
        return pd.DataFrame(columns=["time_sec", "value", "source"])
    arr = np.asarray(pairs, dtype=float)
    df = pd.DataFrame({"time_sec": arr[:, 0], "value": arr[:, 1]})
    if clip_min is not None or clip_max is not None:
        df["value"] = df["value"].clip(lower=clip_min, upper=clip_max)
    df = df.dropna(subset=["time_sec", "value"]).sort_values("time_sec")
    df = df.drop_duplicates(subset=["time_sec"], keep="last").reset_index(drop=True)
    df["source"] = "raw"
    return df


def fill_small_gaps(
    time_sec: np.ndarray,
    value: np.ndarray,
    target_step: float,
    max_gap_s: float = 5.0,
    method: str = "linear",
) -> pd.DataFrame:
    """Сохраняем исходные точки; заполняем только мелкие дыры (dt<=max_gap_s) равномерно с шагом ~target_step."""
    t = np.asarray(time_sec, float)
    v = np.asarray(value, float)
    idx = np.argsort(t)
    t, v = t[idx], v[idx]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, v = t[keep], v[keep]
    if len(t) == 0:
        return pd.DataFrame(columns=["time_sec", "value", "source"])
    out_t, out_v, out_s = [t[0]], [v[0]], ["raw"]
    use_pchip = method.lower() == "pchip"
    # локальный PCHIP (двухточечный) = линейный; для настоящего PCHIP нужен контекст. Оставим линейный как стабильный.
    for i in range(len(t) - 1):
        t0, t1 = t[i], t[i + 1]
        v0, v1 = v[i], v[i + 1]
        dt = t1 - t0
        if dt > 0 and dt > target_step and dt <= max_gap_s:
            n_insert = int(math.floor(dt / target_step)) - 1
            if n_insert > 0:
                new_t = np.linspace(t0, t1, n_insert + 2)[1:-1]
                new_v = np.interp(new_t, [t0, t1], [v0, v1])
                out_t.extend(new_t.tolist())
                out_v.extend(new_v.tolist())
                out_s.extend(["interp"] * n_insert)
        out_t.append(t1)
        out_v.append(v1)
        out_s.append("raw")
    return pd.DataFrame({"time_sec": out_t, "value": out_v, "source": out_s})


def _to_1hz(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()
    s["sec"] = np.floor(s.time_sec).astype(int)

    def any_finite(x):
        x = np.asarray(x, float)
        return float(np.isfinite(x).any())

    agg = (
        s.groupby("sec")
        .agg(
            value=("value", "mean"),
            has_point=("value", any_finite),
            any_interp=("source", lambda x: float((x == "interp").any())),
        )
        .reset_index()
        .rename(columns={"sec": "time_sec"})
    )
    return agg


def frame_from_lists(
    bpm_pairs: list[tuple[float, float]],
    uc_pairs: list[tuple[float, float]],
    method: str = "linear",
    bpm_step: float = 0.25,
    uterus_step: float = 1.0,
    max_gap_s: float = 5.0,
) -> pd.DataFrame | None:
    """Строим общий 1Гц-кадр (fhr,uc,маски) из сырых списков."""
    bpm = _pairs_to_df(bpm_pairs, clip_min=50, clip_max=220)
    uc = _pairs_to_df(uc_pairs, clip_min=0, clip_max=80)
    if bpm.empty or uc.empty:
        return None
    bpm = fill_small_gaps(
        bpm.time_sec.values,
        bpm.value.values,
        target_step=bpm_step,
        max_gap_s=max_gap_s,
        method=method,
    )
    uc = fill_small_gaps(
        uc.time_sec.values,
        uc.value.values,
        target_step=uterus_step,
        max_gap_s=max_gap_s,
        method=method,
    )

    t0 = max(bpm.time_sec.min(), uc.time_sec.min())
    t1 = min(bpm.time_sec.max(), uc.time_sec.max())
    if t1 <= t0:
        return None
    bpm1 = _to_1hz(bpm)
    uc1 = _to_1hz(uc)
    secs = np.arange(int(math.ceil(t0)), int(math.floor(t1)) + 1, dtype=int)
    frame = pd.DataFrame({"time_sec": secs})
    frame = frame.merge(bpm1, on="time_sec", how="left").rename(
        columns={"value": "fhr", "has_point": "fhr_mask", "any_interp": "fhr_interp"}
    )
    frame = frame.merge(uc1, on="time_sec", how="left", suffixes=("", "_uc")).rename(
        columns={"value": "uc", "has_point": "uc_mask", "any_interp": "uc_interp"}
    )
    for c in ("fhr_mask", "uc_mask", "fhr_interp", "uc_interp"):
        frame[c] = frame[c].fillna(0.0).astype(float)
    return frame


# ---------- ФИЧИ + ПАКОВКА ----------
def _build_feats_block(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    fhr_valid = df["fhr"].where(df["fhr_mask"] > 0, np.nan)
    df["fhr_bl"] = fhr_valid.rolling(90, min_periods=1).median()
    df["fhr_stv"] = fhr_valid.diff().abs().rolling(60, min_periods=1).mean()
    uc_valid = df["uc"].where(df["uc_mask"] > 0, np.nan)
    df["uc_lp"] = uc_valid.rolling(5, min_periods=1).mean()
    df["fhr_bl"] = df["fhr_bl"].fillna(140.0)
    df["fhr_stv"] = df["fhr_stv"].fillna(0.0)
    df["uc_lp"] = df["uc_lp"].fillna(0.0)
    return df


def _pack_window(df_win: pd.DataFrame) -> np.ndarray:
    feats = [
        "fhr",
        "uc",
        "fhr_bl",
        "fhr_stv",
        "uc_lp",
        "fhr_mask",
        "uc_mask",
        "fhr_interp",
        "uc_interp",
    ]
    X = df_win[feats].to_numpy(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _load_trained_model(ckpt_path: str | Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = TemporalMTL(**ckpt["cfg"]).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


# ---------- ИНФЕРЕНС ИЗ СПИСКОВ ----------
@torch.no_grad()
def infer_last_window_from_lists(
    bpm_pairs: list[tuple[float, float]],
    uc_pairs: list[tuple[float, float]],
    ckpt_path: str | Path,
    *,
    method_interp: str = "linear",
    bpm_step: float = 0.25,
    uterus_step: float = 1.0,
    max_gap_s: float = 5.0,
    thr_late: float = 0.7,
    thr_h5: float = 0.7,
    thr_h15: float = 0.7,
    min_required: int = 60,
    max_window: int = 600,
) -> tuple[list[str], dict[str, ...]]:
    """
    Вход: сырые списки (t, value) для FHR и UC.
    Выход: messages (list[str]), details (dict).
    """
    frame = frame_from_lists(
        bpm_pairs,
        uc_pairs,
        method=method_interp,
        bpm_step=bpm_step,
        uterus_step=uterus_step,
        max_gap_s=max_gap_s,
    )
    if frame is None or len(frame) == 0:
        raise ValueError("Недостаточно данных (нет пересечения BPM/UC).")

    N_eff = min(len(frame), max_window)
    if N_eff < min_required:
        raise ValueError(f"Данных слишком мало: доступно {len(frame)} с, нужно ≥ {min_required} с.")

    df = _build_feats_block(frame)
    X = _pack_window(df.iloc[-N_eff:])  # [N_eff, 9]

    model = _load_trained_model(ckpt_path)
    x = torch.from_numpy(X).unsqueeze(0).to(DEVICE)
    y_seq, y_clf = model(x)
    fhr_next60 = y_seq.squeeze(0).float().cpu().numpy()
    fhr_next60_mean = float(np.mean(fhr_next60))
    fhr_next60_min = float(np.min(fhr_next60))

    probs = torch.sigmoid(y_clf).squeeze(0).float().cpu().numpy()
    p_late_10m, p_hyp_5m, p_hyp_15m = map(float, probs.tolist())

    messages = []
    if p_late_10m >= thr_late:
        messages.append(f"Высокий риск поздних децелераций в течение 10 мин - {p_late_10m:.0%}")
    elif p_late_10m >= 0.4:
        messages.append(
            f"Умеренный риск поздних децелераций в течение 10 мин - {p_late_10m:.0%}"
        )
    if p_hyp_5m >= thr_h5:
        messages.append(f"Высокий риск гипоксии (горизонт 5 мин) - {p_hyp_5m:.0%}")
    if p_hyp_15m >= thr_h15:
        messages.append(f"Высокий риск гипоксии (горизонт 15 мин) - {p_hyp_15m:.0%}")
    if not messages:
        messages.append("Существенных признаков нарастающего неблагополучия не выявлено")

    details = {
        "used_window_sec": int(N_eff),
        "p_late_10m": p_late_10m,
        "p_hyp_5m": p_hyp_5m,
        "p_hyp_15m": p_hyp_15m,
        "fhr_next60_mean": fhr_next60_mean,
        "fhr_next60_min": fhr_next60_min,
        "explanations": [
            "p_late_10m — вероятность хотя бы одной поздней децелерации в следующие 10 минут",
            "p_hyp_5m / p_hyp_15m — прокси-риск нарастания гипоксии на горизонтах 5 и 15 минут",
            "fhr_next60_mean — средний прогноз ЧСС на ближайшие 60 секунд",
            "fhr_next60_min — минимальная прогнозная ЧСС на ближайшие 60 секунд",
        ],
    }
    return messages, details


# ---------- Форматированный текст отчёта ----------
def format_ctg_report(messages, details, thr_low=0.4, thr_high=0.7):
    def badge(p):
        if p >= thr_high:
            return "🔴 высокий"
        if p >= thr_low:
            return "🟡 умеренный"
        return "🟢 низкий"

    p_late = float(details["p_late_10m"])
    p_h5 = float(details["p_hyp_5m"])
    p_h15 = float(details["p_hyp_15m"])
    mean60 = float(details["fhr_next60_mean"])
    min60 = float(details["fhr_next60_min"])
    N = int(details.get("used_window_sec", 0))
    lines = []
    lines.append(f"Сводка прогноза (последние {N} с)")
    if messages:
        lines.append(f"- {messages[0]}")
        for m in messages[1:]:
            lines.append(f"- {m}")
    lines.append(f"- {badge(p_late)} риск поздних децелераций 10 мин ({p_late:.0%})")
    lines.append(f"- {badge(p_h5)} риск гипоксии 5 мин ({p_h5:.0%})")
    lines.append(f"- {badge(p_h15)} риск гипоксии 15 мин ({p_h15:.0%})")
    lines.append("")
    lines.append(f"Прогноз ЧСС (60 с): среднее {mean60:.1f} bpm, минимум {min60:.1f} bpm")
    lines.append(
        "_Минимум — грубый индикатор ямки; клиническая децелерация оценивается относительно базального_"
    )
    lines.append("")
    lines.append("Что означают показатели:")
    lines.append(
        "- p_late_10m — вероятность хотя бы одной поздней децелерации в следующие 10 минут"
    )
    lines.append(
        "- p_hyp_5m / p_hyp_15m — прокси-риск нарастания гипоксии на горизонтах 5 и 15 минут"
    )
    lines.append("- fhr_next60_mean / min — прогноз ЧСС на ближайшую минуту: среднее и минимум")
    return "\n".join(lines)

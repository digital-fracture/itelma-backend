from bisect import bisect_left
from functools import partial

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def filter_min_max(data: list[tuple[float, float]], min_value, max_value):
    return [(t, v) for t, v in data if min_value <= v <= max_value], sum(
        1 - int(min_value <= v <= max_value) for t, v in data
    )


filter_min_max_bpm = partial(filter_min_max, min_value=50, max_value=220)
filter_min_max_uterus = partial(filter_min_max, min_value=0, max_value=80)


def remove_small_peaks(data: list[tuple[float, float]], value: float, time: float):
    new_data = []
    last_added = -1
    for idx, (t, v) in enumerate(data):
        if not new_data:
            new_data.append((t, v))
            last_added = idx
            continue

        if np.abs(new_data[-1][1] - v) >= value:
            if t - new_data[-1][0] < time:
                continue
            new_data += data[last_added + 1 : idx + 1]
        else:
            new_data.append((t, v))
        last_added = idx
    return new_data, len(data) - len(new_data)


remove_small_peaks_bpm = partial(remove_small_peaks, value=30, time=5)
remove_small_peaks_uterus = partial(remove_small_peaks, value=20, time=5)

filter_uterus = lambda data: remove_small_peaks_uterus(filter_min_max_uterus(data)[0])[0]
filter_bpm = lambda data: remove_small_peaks_bpm(filter_min_max_bpm(data)[0])[0]


# step = 0.25 if channel == "bpm" else (1 if channel == "uterus" else args.bpm_step)
# method = "linear"
def fill_small_gaps(
    data: list[tuple[float, float]],
    target_step: float,
    max_gap_s: float = 5.0,
    method: str = "linear",
) -> list[tuple[float, float]]:
    """
    Возвращает List of tuples: time_sec, value.
    - target_step: желаемый макс. шаг между соседними точками внутри допустимых дыр.
    - max_gap_s: только дыры <= max_gap_s заполняем; больше — не трогаем.
    - method: 'linear' или 'pchip' (если доступен SciPy).
    """
    if not data:
        return []

    t, v = zip(*data)
    t = np.asarray(t, float)
    v = np.asarray(v, float)

    # сортируем, убираем дубли и нестрогую монотонность
    idx = np.argsort(t)
    t, v = t[idx], v[idx]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, v = t[keep], v[keep]

    if t.size == 0:
        return []

    out_data = [(t[0], v[0])]

    # use_pchip = method.lower() == "pchip" and HAVE_PCHIP # Assuming HAVE_PCHIP is defined if pchip is available

    for i in range(len(t) - 1):
        t0, t1 = t[i], t[i + 1]
        v0, v1 = v[i], v[i + 1]
        dt = t1 - t0
        if dt <= 0:
            # патологический случай: пропускаем
            continue

        if dt > target_step and dt <= max_gap_s:
            # сколько новых точек нужны между (t0, t1), чтобы шаг ~ target_step
            # пример: dt=1.0, step=0.25 -> n_insert=3
            n_insert = int(np.floor(dt / target_step)) - 1
            if n_insert > 0:
                new_t = np.linspace(t0, t1, n_insert + 2)[1:-1]
                # if use_pchip:
                #     # локальный PCHIP: построим на узлах [t0, t1]
                #     # (для большей устойчивости можно расширить контекст, но MVP-достаточно)
                #     from scipy.interpolate import PchipInterpolator
                #     f = PchipInterpolator([t0, t1], [v0, v1])
                #     new_v = f(new_t)
                # else:
                # linear
                new_v = np.interp(new_t, [t0, t1], [v0, v1])

                out_data.extend(zip(new_t.tolist(), new_v.tolist()))

        # добавляем правую исходную точку
        out_data.append((t1, v1))

        # большие провалы (> max_gap_s) не заполняем — просто переходим к следующей raw-точке

    return out_data


interpolate_bpm = partial(fill_small_gaps, target_step=1)
interpolate_uterus = partial(fill_small_gaps, target_step=0.25)


def moving_average(data: list[tuple[float, float]], window_size=5) -> list[tuple[float, float]]:
    if not data or len(data) < window_size:
        return []

    times = [t for t, v in data]
    values = [v for t, v in data]

    moving_avg_values = np.convolve(values, np.ones(window_size) / window_size, mode="valid")
    moving_avg_times = times[window_size - 1 :]

    return list(zip(moving_avg_times, moving_avg_values.tolist()))


preprocess_bpm = lambda data: moving_average(interpolate_bpm(filter_bpm(data)))
preprocess_uterus = lambda data: moving_average(interpolate_uterus(filter_uterus(data)))


def find_decelerations(df_bpm, threshold=-10, min_duration=10):
    """
    Находит децелерации (снижения ЧСС).
    Возвращает список словарей с информацией о децелерации: {'start', 'end', 'min_time', 'min_value'}
    """

    # Дифференцируем значения
    diff = df_bpm["value"].diff()
    # Находим значительные падения
    decel_mask = diff < threshold

    # Группируем подряд идущие True
    groups = (decel_mask != decel_mask.shift()).cumsum()
    grouped = df_bpm[decel_mask].groupby(groups)

    decels = []
    for name, group in grouped:
        if len(group) < 2:
            continue
        start_time = group["time_sec"].iloc[0]
        end_time = group["time_sec"].iloc[-1]
        if (end_time - start_time) < min_duration:
            continue  # слишком короткая
        min_idx = group["value"].idxmin()
        min_time = group.loc[min_idx, "time_sec"]
        min_value = group.loc[min_idx, "value"]
        decels.append(
            {"start": start_time, "end": end_time, "min_time": min_time, "min_value": min_value}
        )
    return decels


def find_contraction_peaks(df_uterus, threshold=10.0):
    """
    Находит пики сокращений.
    Возвращает список словарей: [{'time', 'value'}]
    """
    peaks = []
    values = df_uterus["value"].values
    times = df_uterus["time_sec"].values

    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1] and values[i] > threshold:
            peaks.append({"time": times[i], "value": values[i]})
    return peaks


def classify_decelerations(decelerations, contractions, early_window=10, late_window=30):
    """
    Классифицирует децелерации на ранние, поздние, вариабельные.
    """
    early_count = 0
    late_count = 0
    variable_count = 0

    for d in decelerations:
        is_early = any(abs(d["min_time"] - c["time"]) <= early_window for c in contractions)
        is_late = any(
            c["time"] < d["min_time"] and abs(d["min_time"] - c["time"]) <= late_window
            for c in contractions
        )

        if is_early:
            early_count += 1
        elif is_late:
            late_count += 1
        else:
            variable_count += 1

    return {
        "early_decel_count": early_count,
        "late_decel_count": late_count,
        "variable_decel_count": variable_count,
    }


def calculate_metrics_with_deceleration_types(bpm_list: list, uterus_list: list) -> dict:
    def to_df(data_list):
        if data_list and hasattr(data_list[0], "_fields"):  # именованный кортеж
            return pd.DataFrame(
                [(x.time_sec, x.value) for x in data_list], columns=["time_sec", "value"]
            )
        else:
            return pd.DataFrame(data_list, columns=["time_sec", "value"])

    metrics = {}

    df_bpm = to_df(bpm_list)
    df_uterus = to_df(uterus_list)

    # 4. Среднее значение базовой ЧСС
    avg_bpm = df_bpm["value"].mean()
    metrics["avg_bpm"] = round(avg_bpm, 2)

    # 6. Частота акцелераций/децелераций
    diff = df_bpm["value"].diff().dropna()
    accels = (diff > 10).sum()
    decels = (diff < -10).sum()
    metrics["accel_count"] = int(accels)
    metrics["decel_count"] = int(decels)

    # Новые метрики: типы децелераций
    decelerations = find_decelerations(df_bpm)
    contractions = find_contraction_peaks(df_uterus)
    decel_types = classify_decelerations(decelerations, contractions)

    metrics.update(decel_types)

    # 7. Суммарное время тахикардии / брадикардии
    tachy_threshold = 160
    brady_threshold = 110

    tachy_mask = df_bpm["value"] > tachy_threshold
    brady_mask = df_bpm["value"] < brady_threshold

    def sum_consecutive_times(mask):
        if not mask.any():
            return 0.0
        groups = (mask != mask.shift()).cumsum()
        grouped = df_bpm[mask].groupby(groups)
        total_time = 0.0
        for _, group in grouped:
            if len(group) >= 2:
                start_t = group["time_sec"].iloc[0]
                end_t = group["time_sec"].iloc[-1]
                total_time += end_t - start_t
        return total_time

    tachy_time = sum_consecutive_times(tachy_mask)
    brady_time = sum_consecutive_times(brady_mask)

    metrics["tachycardia_time_sec"] = round(tachy_time, 2)
    metrics["bradycardia_time_sec"] = round(brady_time, 2)

    # Дополнительно: разбиение на умеренную/интенсивную
    tachy_moderate_mask = (df_bpm["value"] > 160) & (df_bpm["value"] <= 180)
    tachy_severe_mask = df_bpm["value"] > 180
    brady_moderate_mask = (df_bpm["value"] < 110) & (df_bpm["value"] >= 90)
    brady_severe_mask = df_bpm["value"] < 90

    tachy_moderate_time = sum_consecutive_times(tachy_moderate_mask)
    tachy_severe_time = sum_consecutive_times(tachy_severe_mask)
    brady_moderate_time = sum_consecutive_times(brady_moderate_mask)
    brady_severe_time = sum_consecutive_times(brady_severe_mask)

    metrics["tachycardia_moderate_time_sec"] = round(tachy_moderate_time, 2)
    metrics["tachycardia_severe_time_sec"] = round(tachy_severe_time, 2)
    metrics["bradycardia_moderate_time_sec"] = round(brady_moderate_time, 2)
    metrics["bradycardia_severe_time_sec"] = round(brady_severe_time, 2)

    #  Статус (упрощённо)
    status = "Нормальное состояние"
    if brady_time > 10 or tachy_time > 10:
        status = "Требуется внимание"
    if brady_time > 30 or tachy_time > 30:
        status = "Критическое состояние"
    metrics["status"] = status

    # 5. Частота маточных сокращений
    threshold = 10.0
    peaks = []
    values = df_uterus["value"].values
    times = df_uterus["time_sec"].values

    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1] and values[i] > threshold:
            peaks.append(i)

    total_time_min = (times[-1] - times[0]) / 60.0 if len(times) > 1 else 1.0
    contraction_rate = len(peaks) / total_time_min if total_time_min > 0 else 0.0

    metrics["contraction_count"] = len(peaks)
    metrics["contraction_rate_per_minute"] = round(contraction_rate, 2)

    return metrics


late_decel_text = "Поздняя децелерация"


class CTGEvent:
    def __init__(self, start_time, end_time, text=late_decel_text):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text


def list_to_array(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert list of (t, v) to two numpy arrays."""
    times = np.array([p[0] for p in points], dtype=float)
    vals = np.array([p[1] for p in points], dtype=float)
    return times, vals


def detect_late_decelerations(
    bpm,
    uterus,
    amp_thresh: float = 15.0,
    dur_min: float = 15.0,
    onset_to_nadir_thresh: float = 30.0,
    prolonged_thresh: float = 120.0,
    tol: float = 10.0,
) -> list[CTGEvent]:
    """
    Detect late decelerations from bpm (time, value) and uterus trace.

    Returns a list of CTGEvent objects.
    """
    t_bpm, v_bpm = list_to_array(bpm)
    t_uc, v_uc = list_to_array(uterus)

    mask = v_bpm <= -amp_thresh

    cand_segs = []
    N = len(mask)
    i = 0
    while i < N:
        if mask[i]:
            j = i
            while (j + 1 < N) and mask[j + 1]:
                j += 1
            t0 = t_bpm[i]
            t1 = t_bpm[j]
            duration = t1 - t0
            if duration >= dur_min:
                cand_segs.append((i, j))
            i = j + 1
        else:
            i += 1

    peaks, props = find_peaks(v_uc, prominence=(np.max(v_uc) - np.min(v_uc)) * 0.2, distance=1)

    late_list = []
    for i0, i1 in cand_segs:
        start_time = t_bpm[i0]
        end_time = t_bpm[i1]

        idx_nadir = i0 + np.argmin(v_bpm[i0 : i1 + 1])
        nadir_time = t_bpm[idx_nadir]
        onset_time = start_time

        if (nadir_time - onset_time) < onset_to_nadir_thresh:
            continue

        uc_idx = bisect_left(t_uc, nadir_time)

        best_peak = None
        best_diff = tol + 1
        for pk in peaks:
            t_pk = t_uc[pk]
            diff = abs(t_pk - nadir_time)
            if diff < best_diff:
                best_diff = diff
                best_peak = pk
        if best_peak is None:
            continue
        uc_peak_time = t_uc[best_peak]

        thresh_uc = v_uc[best_peak] * 0.3

        cs = best_peak
        while cs > 0 and v_uc[cs] > thresh_uc:
            cs -= 1
        contraction_start_time = t_uc[cs]

        ce = best_peak
        while ce < len(v_uc) - 1 and v_uc[ce] > thresh_uc:
            ce += 1
        contraction_end_time = t_uc[ce]

        if (nadir_time + tol) >= uc_peak_time and (end_time + tol) >= contraction_end_time:
            late_list.append(CTGEvent(start_time, end_time))

    return late_list


class CTGEvent:
    def __init__(self, start_time, end_time, text=late_decel_text):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text


def list_to_array(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert list of (t, v) to two numpy arrays."""
    times = np.array([p[0] for p in points], dtype=float)
    vals = np.array([p[1] for p in points], dtype=float)
    return times, vals


def detect_late_decelerations(
    bpm,
    uterus,
    amp_thresh: float = 15.0,
    dur_min: float = 15.0,
    onset_to_nadir_thresh: float = 30.0,
    prolonged_thresh: float = 120.0,
    tol: float = 10.0,
) -> list[CTGEvent]:
    """
    Detect late decelerations from bpm (time, value) and uterus trace.
    Returns a list of CTGEvent objects.
    """
    t_bpm, v_bpm = list_to_array(bpm)
    t_uc, v_uc = list_to_array(uterus)

    mask = v_bpm <= -amp_thresh

    cand_segs = []
    N = len(mask)
    i = 0
    while i < N:
        if mask[i]:
            j = i
            while (j + 1 < N) and mask[j + 1]:
                j += 1
            t0 = t_bpm[i]
            t1 = t_bpm[j]
            duration = t1 - t0
            if duration >= dur_min:
                cand_segs.append((i, j))
            i = j + 1
        else:
            i += 1

    peaks, props = find_peaks(v_uc, prominence=(np.max(v_uc) - np.min(v_uc)) * 0.2, distance=1)

    late_list = []
    for i0, i1 in cand_segs:
        start_time = t_bpm[i0]
        end_time = t_bpm[i1]

        idx_nadir = i0 + np.argmin(v_bpm[i0 : i1 + 1])
        nadir_time = t_bpm[idx_nadir]
        onset_time = start_time

        if (nadir_time - onset_time) < onset_to_nadir_thresh:
            continue

        uc_idx = bisect_left(t_uc, nadir_time)

        best_peak = None
        best_diff = tol + 1
        for pk in peaks:
            t_pk = t_uc[pk]
            diff = abs(t_pk - nadir_time)
            if diff < best_diff:
                best_diff = diff
                best_peak = pk
        if best_peak is None:
            continue
        uc_peak_time = t_uc[best_peak]

        thresh_uc = v_uc[best_peak] * 0.3

        cs = best_peak
        while cs > 0 and v_uc[cs] > thresh_uc:
            cs -= 1
        contraction_start_time = t_uc[cs]

        ce = best_peak
        while ce < len(v_uc) - 1 and v_uc[ce] > thresh_uc:
            ce += 1
        contraction_end_time = t_uc[ce]

        if (nadir_time + tol) >= uc_peak_time and (end_time + tol) >= contraction_end_time:
            late_list.append(CTGEvent(start_time, end_time))

    return late_list


class CTGRecord:
    data = None
    preprocessed_bpm = None
    preprocessed_uterus = None
    metrics = None
    regions = None

    def __init__(self, bpm: list[tuple[float, float]], uterus: list[tuple[float, float]]):
        self.bpm = bpm
        self.uterus = uterus

    def preprocess(self, force=False):
        if not force and self.preprocessed_bpm is not None:
            return self.preprocessed_bpm, self.preprocessed_uterus
        self.preprocessed_bpm = preprocess_bpm(self.bpm)
        self.preprocessed_uterus = preprocess_uterus(self.uterus)
        return self.preprocessed_bpm, self.preprocessed_uterus

    def calc_metrics(self, force=False):
        if not force and self.metrics is not None:
            return self.metrics

        self.preprocess()

        self.metrics = calculate_metrics_with_deceleration_types(
            self.preprocessed_bpm, self.preprocessed_uterus
        )
        return self.metrics

    def detect_regions(self, force=False):
        if not force and self.regions is not None:
            return self.regions
        self.preprocess()
        try:
            self.regions = detect_late_decelerations(
                self.preprocessed_bpm,
                self.preprocessed_uterus,
                amp_thresh=15,
                dur_min=15,
                onset_to_nadir_thresh=30,
            )
        except:
            self.regions = []
        return self.regions

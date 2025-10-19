import asyncio
from functools import partial

from app.core import Paths
from app.model import (
    EmulationPrediction,
    ExaminationPartInterval,
    ExaminationPlot,
    ExaminationStats,
    ExaminationVerdict,
    PipelineResult,
)

from .pipeline import CTGRecord
from .predictions import infer_last_window_from_lists
from .verdict import generate_comprehensive_ctg_report


def analyze_sync(plot: ExaminationPlot, *, no_verdict: bool = False) -> PipelineResult:
    try:
        prediction_messages, prediction_details = infer_last_window_from_lists(
            bpm_pairs=plot.bpm,
            uc_pairs=plot.uterus,
            ckpt_path=Paths.ml.predictor_model,
        )
    except ValueError:
        prediction_messages, prediction_details = None, None

    record = CTGRecord(bpm=plot.bpm, uterus=plot.uterus)
    raw_stats = record.calc_metrics()
    raw_intervals = record.detect_regions()

    prediction = (
        EmulationPrediction(
            messages=prediction_messages,
            bpm_average=prediction_details["fhr_next60_mean"],
            bpm_min=prediction_details["fhr_next60_min"],
        )
        if prediction_messages is not None
        else None
    )

    stats = ExaminationStats(
        bpm_average=raw_stats["avg_bpm"],
        uterus_average=raw_stats["contraction_rate_per_minute"],
        acceleration_count=raw_stats["accel_count"],
        deceleration_count=raw_stats["decel_count"],
        late_deceleration_count=raw_stats["late_decel_count"],
        early_deceleration_count=raw_stats["early_decel_count"],
        variable_deceleration_count=raw_stats["variable_decel_count"],
        mild_tachycardia_seconds=raw_stats["tachycardia_moderate_time_sec"],
        severe_tachycardia_seconds=raw_stats["tachycardia_moderate_time_sec"],
        mild_bradycardia_seconds=raw_stats["bradycardia_moderate_time_sec"],
        severe_bradycardia_seconds=raw_stats["bradycardia_severe_time_sec"],
        condition=raw_stats["status"],
    )

    intervals = [
        ExaminationPartInterval(
            start=interval.start_time, end=interval.end_time, message=interval.text
        )
        for interval in raw_intervals
    ]

    verdict = (
        ExaminationVerdict(**generate_comprehensive_ctg_report(prediction_details | raw_stats))
        if not no_verdict and prediction_details is not None
        else None
    )

    return PipelineResult(
        prediction=prediction,
        intervals=intervals,
        stats=stats,
        verdict=verdict,
    )


async def analyze(plot: ExaminationPlot, *, no_verdict: bool = False) -> PipelineResult:
    return await asyncio.to_thread(partial(analyze_sync, plot, no_verdict=no_verdict))

import asyncio

from app.core import Paths
from app.model import EmulationPrediction, ExaminationPlot

from .predictions import infer_last_window_from_lists


def predict_sync(plot: ExaminationPlot) -> EmulationPrediction:
    messages, details = infer_last_window_from_lists(
        bpm_pairs=plot.bpm,
        uc_pairs=plot.uterus,
        ckpt_path=Paths.ml.predictor_model,
    )

    return EmulationPrediction(
        messages=messages,
        bpm_average=details["fhr_next60_mean"],
        bpm_min=details["fhr_next60_min"],
    )


async def predict(plot: ExaminationPlot) -> EmulationPrediction:
    return await asyncio.to_thread(predict_sync, plot)

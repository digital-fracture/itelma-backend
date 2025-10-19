def generate_comprehensive_ctg_report(params):
    """
    Генерирует полный клинический отчёт по КТГ с прогнозом.
    Возвращает словарь с заключением, рекомендациями, зонами внимания и рисками.
    """
    report = {
        "overall_status": "",
        "recommendations": [],
        "attention_zones": [],  # что в норме — можно не трогать
        "risk_zones": [],  # что требует внимания
    }

    # ========== 1. Текущее состояние ПЛОДА ==========
    fhr = params["fhr_avg"]
    has_accelerations = params["accelerations"] >= 2
    late_dec_total = params["late_decelerations"] + params["late_decelerations_current"]
    recurrent_late = late_dec_total >= 3
    prolonged_bradycardia = (
        params["bradycardia_severe_duration"] > 5 or params["bradycardia_mild_duration"] > 20
    )
    prolonged_tachycardia = params["tachycardia_severe_duration"] > 10
    var_dec = params["variable_decelerations"]

    # ========== 2. Прогнозные риски ==========
    p_late_10m = params.get("p_late_10m", 0)
    p_hyp_5m = params.get("p_hyp_5m", 0)
    p_hyp_15m = params.get("p_hyp_15m", 0)
    fhr_mean_pred = params.get("fhr_next60_mean", fhr)
    fhr_min_pred = params.get("fhr_next60_min", fhr)

    # ========== 3. Риск для МАТЕРИ ==========
    contractions_10min = params["uc_avg"]
    total_decelerations = late_dec_total + var_dec
    hyperstimulation = (contractions_10min >= 5) and (total_decelerations >= 3)
    hypotonia = contractions_10min <= 2

    # ========== 4. Анализ и сбор данных для отчёта ==========
    fetal_risks = []
    maternal_risks = []
    fetal_ok = []
    maternal_ok = []

    # --- Плод: текущее состояние ---
    if fhr < 100 or fhr > 180:
        fetal_risks.append("Критическая брадикардия/тахикардия (ЧСС вне 100–180 уд/мин)")
    elif 110 <= fhr <= 160:
        fetal_ok.append("Базальный ритм в норме (110–160 уд/мин)")
    else:
        fetal_risks.append(f"Умеренная тахикардия/брадикардия (ЧСС = {fhr})")

    if has_accelerations:
        fetal_ok.append("Акцелерации присутствуют — признак отсутствия гипоксии")
    else:
        fetal_risks.append("Акцелерации отсутствуют — снижена вариабельность/риск гипоксии")

    if recurrent_late:
        fetal_risks.append(
            "Рецидивирующие поздние децелерации — признак маточно-плацентарной недостаточности"
        )
    elif late_dec_total == 0:
        fetal_ok.append("Поздние децелерации отсутствуют")
    else:
        fetal_risks.append(f"Единичные поздние децелерации ({late_dec_total}) — требуют наблюдения")

    if var_dec >= 5:
        fetal_risks.append("Частые вариабельные децелерации — возможна компрессия пуповины")
    elif var_dec <= 2:
        fetal_ok.append("Вариабельные децелерации редкие или отсутствуют")

    if prolonged_bradycardia:
        fetal_risks.append("Продолжительная брадикардия — угроза острой гипоксии")
    if prolonged_tachycardia:
        fetal_risks.append("Выраженная тахикардия >10 мин — возможна инфекция/гипоксия")

    # --- Прогнозные риски ---
    if p_late_10m >= 0.5:
        fetal_risks.append(
            f"Высокая вероятность поздней децелерации в ближайшие 10 мин (p={p_late_10m:.2f})"
        )
    if p_hyp_5m >= 0.3:
        fetal_risks.append(f"Острый прогноз гипоксии в течение 5 мин (p={p_hyp_5m:.2f})")
    elif p_hyp_15m >= 0.1:
        fetal_risks.append(f"Повышенный риск гипоксии в течение 15 мин (p={p_hyp_15m:.2f})")
    if fhr_mean_pred > 180:
        fetal_risks.append(
            f"Прогнозируемая тахикардия (средняя ЧСС за 60 сек: {fhr_mean_pred:.0f})"
        )
    if fhr_min_pred < 100:
        fetal_risks.append(f"Прогнозируемая брадикардия (мин. ЧСС за 60 сек: {fhr_min_pred:.0f})")

    # --- Мать ---
    if hyperstimulation:
        maternal_risks.append(
            "Гиперстимуляция матки (>5 сокращений/10 мин) + признаки гипоксии — риск разрыва матки"
        )
    elif contractions_10min >= 3 and contractions_10min <= 5:
        maternal_ok.append("Частота сокращений в пределах нормы (3–5 за 10 мин)")
    elif hypotonia:
        maternal_risks.append(
            "Гипотония матки (≤2 сокращения/10 мин) — риск затяжных родов, инфекции"
        )

    # ========== 5. Итоговый статус ==========
    fetal_status = (
        "все плохо"
        if fetal_risks
        and any(
            "Критическая" in r or "Продолжительная брадикардия" in r or "Острый прогноз" in r
            for r in fetal_risks
        )
        else "все хорошо"
        if not fetal_risks
        else "все средне"
    )

    maternal_status = (
        "все плохо" if hyperstimulation else "все средне" if hypotonia else "все хорошо"
    )

    priority = {"все плохо": 3, "все средне": 2, "все хорошо": 1}
    final_score = max(priority[fetal_status], priority[maternal_status])
    overall = {3: "все плохо", 2: "все средне", 1: "все хорошо"}[final_score]
    report["overall_status"] = overall

    # ========== 6. Рекомендации ==========
    recs = []
    if overall == "все плохо":
        recs = [
            "Немедленная оценка акушером-гинекологом",
            "Рассмотреть экстренное родоразрешение",
            "Отменить окситоцин (если применялся)",
            "Изменить положение матери (на левый бок)",
            "Начать внутривенную инфузию жидкости",
            "Подготовить операционную",
        ]
    elif overall == "все средне":
        recs = [
            "Усилить мониторинг (КТГ каждые 15–30 мин)",
            "Оценить температуру, АД, ЧСС матери",
            "Рассмотреть отмену/снижение окситоцина",
            "Изменить положение тела матери",
            "Контролировать кровопотерю и длительность родов",
        ]
    else:
        recs = [
            "Роды протекают благополучно",
            "Продолжать рутинный мониторинг",
            "Поддерживать комфорт матери",
        ]
    report["recommendations"] = recs

    # ========== 7. Формирование зон ==========
    report["attention_zones"] = fetal_ok + maternal_ok
    report["risk_zones"] = fetal_risks + maternal_risks

    # Убираем дубли и пустые строки
    report["attention_zones"] = [x for x in report["attention_zones"] if x]
    report["risk_zones"] = [x for x in report["risk_zones"] if x]

    return report

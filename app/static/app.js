const byId = id => document.getElementById(id);
const state = { conversations: [], checklists: [], run: null, evaluation: null, opened: new Map(), evaluationRequest: 0 };
const roles = { client: "Клиент", operator: "Оператор", bot: "Бот", system: "Система" };
const channels = { "agent chat": "Чат", "voice call": "Звонок" };
const statuses = { queued: "в очереди", running: "в работе", completed: "готово", failed: "ошибка" };
const number = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });

function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (className) node.className = className;
    return node;
}

function option(value, text) {
    const node = element("option", text);
    node.value = value;
    return node;
}

function button(text, handler) {
    const node = element("button", text, "secondary");
    node.type = "button";
    node.addEventListener("click", () => perform(handler, node));
    return node;
}

async function api(path, body) {
    const response = await fetch(path, body === undefined ? {} : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => null);
    if (response.ok && payload === null) throw new Error("Сервер вернул неожиданный ответ.");
    if (response.ok) return payload;
    const detail = payload?.detail;
    const errors = Array.isArray(detail) ? detail : detail?.errors;
    const message = Array.isArray(errors)
        ? errors.map(item => `${item.loc?.join(".") || "Данные"}: ${item.msg}`).join("; ")
        : detail;
    const error = new Error(typeof message === "string" ? message : `Ошибка запроса (${response.status})`);
    error.status = response.status;
    throw error;
}

function notice(message, error = false, target = byId("notice")) {
    target.hidden = !message;
    target.textContent = message;
    target.classList.toggle("error", error);
}

async function perform(action, trigger, target) {
    if (trigger) trigger.disabled = true;
    notice("", false, target);
    try {
        await action();
    } catch (error) {
        notice(error.message, true, target);
    } finally {
        if (trigger) trigger.disabled = false;
    }
}

function groupId() {
    const value = byId("group").value.trim();
    if (!value) throw new Error("Укажите группу диалогов.");
    return value;
}

async function refreshConversations(selectId = byId("conversation").value) {
    state.conversations = await api("/api/conversations");
    const select = byId("conversation");
    select.replaceChildren(...state.conversations.map(item => option(item.id,
        `${item.group_id} · ${item.conversation_id} · ${item.language} · ${channels[item.channel] || item.channel}`)));
    if (!state.conversations.length) select.append(option("", "Сначала загрузите диалоги"));
    if (state.conversations.some(item => String(item.id) === String(selectId))) {
        select.value = selectId;
    } else if (selectId && state.conversations.length) {
        select.prepend(option("", "Выберите диалог"));
        select.value = "";
    }
}

async function importConversations(body) {
    const result = await api("/api/imports", body);
    await refreshConversations(result.conversation_ids[0]);
    const restored = result.reactivated ? ` Восстановлено версий: ${result.reactivated}.` : "";
    notice(`Группа ${result.group_id}. Новых версий: ${result.imported}. Без изменений: ${result.unchanged}.${restored}`);
}

byId("import-organizers").addEventListener("click", event => perform(
    () => importConversations({ group_id: groupId(), source: "organizers" }), event.currentTarget));

byId("import-file").addEventListener("click", event => perform(async () => {
    const file = byId("file").files[0];
    if (!file) throw new Error("Выберите JSON-файл с диалогами.");
    let payload;
    try { payload = JSON.parse(await file.text()); }
    catch { throw new Error("В файле должен быть корректный JSON."); }
    const data = payload?.data || payload;
    const records = Array.isArray(data) ? data : data?.records;
    if (!Array.isArray(records)) throw new Error("Ожидается массив диалогов или объект с полем records.");
    const group = groupId();
    const fileGroups = [payload?.group_id, data?.dataset_group_id, data?.group_id]
        .filter(value => value !== undefined);
    if (fileGroups.some(value => typeof value !== "string" || !value.trim())) {
        throw new Error("В JSON указано некорректное имя группы.");
    }
    const otherGroup = fileGroups.find(value => value.trim() !== group);
    if (otherGroup) {
        throw new Error(`В файле указана группа «${otherGroup.trim()}». Выберите её перед загрузкой.`);
    }
    await importConversations({ group_id: group, source: "json", records });
}, event.currentTarget));

function selectedChecklist() {
    return state.checklists.find(item => String(item.id) === byId("checklist").value);
}

async function refreshChecklists(selectId = byId("checklist").value) {
    const latest = await api("/api/checklists");
    state.checklists = (await Promise.all(latest.map(item => api(`/api/checklists/${item.checklist_id}/versions`)))).flat();
    const select = byId("checklist");
    select.replaceChildren(...state.checklists.map(item => option(item.id, `${item.name} · v${item.version}`)));
    if (state.checklists.some(item => String(item.id) === String(selectId))) select.value = selectId;
    renderEditor();
}

function field(parent, title, name, control, value) {
    const label = element("label", title);
    control.dataset.field = name;
    control.value = value ?? "";
    label.append(control);
    parent.append(label);
    return control;
}

function input(type = "text", required = false) {
    const control = element("input");
    control.type = type;
    control.required = required;
    return control;
}

function textarea(rows) {
    const control = element("textarea");
    control.rows = rows;
    return control;
}

function appendScaleOption(parent, step = {}) {
    const row = element("div", null, "scale-option");
    const value = input("number", true);
    value.min = "0";
    value.max = String(Number.MAX_SAFE_INTEGER);
    field(row, "Значение", "option-value", value, step.value);
    const description = textarea(2);
    description.required = true;
    field(row, "Описание ступени", "option-description", description, step.description);
    row.append(button("Удалить ступень", () => row.remove()));
    parent.append(row);
}

function appendCriterion(criterion, expanded = false) {
    const row = element("details", null, "criterion");
    row.open = expanded;
    row.append(element("summary", criterion.name || "Новый критерий"));
    const grid = element("div", null, "editor-grid");
    field(grid, "ID", "id", input("text", true), criterion.id);
    const name = field(grid, "Название", "name", input("text", true), criterion.name);
    name.addEventListener("input", () => { row.firstElementChild.textContent = name.value || "Новый критерий"; });
    field(grid, "Секция", "section", input(), criterion.section);
    const type = element("select");
    type.append(option("boolean", "Да / Нет"), option("scale", "Шкала"), option("flag", "Флаг нарушения"));
    field(grid, "Тип ответа", "kind", type, criterion.kind || "boolean");
    const weight = input("number", true);
    weight.min = "0";
    weight.step = "any";
    field(grid, "Вес", "weight", weight, criterion.weight ?? 0);
    const naCondition = field(grid, "Условие N/A (если допустимо)", "na_condition", input(), criterion.na_condition);
    row.append(grid);
    const description = textarea(3);
    description.required = true;
    field(row, "Правило оценки", "description", description, criterion.description);
    const scale = element("fieldset", null, "scale-options");
    const steps = element("div");
    scale.append(element("legend", "Ступени шкалы"), steps, button("Добавить ступень", () => appendScaleOption(steps)));
    for (const step of criterion.options || []) appendScaleOption(steps, step);
    row.append(scale);
    let lastScoredWeight = weight.value;
    const updateType = () => {
        const flag = type.value === "flag";
        scale.disabled = scale.hidden = type.value !== "scale";
        naCondition.parentElement.hidden = flag;
        if (flag && !weight.readOnly) {
            lastScoredWeight = weight.value;
            weight.value = "0";
        } else if (!flag && weight.readOnly) {
            weight.value = lastScoredWeight;
        }
        weight.readOnly = flag;
    };
    updateType();
    type.addEventListener("change", updateType);
    row.append(button("Удалить критерий", () => row.remove()));
    row.addEventListener("invalid", () => { row.open = true; }, true);
    byId("criteria-editor").append(row);
}

function renderEditor() {
    const checklist = selectedChecklist();
    byId("criteria-editor").replaceChildren();
    byId("editor-version").textContent = checklist ? `Редактируется версия v${checklist.version}` : "Чек-лист недоступен";
    byId("checklist-name").value = checklist?.name || "";
    for (const criterion of checklist?.criteria || []) appendCriterion(criterion);
}

function readCriterion(row) {
    const value = name => row.querySelector(`[data-field=${name}]`).value.trim();
    const kind = value("kind");
    const options = kind !== "scale" ? [] : [...row.querySelectorAll(".scale-option")].map(step => ({
        value: step.querySelector("[data-field=option-value]").valueAsNumber,
        description: step.querySelector("[data-field=option-description]").value.trim(),
    }));
    return { id: value("id"), name: value("name"), section: value("section") || null, kind,
        weight: Number(value("weight")), description: value("description"),
        na_condition: kind === "flag" ? null : value("na_condition") || null, options };
}

byId("checklist").addEventListener("change", renderEditor);
byId("add-criterion").addEventListener("click", () => appendCriterion({}, true));
byId("checklist-form").addEventListener("submit", event => {
    event.preventDefault();
    perform(async () => {
        const checklist = selectedChecklist();
        if (!checklist) throw new Error("Выберите чек-лист.");
        const criteria = [...byId("criteria-editor").children].map(readCriterion);
        const result = await api(`/api/checklists/${checklist.checklist_id}/versions`, {
            name: byId("checklist-name").value.trim(), criteria,
        });
        await refreshChecklists(result.id);
        notice(`Сохранён чек-лист v${result.version}.`);
    }, event.submitter);
});

function renderRun(run) {
    const { completed, total, failed, queued, running } = run.counts;
    const status = byId("run-status");
    status.hidden = false;
    status.textContent = `Прогон №${run.id}. Готово: ${completed} из ${total}. В очереди: ${queued}. В работе: ${running}. Ошибок: ${failed}.`;
    byId("jobs").replaceChildren(...run.jobs.map(job => {
        const conversation = state.conversations.find(item => item.id === job.conversation_id);
        const row = element("li", `${conversation?.conversation_id || job.conversation_id} · ${statuses[job.status] || job.status}`);
        if (job.evaluation_id) row.append(button("Открыть результат", () => openEvaluation(job.evaluation_id)));
        if (job.error) row.append(element("div", job.error, "meta"));
        return row;
    }));
}

function updatePageParams(params) {
    const url = new URL(location.href);
    for (const [name, value] of Object.entries(params)) {
        if (value == null) url.searchParams.delete(name);
        else url.searchParams.set(name, value);
    }
    history.replaceState(null, "", url);
}

async function pollRun(watch, initial) {
    if (state.run !== watch) return;
    const errorBox = byId("run-error");
    let pending;
    try {
        const run = initial ?? await api(`/api/evaluation-runs/${watch.id}`);
        if (state.run !== watch) return;
        renderRun(run);
        const first = run.jobs.find(job => job.evaluation_id);
        if (first && !watch.opened) {
            await openEvaluation(first.evaluation_id, watch);
            watch.opened = true;
        }
        if (state.run !== watch) return;
        notice("", false, errorBox);
        watch.delay = 950;
        pending = run.counts.queued || run.counts.running;
    } catch (error) {
        if (state.run !== watch) return;
        pending = !error.status || [408, 429].includes(error.status) || error.status >= 500;
        notice(pending ? `Не удалось обновить прогон. Повторим запрос. ${error.message}` : error.message, true, errorBox);
        watch.delay = Math.min(watch.delay * 2, 8000);
    }
    if (pending) watch.timer = setTimeout(() => pollRun(watch), watch.delay);
}

function watchRun(id, { initial, opened = false } = {}) {
    clearTimeout(state.run?.timer);
    const watch = { id, opened, delay: 950, timer: null };
    state.run = watch;
    notice("", false, byId("run-error"));
    if (!initial) {
        byId("run-status").hidden = false;
        byId("run-status").textContent = `Загружается прогон №${id}.`;
    }
    void pollRun(watch, initial);
}

async function startRun(batch) {
    const checklist = selectedChecklist();
    if (!checklist) throw new Error("Выберите чек-лист.");
    const conversation = Number(byId("conversation").value);
    if (!batch && !conversation) throw new Error("Выберите диалог.");
    const scope = batch ? { group_id: groupId() } : { conversation_ids: [conversation] };
    const run = await api("/api/evaluation-runs", { checklist_version_id: checklist.id, ...scope });
    updatePageParams({ run: run.id, evaluation: null });
    watchRun(run.id, { initial: run });
}

byId("run-one").addEventListener("click", event => perform(() => startRun(false), event.currentTarget));
byId("run-group").addEventListener("click", event => perform(() => startRun(true), event.currentTarget));

function answerText(answer) {
    if (typeof answer === "boolean") return answer ? "Да" : "Нет";
    return answer === "N/A" ? "N/A" : number.format(answer);
}

function scoreText(total) {
    return total == null ? "—" : number.format(total);
}

function renderTranscript(conversation, transcriptId = "transcript", audioId = "audio") {
    byId(transcriptId).replaceChildren(...conversation.transcript.map(message => {
        const row = element("li", null, "message");
        if (transcriptId === "transcript") row.id = `message-${message.id}`;
        row.tabIndex = -1;
        row.dataset.speaker = message.speaker;
        row.append(element("div", `${message.timestamp} · ${roles[message.speaker] || message.speaker}`, "meta"),
            element("div", message.text));
        return row;
    }));
    const audio = byId(audioId);
    audio.hidden = true;
    audio.removeAttribute("href");
    if (conversation.audio_url && /^https?:\/\//i.test(conversation.audio_url)) {
        audio.href = conversation.audio_url;
        audio.hidden = false;
    }
}

function renderPreview(conversation) {
    byId("preview-meta").textContent = `${conversation.group_id} · ${conversation.conversation_id} · ${conversation.language}`;
    renderTranscript(conversation, "preview-transcript", "preview-audio");
}

byId("preview-conversation").addEventListener("click", event => perform(async () => {
    const id = Number(byId("conversation").value);
    if (!id) throw new Error("Выберите диалог.");
    const [conversation, versions] = await Promise.all([
        api(`/api/conversations/${id}`), api(`/api/conversations/${id}/versions`),
    ]);
    const version = byId("preview-version");
    version.replaceChildren(...versions.map(item => option(item.id, `v${item.version}`)));
    version.value = id;
    notice("", false, byId("preview-error"));
    renderPreview(conversation);
    byId("preview-dialog").showModal();
}, event.currentTarget));

byId("preview-version").addEventListener("change", event => {
    const select = event.currentTarget;
    perform(async () => {
        const id = select.value;
        renderPreview(await api(`/api/conversations/${id}`));
    }, select, byId("preview-error"));
});

function evidenceLink(message) {
    const quote = message.text.length > 130 ? `${message.text.slice(0, 127)}…` : message.text;
    const link = element("a", `${message.timestamp} · ${quote}`);
    link.href = `#message-${message.id}`;
    link.addEventListener("click", event => {
        event.preventDefault();
        const target = byId(`message-${message.id}`);
        byId("transcript").querySelector(".highlight")?.classList.remove("highlight");
        target.classList.add("highlight");
        target.focus();
        target.scrollIntoView({ block: "nearest" });
    });
    return link;
}

function renderResult(criterion, evaluation) {
    const { original, final, appeals } = evaluation;
    const answer = original.answers.find(item => item.criterion_id === criterion.id);
    const finalAnswer = final.answers.find(item => item.criterion_id === criterion.id);
    const points = final.score.criteria.find(item => item.criterion_id === criterion.id);
    const history = appeals.filter(item => item.criterion_id === criterion.id);
    const row = element("article", null, "result");
    row.append(element("h4", criterion.name));
    const answerLine = answer.answer === finalAnswer.answer ? answerText(answer.answer)
        : `${answerText(answer.answer)} → ${answerText(finalAnswer.answer)}`;
    row.append(element("p", `${answerLine} · вес ${number.format(criterion.weight)}${points?.points == null ? "" : ` · ${number.format(points.points)} балла`}`, "meta"));
    row.append(element("p", answer.reason));
    const evidence = answer.evidence_messages || [];
    if (evidence.length) {
        const items = element("ul");
        const cited = evidence.length > 5 ? [evidence[0], evidence.at(-1)] : evidence;
        for (const message of cited) {
            const item = element("li");
            item.append(evidenceLink(message));
            items.append(item);
        }
        row.append(items);
    }
    for (const appeal of history) {
        const item = element("div", null, "appeal-history");
        item.append(element("p", `Апелляция: ${appeal.comment}`));
        if (appeal.status === "open") item.append(button("Рассмотреть апелляцию", () => showAppeal(criterion, appeal)));
        else item.append(element("p", `${appeal.decision === "change" ? "Оценка изменена" : "Оценка подтверждена"}. ${appeal.reason}`));
        row.append(item);
    }
    if (!history.some(appeal => appeal.status === "open")) {
        row.append(button("Оспорить критерий", () => showAppeal(criterion)));
    }
    return row;
}

function renderEvaluation(evaluation) {
    const { conversation, checklist, original, final } = evaluation;
    byId("review").hidden = false;
    byId("review-meta").textContent = `Оценка №${evaluation.id} · ${conversation.conversation_id} · ${checklist.name} v${checklist.version}`;
    byId("scores").replaceChildren(...[["Исходный балл", original.score.total], ["Итоговый балл", final.score.total]].map(([label, total]) => {
        const box = element("div", label, "score");
        box.append(element("strong", scoreText(total)));
        return box;
    }));
    const flags = final.score.gross_flags.map(id => checklist.criteria.find(item => item.id === id)?.name || id);
    notice(flags.length ? `Грубые нарушения: ${flags.join(", ")}` : "", true, byId("gross-flags"));
    const sections = byId("section-scores");
    sections.hidden = !final.score.sections.length;
    sections.textContent = final.score.sections.map(item => `${item.section}: ${scoreText(item.total)}`).join(" · ");
    renderTranscript(conversation);
    byId("results").replaceChildren(...checklist.criteria.map(criterion => renderResult(criterion, evaluation)));
}

async function openEvaluation(id, watch) {
    const request = ++state.evaluationRequest;
    const evaluation = await api(`/api/evaluations/${id}`);
    if (request !== state.evaluationRequest || (watch && state.run !== watch)) return;
    state.evaluation = evaluation;
    renderEvaluation(state.evaluation);
    state.opened.set(id, state.evaluation.checklist.version);
    byId("opened-evaluations").replaceChildren(...[...state.opened].map(([evaluationId, version]) =>
        button(`Оценка №${evaluationId} · v${version}`, () => openEvaluation(evaluationId))));
    updatePageParams({ evaluation: id });
}

function showAppeal(criterion, appeal) {
    const dialog = byId("appeal-dialog");
    dialog.dataset.criterionId = criterion.id;
    dialog.dataset.appealId = appeal?.id || "";
    dialog.dataset.evaluationId = state.evaluation.id;
    byId("appeal-title").textContent = `${appeal ? "Решение" : "Апелляция"} · ${criterion.name}`;
    byId("appeal-comment-label").textContent = appeal ? "Причина решения" : "Комментарий";
    byId("appeal-comment").value = "";
    byId("decision").value = "confirm";
    byId("decision-field").hidden = !appeal;
    byId("answer-field").hidden = true;
    const values = criterion.kind === "scale" ? criterion.options.map(item => item.value) : [true, false];
    if (criterion.na_condition) values.push("N/A");
    byId("answer").replaceChildren(...values.map(value => option(JSON.stringify(value), answerText(value))));
    notice("", false, byId("appeal-error"));
    dialog.showModal();
}

byId("decision").addEventListener("change", event => { byId("answer-field").hidden = event.target.value !== "change"; });
byId("cancel-appeal").addEventListener("click", () => byId("appeal-dialog").close());
byId("appeal-form").addEventListener("submit", event => {
    event.preventDefault();
    const dialog = byId("appeal-dialog");
    perform(async () => {
        const { criterionId, appealId, evaluationId } = dialog.dataset;
        const comment = byId("appeal-comment").value.trim();
        if (!comment) throw new Error("Заполните комментарий или причину решения.");
        if (appealId) {
            const decision = byId("decision").value;
            const body = { decision, reason: comment };
            if (decision === "change") body.answer = JSON.parse(byId("answer").value);
            await api(`/api/appeals/${appealId}/decision`, body);
        } else {
            await api(`/api/evaluations/${evaluationId}/appeals`, { criterion_id: criterionId, comment });
        }
        await openEvaluation(Number(evaluationId));
        dialog.close();
    }, event.submitter, byId("appeal-error"));
});

perform(async () => {
    const params = new URLSearchParams(location.search);
    const id = Number(params.get("evaluation"));
    const runId = Number(params.get("run"));
    await Promise.all([refreshConversations(), refreshChecklists()]);
    if (Number.isSafeInteger(runId) && runId > 0) watchRun(runId, { opened: id > 0 });
    if (Number.isSafeInteger(id) && id > 0) await openEvaluation(id);
});

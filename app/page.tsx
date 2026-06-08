"use client";

import {
  BookOpenText,
  Download,
  FileText,
  Highlighter,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Send,
  Square,
  Upload
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  CustomAgentInput,
  DiscussionMessage,
  DiscussionSettings,
  Note,
  SubmitAnswers,
  SummaryMode,
  UserBehaviorLog
} from "@/lib/types";

type FixedAgentPreview = {
  id: string;
  name: string;
  kind: "skill";
  source: string;
  shortRole: string;
};

const CUSTOM_AGENT_COUNT = 8;
const PARTICIPANT_COUNT = 6;

const defaultCustomAgents: CustomAgentInput[] = Array.from({ length: CUSTOM_AGENT_COUNT }, (_, index) => ({
  name: `自定义 Agent ${index + 1}`,
  role: ""
}));

const submitLabels: Array<{ key: keyof SubmitAnswers; label: string }> = [
  { key: "userNeed", label: "用户需求" },
  { key: "designProblem", label: "设计问题" },
  { key: "designDirection", label: "设计方向" }
];

function normalizeSnippet(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

function renderHighlightedText(text: string, snippets: string[]) {
  const uniqueSnippets = Array.from(new Set(snippets.map(normalizeSnippet).filter(Boolean))).sort(
    (left, right) => right.length - left.length
  );

  let segments: Array<{ text: string; highlighted: boolean }> = [{ text, highlighted: false }];
  for (const snippet of uniqueSnippets) {
    segments = segments.flatMap((segment) => {
      if (segment.highlighted || !segment.text.includes(snippet)) {
        return [segment];
      }

      const pieces = segment.text.split(snippet);
      return pieces.flatMap((piece, index) => {
        const next = [{ text: piece, highlighted: false }];
        if (index < pieces.length - 1) {
          next.push({ text: snippet, highlighted: true });
        }
        return next;
      });
    });
  }

  return segments.map((segment, index) =>
    segment.highlighted ? (
      <mark key={`${segment.text}-${index}`}>{segment.text}</mark>
    ) : (
      <span key={`${segment.text}-${index}`}>{segment.text}</span>
    )
  );
}

function parseFinalInsightGroups(summary: string) {
  const groups = [
    { key: "userNeed", title: "用户需求", items: [] as string[] },
    { key: "designProblem", title: "设计问题", items: [] as string[] },
    { key: "designDirection", title: "设计方向", items: [] as string[] }
  ];

  let currentIndex = -1;
  for (const rawLine of summary.split(/\n+/)) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }

    const headingIndex = groups.findIndex((group) => line.includes(group.title));
    if (headingIndex >= 0 && line.length <= 12) {
      currentIndex = headingIndex;
      continue;
    }

    const cleaned = line.replace(/^\s*(?:[-*]|\d+[、.．]|\(\d+\))\s*/, "").trim();
    if (!cleaned) {
      continue;
    }

    if (headingIndex >= 0) {
      const inlineContent = cleaned.replace(groups[headingIndex].title, "").replace(/^[:：]\s*/, "").trim();
      if (inlineContent) {
        groups[headingIndex].items.push(inlineContent);
      }
      currentIndex = headingIndex;
      continue;
    }

    groups[currentIndex >= 0 ? currentIndex : 2].items.push(cleaned);
  }

  if (groups.every((group) => group.items.length === 0) && summary.trim()) {
    groups[2].items.push(summary.trim());
  }

  return groups;
}

function isDiscussionMessage(value: unknown): value is DiscussionMessage {
  if (!value || typeof value !== "object") {
    return false;
  }

  const message = value as Partial<DiscussionMessage>;
  return (
    typeof message.id === "string" &&
    typeof message.agentId === "string" &&
    typeof message.agentName === "string" &&
    (message.kind === "skill" || message.kind === "custom" || message.kind === "summary") &&
    typeof message.round === "number" &&
    typeof message.turnIndex === "number" &&
    typeof message.content === "string" &&
    typeof message.createdAt === "string"
  );
}

export default function Home() {
  const [fixedAgents, setFixedAgents] = useState<FixedAgentPreview[]>([]);
  const [customAgents, setCustomAgents] = useState<CustomAgentInput[]>(defaultCustomAgents);
  const [briefText, setBriefText] = useState("");
  const [briefFileName, setBriefFileName] = useState("");
  const [initialPrompt, setInitialPrompt] = useState("");
  const [messages, setMessages] = useState<DiscussionMessage[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [userLogs, setUserLogs] = useState<UserBehaviorLog[]>([]);
  const [highlights, setHighlights] = useState<Record<string, string[]>>({});
  const [summary, setSummary] = useState("");
  const [finalSummary, setFinalSummary] = useState("");
  const [roundPaused, setRoundPaused] = useState(false);
  const [summarizedRounds, setSummarizedRounds] = useState<number[]>([]);
  const [roundCount, setRoundCount] = useState(3);
  const [speechesPerAgentPerRound, setSpeechesPerAgentPerRound] = useState(1);
  const [submissions, setSubmissions] = useState<SubmitAnswers>({
    userNeed: "",
    designProblem: "",
    designDirection: ""
  });
  const [currentTurn, setCurrentTurn] = useState(0);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [awaitingTurn, setAwaitingTurn] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [error, setError] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let mounted = true;
    fetch("/api/agents/fixed")
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "读取固定 Agent 失败。");
        }
        if (mounted) {
          setFixedAgents(data.agents);
        }
      })
      .catch((fetchError: Error) => setError(fetchError.message));

    return () => {
      mounted = false;
    };
  }, []);

  const activeCustomAgents = useMemo(
    () =>
      customAgents
        .map((agent, index) => ({
          name: agent.name.trim() || `自定义 Agent ${index + 1}`,
          role: agent.role.trim()
        }))
        .filter((agent) => agent.role.length > 0),
    [customAgents]
  );

  const roster = useMemo(
    () => {
      const agents = [
        ...fixedAgents.slice(0, 4).map((agent) => ({ id: agent.id, name: agent.name, kind: "skill" as const })),
        ...activeCustomAgents.slice(0, Math.max(0, PARTICIPANT_COUNT - fixedAgents.length)).map((agent, index) => ({
          id: `custom-agent-${index + 1}`,
          name: agent.name,
          kind: "custom" as const
        }))
      ];

      while (agents.length < PARTICIPANT_COUNT) {
        const displayIndex = agents.length + 1;
        agents.push({
          id: `fallback-agent-${displayIndex}`,
          name: `补位 Agent ${displayIndex}`,
          kind: "custom" as const
        });
      }

      return agents.slice(0, PARTICIPANT_COUNT);
    },
    [activeCustomAgents, fixedAgents]
  );

  const settings: DiscussionSettings = useMemo(
    () => ({
      participantCount: PARTICIPANT_COUNT,
      roundCount,
      speechesPerAgentPerRound
    }),
    [roundCount, speechesPerAgentPerRound]
  );
  const turnsPerRound = PARTICIPANT_COUNT * speechesPerAgentPerRound;
  const totalTurns = PARTICIPANT_COUNT * roundCount * speechesPerAgentPerRound;
  const displayRound = Math.min(roundCount, Math.floor(currentTurn / turnsPerRound) + 1);
  const pausedRound = Math.max(1, Math.ceil(currentTurn / turnsPerRound));
  const settingsLocked = running || awaitingTurn || messages.length > 0 || summarizing;
  const currentRoundAlreadySummarized = summarizedRounds.includes(pausedRound);
  const readyToStart = fixedAgents.length === 4 && briefText.trim().length > 0 && initialPrompt.trim().length > 0;

  const progress = totalTurns > 0 ? Math.round((currentTurn / totalTurns) * 100) : 0;
  const activeAgent = roster.length > 0 ? roster[currentTurn % roster.length] : undefined;
  const finalInsightGroups = useMemo(() => parseFinalInsightGroups(finalSummary), [finalSummary]);

  const createUserLog = useCallback(
    (action: string, details: Record<string, unknown> = {}, roundOverride?: number): UserBehaviorLog => ({
      id: `log-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      createdAt: new Date().toISOString(),
      round: roundOverride ?? Math.max(1, Math.min(roundCount, pausedRound || displayRound)),
      turnIndex: currentTurn,
      action,
      details
    }),
    [currentTurn, displayRound, pausedRound, roundCount]
  );

  const addUserLog = useCallback(
    (action: string, details: Record<string, unknown> = {}, roundOverride?: number) => {
      const entry = createUserLog(action, details, roundOverride);
      setUserLogs((current) => [...current, entry]);
      return entry;
    },
    [createUserLog]
  );

  const groupLogsByRound = (logs: UserBehaviorLog[]) =>
    logs.reduce<Record<string, UserBehaviorLog[]>>((grouped, log) => {
      const key = `round_${log.round}`;
      grouped[key] = [...(grouped[key] ?? []), log];
      return grouped;
    }, {});

  const parseFile = async (file: File) => {
    setError("");
    setParsing(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/document/parse", {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "解析文档失败。");
      }
      setBriefText(data.text);
      setBriefFileName(data.fileName);
      addUserLog("upload_brief_document", {
        fileName: data.fileName,
        textLength: data.text.length
      });
    } catch (parseError) {
      setError(parseError instanceof Error ? parseError.message : "解析文档失败。");
    } finally {
      setParsing(false);
    }
  };

  const requestSummary = useCallback(
    async (mode: SummaryMode) => {
      setSummarizing(true);
      setError("");
      const targetRound = mode === "final" ? roundCount : pausedRound;
      addUserLog(
        mode === "final" ? "click_generate_final_summary" : "click_submit_notes_and_summarize",
        {
          noteCount: notes.length,
          messageCount: messages.length,
          highlightedNotes: notes.map((note) => ({
            text: note.text,
            agentName: note.agentName,
            messageId: note.messageId
          }))
        },
        targetRound
      );
      try {
        const response = await fetch("/api/discussion/summary", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            brief: briefText,
            prompt: initialPrompt,
            history: messages,
            notes,
            expectedTurns: totalTurns,
            mode,
            round: targetRound
          })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "总结失败。");
        }
        const summaryMessage: DiscussionMessage = {
          id: `summary-${mode}-${targetRound}-${Date.now()}`,
          agentId: "summary-agent",
          agentName: mode === "final" ? "总结 Agent：全部洞察" : `总结 Agent：第 ${targetRound} 轮 Harvest`,
          kind: "summary",
          round: targetRound,
          turnIndex: currentTurn,
          content: data.summary,
          createdAt: new Date().toISOString()
        };
        setMessages((current) => [...current, summaryMessage]);
        setSummary(data.summary);
        addUserLog(
          mode === "final" ? "final_summary_generated" : "round_summary_generated",
          {
            summary: data.summary,
            messageId: summaryMessage.id
          },
          targetRound
        );
        if (mode === "final") {
          setFinalSummary(data.summary);
          setRoundPaused(false);
          setPaused(false);
        } else {
          setSummarizedRounds((current) => Array.from(new Set([...current, targetRound])));
        }
      } catch (summaryError) {
        setError(summaryError instanceof Error ? summaryError.message : "总结失败。");
      } finally {
        setSummarizing(false);
      }
    },
    [addUserLog, briefText, currentTurn, initialPrompt, messages, notes, pausedRound, roundCount, totalTurns]
  );

  const runNextTurn = useCallback(async () => {
    if (!running || paused || awaitingTurn || currentTurn >= totalTurns) {
      return;
    }

    setAwaitingTurn(true);
    setError("");
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/discussion/step", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        signal: controller.signal,
        body: JSON.stringify({
          customAgents: activeCustomAgents,
          brief: briefText,
          prompt: initialPrompt,
          history: messages,
          turnIndex: currentTurn,
          settings
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "生成发言失败。");
      }

      if (!isDiscussionMessage(data.message)) {
        throw new Error("接口没有返回有效的 Agent 发言。");
      }

      const updatedHistory = [...messages, data.message];
      const nextTurn = currentTurn + 1;
      setMessages(updatedHistory);
      setCurrentTurn(nextTurn);

      if (nextTurn >= totalTurns || nextTurn % turnsPerRound === 0) {
        setRunning(false);
        setPaused(true);
        setRoundPaused(true);
        addUserLog(
          "round_auto_paused",
          {
            completedTurn: nextTurn,
            totalTurns,
            reason: nextTurn >= totalTurns ? "all_rounds_completed" : "round_completed"
          },
          Math.ceil(nextTurn / turnsPerRound)
        );
      }
    } catch (turnError) {
      if (turnError instanceof DOMException && turnError.name === "AbortError") {
        return;
      }
      setRunning(false);
      setError(turnError instanceof Error ? turnError.message : "生成发言失败。");
    } finally {
      setAwaitingTurn(false);
      abortRef.current = null;
    }
  }, [
    awaitingTurn,
    activeCustomAgents,
    addUserLog,
    briefText,
    currentTurn,
    initialPrompt,
    messages,
    paused,
    running,
    settings,
    turnsPerRound,
    totalTurns
  ]);

  useEffect(() => {
    if (running && !paused && !awaitingTurn && currentTurn < totalTurns) {
      const timeoutId = window.setTimeout(() => {
        void runNextTurn();
      }, 0);

      return () => window.clearTimeout(timeoutId);
    }
  }, [awaitingTurn, currentTurn, paused, runNextTurn, running, totalTurns]);

  const startRun = () => {
    if (!readyToStart) {
      setError("请确认 4 个固定 Skill Agent 已读取，并完成 brief 文档和初始 prompt。");
      return;
    }
    const startLog = createUserLog(
      "click_start_run",
      {
        settings,
        briefFileName,
        promptLength: initialPrompt.length,
        participatingAgents: roster.map((agent) => agent.name)
      },
      1
    );
    abortRef.current?.abort();
    setMessages([]);
    setNotes([]);
    setUserLogs((current) => [...current, startLog]);
    setHighlights({});
    setSummary("");
    setFinalSummary("");
    setRoundPaused(false);
    setSummarizedRounds([]);
    setCurrentTurn(0);
    setError("");
    setPaused(false);
    setRunning(true);
  };

  const pauseRun = () => {
    addUserLog("click_pause", { currentTurn, totalTurns });
    abortRef.current?.abort();
    setPaused(true);
  };

  const stopRun = () => {
    addUserLog("click_stop", { currentTurn, totalTurns });
    abortRef.current?.abort();
    setRunning(false);
    setPaused(false);
    setRoundPaused(false);
    setAwaitingTurn(false);
  };

  const resetRun = () => {
    const resetLog = createUserLog("click_reset", { previousTurn: currentTurn, previousMessageCount: messages.length });
    abortRef.current?.abort();
    setMessages([]);
    setNotes([]);
    setUserLogs([resetLog]);
    setHighlights({});
    setSummary("");
    setFinalSummary("");
    setRoundPaused(false);
    setSummarizedRounds([]);
    setCurrentTurn(0);
    setRunning(false);
    setPaused(false);
    setAwaitingTurn(false);
    setError("");
  };

  const continueNextRound = () => {
    if (currentTurn >= totalTurns) {
      return;
    }
    addUserLog("click_continue_next_round", { nextTurn: currentTurn + 1 }, Math.floor(currentTurn / turnsPerRound) + 1);
    setRoundPaused(false);
    setPaused(false);
    setRunning(true);
  };

  const canSubmitNotebookSummary =
    (roundPaused || paused) &&
    messages.length > 0 &&
    !awaitingTurn &&
    !summarizing &&
    !finalSummary &&
    !currentRoundAlreadySummarized;

  const captureSelection = () => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
      return;
    }

    const text = normalizeSnippet(selection.toString());
    if (!text) {
      return;
    }

    const range = selection.getRangeAt(0);
    const node = range.commonAncestorContainer;
    const element = node.nodeType === Node.ELEMENT_NODE ? (node as Element) : node.parentElement;
    const messageElement = element?.closest("[data-message-id]");
    const messageId = messageElement?.getAttribute("data-message-id");
    const sourceMessage = messages.find((message) => message.id === messageId);
    if (!messageId || !sourceMessage || !sourceMessage.content.includes(text)) {
      return;
    }

    setHighlights((current) => ({
      ...current,
      [messageId]: Array.from(new Set([...(current[messageId] ?? []), text]))
    }));
    setNotes((current) => [
      ...current,
      {
        id: `note-${Date.now()}`,
        text,
        messageId,
        agentName: sourceMessage.agentName,
        createdAt: new Date().toISOString()
      }
    ]);
    addUserLog(
      "highlight_note",
      {
        text,
        messageId,
        sourceAgent: sourceMessage.agentName,
        sourceRound: sourceMessage.round
      },
      sourceMessage.round
    );
    selection.removeAllRanges();
  };

  const exportResult = () => {
    const finalSubmissionLog = createUserLog(
      "final_submission_snapshot",
      {
        submissions,
        finalSummary,
        noteCount: notes.length,
        messageCount: messages.length
      },
      roundCount
    );
    const exportLog = createUserLog("click_export_result", { fileType: "experiment_result_json" }, roundCount);
    const logsForExport = [...userLogs, finalSubmissionLog, exportLog];
    setUserLogs(logsForExport);
    const payload = {
      exportedAt: new Date().toISOString(),
      briefFileName,
      brief: briefText,
      prompt: initialPrompt,
      fixedAgents,
      customAgents,
      activeCustomAgents,
      participatingAgents: roster,
      settings,
      messages,
      notes,
      summary,
      finalSummary,
      submissions,
      userBehaviorLogs: logsForExport,
      userBehaviorLogsByRound: groupLogsByRound(logsForExport)
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json;charset=utf-8"
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `design-agent-run-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const downloadSystemLog = () => {
    const downloadLog = createUserLog(
      "click_download_system_log",
      {
        fileType: "user_behavior_log_json",
        finalSubmissions: submissions,
        noteCount: notes.length,
        messageCount: messages.length
      },
      finalSummary ? roundCount : undefined
    );
    const logsForDownload = [...userLogs, downloadLog];
    setUserLogs(logsForDownload);
    const payload = {
      exportedAt: new Date().toISOString(),
      briefFileName,
      settings,
      currentTurn,
      totalTurns,
      finalSubmissions: submissions,
      logs: logsForDownload,
      logsByRound: groupLogsByRound(logsForDownload)
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json;charset=utf-8"
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `design-agent-user-log-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">LangChain / LangGraph</p>
          <h1>Design Brief Agent Lab</h1>
        </div>
        <div className="status-strip">
          <span>{currentTurn}/{totalTurns || 0}</span>
          <div className="progress-track" aria-label="讨论进度">
            <div style={{ width: `${progress}%` }} />
          </div>
          <span>
            {summarizing
              ? "总结中"
              : finalSummary
                ? "已完成"
                : roundPaused
                  ? "轮次暂停"
                  : running
                    ? paused
                      ? "已暂停"
                      : "讨论中"
                    : "待开始"}
          </span>
        </div>
      </header>

      <section className="workspace">
        <aside className="setup-panel" aria-label="实验设置">
          <div className="panel-heading">
            <h2>实验设置</h2>
            <button className="icon-button" type="button" title="重置实验" onClick={resetRun}>
              <RotateCcw size={18} />
            </button>
          </div>

          <section className="setup-section">
            <h3>固定 Skill Agent</h3>
            <div className="fixed-agent-list">
              {fixedAgents.map((agent) => (
                <div className="fixed-agent-row" key={agent.id}>
                  <strong>{agent.name}</strong>
                  <span>{agent.source}</span>
                </div>
              ))}
              {fixedAgents.length === 0 && <p className="muted">读取中</p>}
            </div>
          </section>

          <section className="setup-section">
            <h3>实验参数</h3>
            <div className="settings-grid">
              <label>
                <span>参与 Agent 数</span>
                <input value={PARTICIPANT_COUNT} readOnly aria-label="参与 Agent 数" />
              </label>
              <label>
                <span>轮次数</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={roundCount}
                  disabled={settingsLocked}
                  aria-label="轮次数"
                  onChange={(event) => setRoundCount(Math.min(10, Math.max(1, Number(event.target.value) || 1)))}
                />
              </label>
              <label>
                <span>每轮每 Agent 发言次数</span>
                <input
                  type="number"
                  min={1}
                  max={5}
                  value={speechesPerAgentPerRound}
                  disabled={settingsLocked}
                  aria-label="每轮每 Agent 发言次数"
                  onChange={(event) =>
                    setSpeechesPerAgentPerRound(Math.min(5, Math.max(1, Number(event.target.value) || 1)))
                  }
                />
              </label>
            </div>
          </section>

          <section className="setup-section">
            <h3>自定义 Agent</h3>
            <div className="custom-agent-list">
              {customAgents.map((agent, index) => (
                <div className="agent-editor" key={index}>
                  <input
                    value={agent.name}
                    aria-label={`自定义 Agent ${index + 1} 名称`}
                    onChange={(event) => {
                      const next = [...customAgents];
                      next[index] = { ...next[index], name: event.target.value };
                      setCustomAgents(next);
                    }}
                  />
                  <textarea
                    value={agent.role}
                    rows={3}
                    placeholder="角色设定"
                    aria-label={`自定义 Agent ${index + 1} 角色设定`}
                    onChange={(event) => {
                      const next = [...customAgents];
                      next[index] = { ...next[index], role: event.target.value };
                      setCustomAgents(next);
                    }}
                  />
                </div>
              ))}
            </div>
          </section>
        </aside>

        <section className="main-panel">
          <div className="input-band">
            <div
              className={`dropzone ${dragging ? "is-dragging" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                const file = event.dataTransfer.files[0];
                if (file) {
                  void parseFile(file);
                }
              }}
              role="button"
              tabIndex={0}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  fileInputRef.current?.click();
                }
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.md,.pdf,.docx"
                hidden
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    void parseFile(file);
                  }
                }}
              />
              {parsing ? <Loader2 className="spin" size={22} /> : <Upload size={22} />}
              <div>
                <strong>{briefFileName || "拖入 Brief 文档"}</strong>
                <span>{briefText ? `${briefText.length} 字符` : ".txt / .md / .pdf / .docx"}</span>
              </div>
            </div>

            <label className="prompt-box">
              <span>初始 Prompt</span>
              <textarea
                value={initialPrompt}
                rows={4}
                placeholder="输入本次讨论的目标、关注点或输出偏好"
                onChange={(event) => setInitialPrompt(event.target.value)}
              />
            </label>
          </div>

          <div className="runbar">
            <div>
              <strong>{activeAgent ? activeAgent.name : "等待 Agent"}</strong>
              <span>
                第 {displayRound}/{roundCount} 轮 · {Math.min(currentTurn + 1, totalTurns || 1)}/{totalTurns || 0}
              </span>
            </div>
            <div className="run-actions">
              {!running && !roundPaused && !finalSummary ? (
                <button className="primary-button" type="button" onClick={startRun} disabled={!readyToStart || parsing}>
                  <Play size={18} />
                  开始
                </button>
              ) : running && paused ? (
                <button className="primary-button" type="button" onClick={() => setPaused(false)}>
                  <Play size={18} />
                  继续
                </button>
              ) : running ? (
                <button className="secondary-button" type="button" onClick={pauseRun}>
                  <Pause size={18} />
                  暂停
                </button>
              ) : (
                <button className="secondary-button" type="button" disabled>
                  <Pause size={18} />
                  已暂停
                </button>
              )}
              <button className="icon-button" type="button" title="停止" onClick={stopRun} disabled={!running && !awaitingTurn}>
                <Square size={17} />
              </button>
            </div>
          </div>

          {error && <div className="error-bar">{error}</div>}

          <section className="discussion-panel" onMouseUp={captureSelection} aria-label="讨论记录">
            <div className="section-title">
              <BookOpenText size={18} />
              <h2>共享讨论空间</h2>
            </div>

            <div className="message-list">
              {messages.map((message) => (
                <article
                  className={`message-card ${message.kind === "summary" ? "is-summary" : ""}`}
                  key={message.id}
                  data-message-id={message.id}
                >
                  <div className="message-meta">
                    <strong>{message.agentName}</strong>
                    <span>
                      {message.kind === "summary"
                        ? `第 ${message.round} 轮 · 总结`
                        : `第 ${message.round} 轮 · ${message.turnIndex + 1}/${totalTurns}`}
                    </span>
                  </div>
                  <p>{renderHighlightedText(message.content, highlights[message.id] ?? [])}</p>
                </article>
              ))}

              {roundPaused && !finalSummary && (
                <article className="round-notice">
                  <div>
                    <strong>本轮次发言已结束</strong>
                    <span>请回顾发言内容后提交笔记生成总结。</span>
                  </div>
                  {currentTurn >= totalTurns ? (
                    <button className="primary-button" type="button" onClick={() => void requestSummary("final")} disabled={summarizing}>
                      <Send size={18} />
                      生成全部总结
                    </button>
                  ) : (
                    <button className="primary-button" type="button" onClick={continueNextRound} disabled={summarizing}>
                      <Play size={18} />
                      开启下一轮
                    </button>
                  )}
                </article>
              )}

              {awaitingTurn && (
                <article className="message-card is-loading">
                  <div className="message-meta">
                    <strong>{activeAgent?.name ?? "Agent"}</strong>
                    <span>生成中</span>
                  </div>
                  <p>
                    <Loader2 className="spin inline-icon" size={17} />
                    正在发言
                  </p>
                </article>
              )}

              {messages.length === 0 && !awaitingTurn && (
                <div className="empty-state">
                  <FileText size={24} />
                  <span>等待开始</span>
                </div>
              )}
            </div>
          </section>

          {finalSummary && (
            <section className="summary-panel final-insights">
              <div className="section-title">
                <Send size={18} />
                <h2>结构化洞察</h2>
              </div>
              <div className="insight-grid">
                {finalInsightGroups.map((group) => (
                  <article className="insight-card" key={group.key}>
                    <span>{group.title}</span>
                    <div>
                      {group.items.length > 0 ? (
                        group.items.map((item) => <p key={item}>{item}</p>)
                      ) : (
                        <p>暂无内容</p>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            </section>
          )}

          {finalSummary && (
            <section className="submission-panel">
              <div className="section-title">
                <Send size={18} />
                <h2>最终提交</h2>
              </div>
              <div className="submission-grid">
                {submitLabels.map((item) => (
                  <label key={item.key}>
                    <span>{item.label}</span>
                    <textarea
                      rows={4}
                      value={submissions[item.key]}
                      onChange={(event) =>
                        setSubmissions((current) => ({
                          ...current,
                          [item.key]: event.target.value
                        }))
                      }
                      onBlur={(event) =>
                        addUserLog(
                          "final_submission_field_updated",
                          {
                            field: item.key,
                            label: item.label,
                            value: event.currentTarget.value
                          },
                          roundCount
                        )
                      }
                    />
                  </label>
                ))}
              </div>
              <div className="export-actions">
                <button className="primary-button export-button" type="button" onClick={exportResult}>
                  <Download size={18} />
                  导出结果
                </button>
                <button className="secondary-button export-button" type="button" onClick={downloadSystemLog}>
                  <Download size={18} />
                  下载系统日志
                </button>
              </div>
            </section>
          )}
        </section>

        <aside className="notes-panel" aria-label="笔记本">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Notebook</p>
              <h2>笔记本</h2>
            </div>
            <Highlighter size={18} />
          </div>
          <button
            className="primary-button notebook-submit"
            type="button"
            onClick={() => void requestSummary("round")}
            disabled={!canSubmitNotebookSummary}
          >
            {summarizing ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            提交并总结
          </button>
          <button className="secondary-button notebook-submit" type="button" onClick={downloadSystemLog}>
            <Download size={18} />
            下载系统日志
          </button>
          <div className="note-list">
            {notes.map((note) => (
              <article className="note-card" key={note.id}>
                <strong>{note.agentName}</strong>
                <p>{note.text}</p>
              </article>
            ))}
            {notes.length === 0 && <p className="muted">选中讨论文本后记录</p>}
          </div>
        </aside>
      </section>
    </main>
  );
}

import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { HumanMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { Anthropic } from "@anthropic-ai/sdk";
import { ChatOpenAI } from "@langchain/openai";
import type {
  AgentConfig,
  DiscussionMessage,
  DiscussionSettings,
  Note,
  SummaryMode
} from "@/lib/types";
import { clipText, countCjkLikeChars, enforceMaxChars, trimModelText } from "@/lib/text";

const DiscussionState = Annotation.Root({
  agents: Annotation<AgentConfig[]>({
    reducer: (_left, right) => right,
    default: () => []
  }),
  brief: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => ""
  }),
  prompt: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => ""
  }),
  settings: Annotation<DiscussionSettings>({
    reducer: (_left, right) => right,
    default: () => ({
      participantCount: 6,
      roundCount: 3,
      speechesPerAgentPerRound: 1
    })
  }),
  history: Annotation<DiscussionMessage[]>({
    reducer: (_left, right) => right,
    default: () => []
  }),
  turnIndex: Annotation<number>({
    reducer: (_left, right) => right,
    default: () => 0
  }),
  message: Annotation<DiscussionMessage | null>({
    reducer: (_left, right) => right,
    default: () => null
  })
});

const SummaryState = Annotation.Root({
  brief: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => ""
  }),
  prompt: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => ""
  }),
  history: Annotation<DiscussionMessage[]>({
    reducer: (_left, right) => right,
    default: () => []
  }),
  notes: Annotation<Note[]>({
    reducer: (_left, right) => right,
    default: () => []
  }),
  mode: Annotation<SummaryMode>({
    reducer: (_left, right) => right,
    default: () => "round"
  }),
  summary: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => ""
  })
});

function getAnthropicModelName() {
  return (
    process.env.ANTHROPIC_MODEL ||
    process.env.ANTHROPIC_DEFAULT_SONNET_MODEL ||
    process.env.ANTHROPIC_DEFAULT_OPUS_MODEL ||
    process.env.ANTHROPIC_DEFAULT_HAIKU_MODEL ||
    "claude-sonnet-4-5-20250929"
  );
}

function normalizeCredential(value: string) {
  return value.replace(/^Bearer\s+/i, "").trim();
}

let openAiKeyCursor = 0;

function splitCredentialList(value: string | undefined) {
  return (value ?? "")
    .split(/[\n,;]+/)
    .map(normalizeCredential)
    .filter(Boolean);
}

function getNextOpenAIKey(keys = splitCredentialList(process.env.OPENAI_API_KEYS)) {
  if (keys.length === 0) {
    return process.env.OPENAI_API_KEY ? normalizeCredential(process.env.OPENAI_API_KEY) : "";
  }

  const key = keys[openAiKeyCursor % keys.length];
  openAiKeyCursor = (openAiKeyCursor + 1) % keys.length;
  return key;
}

function normalizeOpenAIBaseURL(value: string | undefined) {
  const clean = value?.replace(/\/+$/, "");
  if (!clean) {
    return undefined;
  }
  return clean.endsWith("/v1") ? clean : `${clean}/v1`;
}

function parseOpenAIApiKeys() {
  const raw = process.env.OPENAI_API_KEYS?.trim();
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed.map((item) => String(item).trim()).filter(Boolean);
    }
  } catch {
    // Plain delimiter-separated values are the expected local env format.
  }

  return splitCredentialList(raw);
}

function shouldUseOpenAICompatibleProxy(model: string) {
  const provider = (process.env.MODEL_PROVIDER || process.env.LLM_PROVIDER || process.env.ANTHROPIC_API_FORMAT || "")
    .toLowerCase()
    .trim();

  return provider === "openai-compatible" || provider === "openai" || /^gpt[-_]/i.test(model);
}

function getModel(options: { purpose?: "discussion" | "summary"; apiKeyIndex?: number } = {}): BaseChatModel {
  const openAiKeys = parseOpenAIApiKeys();
  const indexedOpenAiKey =
    typeof options.apiKeyIndex === "number" && options.apiKeyIndex >= 0 ? openAiKeys[options.apiKeyIndex] : undefined;
  const summaryOpenAiKey = options.purpose === "summary" ? openAiKeys[6] : undefined;
  const selectedOpenAiKey = indexedOpenAiKey || summaryOpenAiKey || getNextOpenAIKey(openAiKeys);
  const anthropicBaseUrl = process.env.ANTHROPIC_BASE_URL;
  const anthropicModel = getAnthropicModelName();
  const anthropicAuthToken = process.env.ANTHROPIC_AUTH_TOKEN
    ? normalizeCredential(process.env.ANTHROPIC_AUTH_TOKEN)
    : "";
  const anthropicApiKey = process.env.ANTHROPIC_API_KEY ? normalizeCredential(process.env.ANTHROPIC_API_KEY) : "";
  const anthropicCredential = anthropicApiKey || anthropicAuthToken;

  if (selectedOpenAiKey) {
    return new ChatOpenAI({
      apiKey: selectedOpenAiKey,
      model: process.env.OPENAI_MODEL || "gpt-4.1-mini",
      temperature: Number(process.env.OPENAI_TEMPERATURE ?? 0.7),
      configuration: process.env.OPENAI_BASE_URL
        ? {
            baseURL: normalizeOpenAIBaseURL(process.env.OPENAI_BASE_URL)
          }
        : undefined
    });
  }

  if (anthropicCredential && anthropicBaseUrl && shouldUseOpenAICompatibleProxy(anthropicModel)) {
    return new ChatOpenAI({
      apiKey: anthropicCredential,
      model: anthropicModel,
      temperature: Number(process.env.ANTHROPIC_TEMPERATURE ?? process.env.OPENAI_TEMPERATURE ?? 0.7),
      configuration: {
        baseURL: normalizeOpenAIBaseURL(anthropicBaseUrl)
      }
    });
  }

  if (anthropicApiKey) {
    return new ChatAnthropic({
      apiKey: anthropicApiKey,
      model: anthropicModel,
      temperature: Number(process.env.ANTHROPIC_TEMPERATURE ?? process.env.OPENAI_TEMPERATURE ?? 0.7),
      anthropicApiUrl: anthropicBaseUrl,
      maxTokens: Number(process.env.ANTHROPIC_MAX_TOKENS ?? 1400)
    });
  }

  if (anthropicAuthToken) {
    return new ChatAnthropic({
      model: anthropicModel,
      temperature: Number(process.env.ANTHROPIC_TEMPERATURE ?? process.env.OPENAI_TEMPERATURE ?? 0.7),
      maxTokens: Number(process.env.ANTHROPIC_MAX_TOKENS ?? 1400),
      createClient: (options) =>
        new Anthropic({
          ...options,
          apiKey: null,
          authToken: anthropicAuthToken,
          baseURL: anthropicBaseUrl ?? options.baseURL
        })
    });
  }

  if (!selectedOpenAiKey) {
    throw new Error(
      "缺少模型 API 配置。请在 .env.local 中配置 OPENAI_API_KEYS，或使用 OPENAI_API_KEY。"
    );
  }

  return new ChatOpenAI({
    apiKey: selectedOpenAiKey,
    model: process.env.OPENAI_MODEL || "gpt-4.1-mini",
    temperature: Number(process.env.OPENAI_TEMPERATURE ?? 0.7),
    configuration: process.env.OPENAI_BASE_URL
      ? {
          baseURL: normalizeOpenAIBaseURL(process.env.OPENAI_BASE_URL)
        }
      : undefined
  });
}

function formatHistory(history: DiscussionMessage[]) {
  if (history.length === 0) {
    return "暂无历史发言。";
  }

  return history
    .map((item, index) => {
      const order = item.kind === "summary" ? `总结${index + 1}` : `第${item.turnIndex + 1}条`;
      return `${order} / 第${item.round}轮 / ${item.agentName}：${item.content}`;
    })
    .join("\n");
}

function formatAgentRoster(agents: AgentConfig[]) {
  return agents.map((agent, index) => `${index + 1}. ${agent.name}（${agent.kind === "skill" ? "skill" : "自定义"}）`).join("\n");
}

function messageContentToString(message: BaseMessage) {
  if (typeof message.content === "string") {
    return message.content;
  }

  return message.content
    .map((part) => {
      if (typeof part === "string") {
        return part;
      }
      if ("text" in part && typeof part.text === "string") {
        return part.text;
      }
      return "";
    })
    .join("");
}

async function invokeAgentSpeech(params: {
  model: BaseChatModel;
  agents: AgentConfig[];
  speaker: AgentConfig;
  brief: string;
  prompt: string;
  history: DiscussionMessage[];
  turnIndex: number;
  settings: DiscussionSettings;
  retry?: boolean;
}) {
  const turnsPerRound = params.agents.length * params.settings.speechesPerAgentPerRound;
  const round = Math.floor(params.turnIndex / turnsPerRound) + 1;
  const speechPass = Math.floor((params.turnIndex % turnsPerRound) / params.agents.length) + 1;
  const totalTurns = turnsPerRound * params.settings.roundCount;
  const system = [
    `你是一个多 Agent 设计 brief 研讨系统中的发言者：${params.speaker.name}。`,
    `你与其他 ${Math.max(params.agents.length - 1, 0)} 个 Agent 位于同一个共享讨论空间，你能看到此前完整讨论记录。`,
    "你必须严格遵守：只输出一段中文发言；不要标题、编号、寒暄、自我介绍或 Markdown；长度不超过150个中文字符；不要重复已有观点；每次发言要推进设计洞察。",
    "你的发言必须基于此前完整讨论记录，承接、回应或推进前文至少一个具体观点。",
    `当前是第 ${round}/${params.settings.roundCount} 轮，本轮你第 ${speechPass}/${params.settings.speechesPerAgentPerRound} 次发言，也是全局第 ${params.turnIndex + 1}/${totalTurns} 条 Agent 发言。`,
    params.retry ? "上一版长度不符合要求，请重写为150个中文字符以内。" : "",
    "",
    "你的角色/skill 设定如下：",
    clipText(params.speaker.role, 14000)
  ]
    .filter(Boolean)
    .join("\n");

  const human = [
    "初始设计 brief：",
    clipText(params.brief, 12000),
    "",
    "用户初始 prompt：",
    clipText(params.prompt, 3000),
    "",
    "参与 Agent 列表：",
    formatAgentRoster(params.agents),
    "",
    "此前完整讨论记录：",
    clipText(formatHistory(params.history), 18000),
    "",
    `请以 ${params.speaker.name} 的身份继续发言。`
  ].join("\n");

  const response = await params.model.invoke([new SystemMessage(system), new HumanMessage(human)]);
  return trimModelText(messageContentToString(response));
}

export async function runDiscussionStep(input: {
  agents: AgentConfig[];
  brief: string;
  prompt: string;
  history: DiscussionMessage[];
  turnIndex: number;
  settings: DiscussionSettings;
}) {
  const workflow = new StateGraph(DiscussionState)
    .addNode("agent_turn", async (state) => {
      const turnsPerRound = state.agents.length * state.settings.speechesPerAgentPerRound;
      const speaker = state.agents[state.turnIndex % state.agents.length];
      const model = getModel({
        purpose: "discussion",
        apiKeyIndex: speaker.apiKeyIndex
      });
      const round = Math.floor(state.turnIndex / turnsPerRound) + 1;
      let content = await invokeAgentSpeech({
        model,
        agents: state.agents,
        speaker,
        brief: state.brief,
        prompt: state.prompt,
        history: state.history,
        turnIndex: state.turnIndex,
        settings: state.settings
      });

      const length = countCjkLikeChars(content);
      if (length > 150) {
        content = await invokeAgentSpeech({
          model,
          agents: state.agents,
          speaker,
          brief: state.brief,
          prompt: state.prompt,
          history: state.history,
          turnIndex: state.turnIndex,
          settings: state.settings,
          retry: true
        });
      }

      const message: DiscussionMessage = {
        id: `turn-${state.turnIndex + 1}-${Date.now()}`,
        agentId: speaker.id,
        agentName: speaker.name,
        kind: speaker.kind,
        round,
        turnIndex: state.turnIndex,
        content: enforceMaxChars(content, 150),
        createdAt: new Date().toISOString()
      };

      return {
        message,
        history: [...state.history, message]
      };
    })
    .addEdge(START, "agent_turn")
    .addEdge("agent_turn", END)
    .compile();

  const result = await workflow.invoke(input);
  if (!result.message) {
    throw new Error("Agent 没有生成发言。");
  }
  return result.message;
}

export async function runSummary(input: {
  brief: string;
  prompt: string;
  history: DiscussionMessage[];
  notes: Note[];
  mode: SummaryMode;
  round?: number;
}) {
  const workflow = new StateGraph(SummaryState)
    .addNode("summary_agent", async (state) => {
      const model = getModel({ purpose: "summary" });
      const system =
        state.mode === "final"
          ? [
              "你负责将目前讨论空间内的所有文本转化为可直接给用户看的结构化设计洞察。",
              "请严格使用三个小标题：用户需求、设计问题、设计方向。",
              "三个小标题下合计输出四条洞察：用户需求1条，设计问题1条，设计方向2条。",
              "四条洞察总字符数必须500字以内，可直接给用户看。",
              "不要写成长段落，不要复述每个人发言，不要展示额外推理过程。"
            ].join("\n")
          : [
              "以上是目前讨论的内容和用户高亮的内容，请阅读并从中总结结束语。",
              "结束语的任务不是完整总结，而是让参与者看见本轮的pattern：共识、转变、隐藏观点、张力。",
              "用关键词式短句，300字以内，最多4行；不要写成长段落，不要复述每个人发言。"
            ].join("\n");

      const notesText =
        state.notes.length === 0
          ? "暂无用户高亮笔记。"
          : state.notes.map((note, index) => `${index + 1}. ${note.agentName}：${note.text}`).join("\n");

      const human = [
        `总结类型：${state.mode === "final" ? "全部轮次最终总结" : "本轮结束语"}`,
        "",
        "初始设计 brief：",
        clipText(state.brief, 12000),
        "",
        "用户初始 prompt：",
        clipText(state.prompt, 3000),
        "",
        "Agent 讨论记录：",
        clipText(formatHistory(state.history), 24000),
        "",
        "用户高亮笔记：",
        clipText(notesText, 8000)
      ].join("\n");

      const response = await model.invoke([new SystemMessage(system), new HumanMessage(human)]);
      const maxChars = state.mode === "final" ? 800 : 300;
      return {
        summary: enforceMaxChars(messageContentToString(response), maxChars)
      };
    })
    .addEdge(START, "summary_agent")
    .addEdge("summary_agent", END)
    .compile();

  const result = await workflow.invoke(input);
  return result.summary;
}

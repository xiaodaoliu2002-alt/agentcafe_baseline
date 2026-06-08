import { NextResponse } from "next/server";
import { runDiscussionStep } from "@/lib/agent-graph";
import { getFixedSkillAgents } from "@/lib/skill-agents";
import type { AgentConfig, CustomAgentInput, DiscussionMessage, DiscussionSettings } from "@/lib/types";

export const runtime = "nodejs";

const PARTICIPANT_COUNT = 6;

const FALLBACK_ROLES = [
  "从用户需求、使用情境和行为动机角度发言，帮助团队看见 brief 背后的真实人群与未被说出的需求。",
  "从设计策略、体验原型和落地验证角度发言，帮助团队把讨论转化为可推进的设计机会。"
];

function normalizeSettings(settings?: Partial<DiscussionSettings>): DiscussionSettings {
  return {
    participantCount: PARTICIPANT_COUNT,
    roundCount: Math.min(10, Math.max(1, Number(settings?.roundCount ?? 3))),
    speechesPerAgentPerRound: Math.min(5, Math.max(1, Number(settings?.speechesPerAgentPerRound ?? 1)))
  };
}

function normalizeCustomAgents(customAgents: CustomAgentInput[]) {
  return customAgents
    .slice(0, 8)
    .map((input, index) => {
      const role = input?.role?.trim() ?? "";

      return {
        id: `custom-agent-${index + 1}`,
        name: input?.name?.trim() || `自定义 Agent ${index + 1}`,
        role,
        kind: "custom" as const,
        shortRole: role.slice(0, 220)
      };
    })
    .filter((agent) => agent.role.length > 0);
}

function buildParticipatingAgents(fixedAgents: AgentConfig[], customAgents: ReturnType<typeof normalizeCustomAgents>) {
  const agents: AgentConfig[] = fixedAgents.slice(0, 4).map((agent, index) => ({
    ...agent,
    apiKeyIndex: index
  }));

  for (const customAgent of customAgents) {
    if (agents.length >= PARTICIPANT_COUNT) {
      break;
    }
    agents.push({
      ...customAgent,
      apiKeyIndex: agents.length
    });
  }

  while (agents.length < PARTICIPANT_COUNT) {
    const fallbackIndex = agents.length - fixedAgents.length;
    const displayIndex = agents.length + 1;
    const role = FALLBACK_ROLES[fallbackIndex] ?? FALLBACK_ROLES[FALLBACK_ROLES.length - 1];
    agents.push({
      id: `fallback-agent-${displayIndex}`,
      name: `补位 Agent ${displayIndex}`,
      kind: "custom",
      role,
      shortRole: role.slice(0, 220),
      apiKeyIndex: agents.length
    });
  }

  return agents;
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      customAgents: CustomAgentInput[];
      brief: string;
      prompt: string;
      history: DiscussionMessage[];
      turnIndex: number;
      settings?: Partial<DiscussionSettings>;
    };

    if (!body.brief?.trim()) {
      return NextResponse.json({ error: "请先拖入并解析 brief 文档。" }, { status: 400 });
    }
    if (!body.prompt?.trim()) {
      return NextResponse.json({ error: "请输入初始 prompt。" }, { status: 400 });
    }
    const fixedAgents = await getFixedSkillAgents();
    const customAgents = normalizeCustomAgents(body.customAgents ?? []);
    const settings = normalizeSettings(body.settings);
    const agents = buildParticipatingAgents(fixedAgents, customAgents);
    const totalTurns = agents.length * settings.roundCount * settings.speechesPerAgentPerRound;

    if (fixedAgents.length < 4) {
      return NextResponse.json({ error: "固定 Skill Agent 未读取完整。" }, { status: 500 });
    }
    if (body.turnIndex < 0 || body.turnIndex >= totalTurns) {
      return NextResponse.json({ error: "讨论轮次超出范围。" }, { status: 400 });
    }

    const message = await runDiscussionStep({
      agents,
      brief: body.brief,
      prompt: body.prompt,
      history: body.history ?? [],
      turnIndex: body.turnIndex,
      settings
    });

    return NextResponse.json({ message });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "生成 Agent 发言失败。"
      },
      { status: 500 }
    );
  }
}

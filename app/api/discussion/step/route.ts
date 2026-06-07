import { NextResponse } from "next/server";
import { runDiscussionStep } from "@/lib/agent-graph";
import { getFixedSkillAgents } from "@/lib/skill-agents";
import type { CustomAgentInput, DiscussionMessage } from "@/lib/types";

export const runtime = "nodejs";

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

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      customAgents: CustomAgentInput[];
      brief: string;
      prompt: string;
      history: DiscussionMessage[];
      turnIndex: number;
    };

    if (!body.brief?.trim()) {
      return NextResponse.json({ error: "请先拖入并解析 brief 文档。" }, { status: 400 });
    }
    if (!body.prompt?.trim()) {
      return NextResponse.json({ error: "请输入初始 prompt。" }, { status: 400 });
    }
    const fixedAgents = await getFixedSkillAgents();
    const customAgents = normalizeCustomAgents(body.customAgents ?? []);
    const agents = [...fixedAgents, ...customAgents];
    const totalTurns = agents.length * 3;

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
      turnIndex: body.turnIndex
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

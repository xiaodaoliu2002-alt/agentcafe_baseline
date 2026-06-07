import { NextResponse } from "next/server";
import { getFixedSkillAgents } from "@/lib/skill-agents";

export const runtime = "nodejs";

export async function GET() {
  try {
    const agents = await getFixedSkillAgents();
    return NextResponse.json({
      agents: agents.map((agent) => ({
        id: agent.id,
        name: agent.name,
        kind: agent.kind,
        source: agent.source,
        shortRole: agent.shortRole
      }))
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "读取 skill Agent 失败。"
      },
      { status: 500 }
    );
  }
}

import { NextResponse } from "next/server";
import { runSummary } from "@/lib/agent-graph";
import type { DiscussionMessage, Note } from "@/lib/types";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      brief: string;
      prompt: string;
      history: DiscussionMessage[];
      notes: Note[];
      expectedTurns?: number;
    };

    const expectedTurns = Math.max(1, body.expectedTurns ?? body.history?.length ?? 0);

    if (!body.history || body.history.length < expectedTurns) {
      return NextResponse.json({ error: `需要 ${expectedTurns} 条 Agent 发言后才能总结。` }, { status: 400 });
    }

    const summary = await runSummary({
      brief: body.brief,
      prompt: body.prompt,
      history: body.history,
      notes: body.notes ?? []
    });

    return NextResponse.json({ summary });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "生成总结失败。"
      },
      { status: 500 }
    );
  }
}

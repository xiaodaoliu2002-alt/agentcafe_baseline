import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Design Brief Agent Lab",
  description: "A shared-space multi-agent design brief discussion app."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import Providers from "./providers";

const inter = Inter({ 
  variable: "--font-inter", 
  subsets: ["latin"],
  display: "swap"
});
const jetBrainsMono = JetBrains_Mono({ 
  variable: "--font-jetbrains-mono", 
  subsets: ["latin"],
  display: "swap"
});

export const metadata: Metadata = {
  title: "DataVoyager",
  description: "Auth Microsoft (MSA) + Chat",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" suppressHydrationWarning>
      <body className={`${inter.variable} ${jetBrainsMono.variable} antialiased bg-[var(--bg)] text-[var(--text)]`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}

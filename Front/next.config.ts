import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    // Keep Turbopack's workspace boundary at this frontend project.
    root: __dirname,
  },
  async rewrites() {
    return [
      {
        source: "/rag/:path*",
        destination: "http://127.0.0.1:8765/:path*", // backend FastAPI
      },
    ];
  },
};

export default nextConfig;

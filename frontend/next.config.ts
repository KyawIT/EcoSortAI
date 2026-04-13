import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,

  // Don't leak server info in response headers
  poweredByHeader: false,

  // Strict mode for images (even though we use CSS backgrounds, good hygiene)
  images: {
    formats: ["image/avif", "image/webp"],
  },

  // Security headers
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Permissions-Policy",
            // camera is required for the live camera feature
            value: "camera=self, microphone=(), geolocation=()",
          },
        ],
      },
    ];
  },
};

export default nextConfig;

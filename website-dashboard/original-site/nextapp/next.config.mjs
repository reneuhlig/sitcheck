/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: { unoptimized: true },
  distDir: '.next-local',
};

export default nextConfig;

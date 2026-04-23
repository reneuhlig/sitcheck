FROM node:22-slim
WORKDIR /app
COPY apps/mcp-sitcheck/package*.json /app/
RUN npm install
COPY apps/mcp-sitcheck /app
COPY packages/shared/schemas /app/schemas
CMD ["node", "server.mjs"]

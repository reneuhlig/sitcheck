FROM node:22-alpine AS build
WORKDIR /app

ARG VITE_API_BASE_URL=
ARG VITE_DEFAULT_ZONE_ID=default-zone
ARG VITE_DEFAULT_HORIZON=60
ARG VITE_OLLAMA_MODEL=qwen2.5:0.5b
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_DEFAULT_ZONE_ID=$VITE_DEFAULT_ZONE_ID
ENV VITE_DEFAULT_HORIZON=$VITE_DEFAULT_HORIZON
ENV VITE_OLLAMA_MODEL=$VITE_OLLAMA_MODEL

COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

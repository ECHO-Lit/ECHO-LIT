# syntax=docker/dockerfile:1.7

# ── Install deps ───────────────────────────────────────────────
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ── Dev target: source arrives via bind-mount at runtime ───────
FROM node:20-alpine AS dev
WORKDIR /app
# Bring pre-built node_modules from deps stage.
# Container's node_modules is kept in a named volume so the host
# directory (potentially Windows-incompatible) never shadows it.
COPY --from=deps /app/node_modules ./node_modules
COPY package.json package-lock.json ./
EXPOSE 8080
CMD ["npm", "run", "dev"]

# ── Production build ───────────────────────────────────────────
FROM deps AS build
COPY . .
RUN npm run build

FROM nginx:alpine AS prod
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80

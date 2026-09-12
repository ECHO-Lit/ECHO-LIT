# syntax=docker/dockerfile:1.7

# ── Install deps ───────────────────────────────────────────────
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
# Cache mount persists npm's download cache across builds — only new/changed
# packages hit the network on subsequent package.json edits.
RUN --mount=type=cache,target=/root/.npm npm ci

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
# Vite bakes the API base into the bundle at build time, and .dockerignore
# keeps .env out of the context, so the production API is a build argument:
#   docker build --target prod --build-arg VITE_API_BASE_URL=https://api.example.org .
# Left empty, the client calls port 8000 on the host that served the page.
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
COPY . .
RUN npm run build

FROM nginx:alpine AS prod
# SPA fallback: client-side routes such as /j-lens are served index.html.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80

# CrediLens AI - frontend image (React/Vite build served by nginx)
#     docker build -t credilens-frontend ./frontend

# ---- Stage 1: build the static site ----
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# VITE_API_BASE_URL is not set -> the app calls the relative path `/api`,
# which nginx proxies to the backend container.
RUN npm run build

# ---- Stage 2: serve with nginx ----
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
